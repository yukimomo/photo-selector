from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from photo_selector.analyzer import (
	analyze_quality,
	apply_quality_corrections,
	compute_image_hash,
	encode_image_base64,
	get_image_info,
)
from photo_selector.audio_analyzer import AudioAnalysis, analyze_audio
from photo_selector.dedupe_utils import hash_to_int, is_near_duplicate
from photo_selector.frame_extractor import extract_representative_frame
from photo_selector.log_utils import log_event
from photo_selector.ollama_client import OllamaClient
from photo_selector.output_paths import (
	concat_list_path,
	digest_clips_source_dir,
	final_digest_path,
	get_video_paths,
	VideoOutputPaths,
)
from photo_selector.score_schema import normalize_analysis
from photo_selector.video_splitter import ClipInfo, collect_video_paths, split_video


SCHEMA_TEMPLATE = {
	"caption": "",
	"tags": [],
	"risks": {
		"blur": False,
		"dark": False,
		"overexposed": False,
		"out_of_focus": False,
	},
	"score": 0.0,
}
MIN_BRIGHTNESS_GATE = 15.0


@dataclass
class DigestResult:
	sources: List[Dict[str, Any]]
	job_state: Dict[str, Dict[str, Any]]


@dataclass
class JobContext:
	state: Dict[str, Dict[str, Any]] = field(
		default_factory=lambda: {
			"split": {},
			"score": {},
			"select": {},
			"concat": {},
		}
	)

	def record(self, step: str, key: str, status: str, error: str | None = None) -> None:
		entry: Dict[str, Any] = {"status": status}
		if error:
			entry["error"] = error
		self.state.setdefault(step, {})[key] = entry


def run_video_digest(
	input_path: Path,
	output_dir: Path,
	max_source_seconds: int,
	min_clip: int,
	max_clip: int,
	model: str,
	base_url: str,
	keep_temp: bool,
	delete_split_files: bool,
	preset: str,
	concat_in_digest_folder: bool,
	use_hwaccel: bool,
	dedupe_enabled: bool,
	dedupe_hamming_threshold: int,
	dedupe_scope: str,
	max_selected_clips: int,
	target_digest_seconds: int,
) -> DigestResult:
	output_dir.mkdir(parents=True, exist_ok=True)
	paths = get_video_paths(output_dir)
	temp_dir = paths.temp_dir
	clip_dir = temp_dir / "clips"
	frame_dir = temp_dir / "frames"
	audio_dir = temp_dir / "audio"

	video_paths = collect_video_paths(input_path)
	clips: list[ClipInfo] = []
	job = JobContext()
	for video_path in video_paths:
		video_clip_dir = clip_dir / video_path.stem
		try:
			clips.extend(
				split_video(
					video_path,
					video_clip_dir,
					min_clip,
					max_clip,
					use_hwaccel=use_hwaccel,
				)
			)
			job.record("split", str(video_path), "ok")
		except Exception as exc:  # noqa: BLE001
			message = str(exc)
			log_event(
				"plain",
				level="error",
				event_type="split_failed",
				file_path=str(video_path),
				message=message,
			)
			job.record("split", str(video_path), "failed", message)

	client = OllamaClient(base_url=base_url)
	clip_records: list[Dict[str, Any]] = []

	for clip in tqdm(clips, desc="Analyzing clips", unit="clip"):
		record: Dict[str, Any] = {
			"source_path": str(clip.source_path),
			"clip_path": str(clip.clip_path),
			"start": clip.start,
			"end": clip.end,
			"duration": clip.duration,
		}
		try:
			frame_path = frame_dir / clip.source_path.stem / f"{clip.clip_path.stem}.jpg"
			extract_representative_frame(clip.clip_path, frame_path)
			info = get_image_info(frame_path)
			frame_hash = compute_image_hash(frame_path)
			quality = analyze_quality(frame_path)
			image_b64 = encode_image_base64(frame_path)
			prompt = _build_prompt(quality)
			analysis = client.chat(model, image_b64, prompt)
			analysis = _validate_analysis(analysis)
			analysis["score"] = apply_quality_corrections(
				float(analysis["score"]),
				quality,
				info.width,
				info.height,
			)
			analysis["score"] = _apply_risk_penalties(
				float(analysis["score"]),
				analysis.get("risks", {}),
			)

			audio: AudioAnalysis | None = None
			try:
				audio = analyze_audio(clip.clip_path, audio_dir)
			except Exception:
				audio = None

			record.update(
				{
					"frame_path": str(frame_path),
					"frame_hash": f"{frame_hash:016x}",
					"clip_width": clip.width,
					"clip_height": clip.height,
					"clip_fps": clip.fps,
					"frame_width": info.width,
					"frame_height": info.height,
					"frame_orientation": info.orientation,
					"analysis": analysis,
					"quality": quality,
					"has_speech": audio.has_speech if audio else None,
					"audio_rms": audio.rms if audio else None,
					"score_final": float(analysis["score"]),
					"error": None,
				}
			)
			job.record("score", str(clip.clip_path), "ok")
		except Exception as exc:  # noqa: BLE001
			message = str(exc)
			record.update(
				{
					"analysis": None,
					"quality": None,
					"has_speech": None,
					"audio_rms": None,
					"score_final": None,
					"error": message,
				}
			)
			log_event(
				"plain",
				level="error",
				event_type="score_failed",
				file_path=str(clip.clip_path),
				message=message,
			)
			job.record("score", str(clip.clip_path), "failed", message)

		clip_records.append(record)

	source_results = _process_sources(
		clip_records,
		paths=paths,
		max_source_seconds=max_source_seconds,
		preset=preset,
		concat_in_digest_folder=concat_in_digest_folder,
		use_hwaccel=use_hwaccel,
		job=job,
		dedupe_enabled=dedupe_enabled,
		dedupe_hamming_threshold=dedupe_hamming_threshold,
		dedupe_scope=dedupe_scope,
		max_selected_clips=max_selected_clips,
		target_digest_seconds=target_digest_seconds,
	)

	_cleanup_temp_artifacts(
		output_dir=output_dir,
		clip_dir=clip_dir,
		temp_dir=temp_dir,
		keep_temp=keep_temp,
		delete_split_files=delete_split_files,
		sources=source_results,
		job_state=job.state,
	)

	return DigestResult(sources=source_results, job_state=job.state)


def _cleanup_temp_artifacts(
	*,
	output_dir: Path,
	clip_dir: Path,
	temp_dir: Path,
	keep_temp: bool,
	delete_split_files: bool,
	sources: List[Dict[str, Any]],
	job_state: Dict[str, Dict[str, Any]],
) -> None:
	log_event(
		"plain",
		level="info",
		event_type="cleanup_config",
		message="cleanup configuration",
		extra={
			"keep_temp": keep_temp,
			"delete_split_files": delete_split_files,
			"temp_dir": str(temp_dir),
			"clip_dir": str(clip_dir),
		},
	)

	if keep_temp:
		log_event(
			"plain",
			level="info",
			event_type="cleanup_skipped",
			message="cleanup skipped because keep_temp is enabled",
				extra={"temp_dir": str(temp_dir)},
		)
		return

	if _has_failures(sources, job_state):
		log_event(
			"plain",
			level="info",
			event_type="cleanup_skipped",
			message="cleanup skipped due to failures",
			extra={"temp_dir": str(temp_dir)},
		)
		return

	if not _is_safe_temp_dir(temp_dir, output_dir):
		log_event(
			"plain",
			level="warning",
			event_type="cleanup_skipped",
			message="cleanup skipped due to unsafe temp path",
			extra={"temp_dir": str(temp_dir), "output_dir": str(output_dir)},
		)
		return

	clip_files = _collect_files(clip_dir) if clip_dir.exists() else []
	other_files = _collect_files(temp_dir)
	other_files = [path for path in other_files if not _is_relative_to(path, clip_dir)]

	deleted_files, failed_files = _delete_files_with_retries(clip_files + other_files)
	log_event(
		"plain",
		level="info",
		event_type="cleanup_files",
		message="cleanup file results",
		extra={
			"targeted": len(clip_files) + len(other_files),
			"deleted": deleted_files,
			"failed": len(failed_files),
			"failed_items": failed_files,
		},
	)

	removed_dirs, skipped_dirs = _remove_empty_dirs(
		directories=_sorted_dirs_descending([clip_dir, temp_dir]),
	)
	log_event(
		"plain",
		level="info",
		event_type="cleanup_dirs",
		message="cleanup directory results",
		extra={
			"removed": removed_dirs,
			"skipped": skipped_dirs,
		},
	)


def _has_failures(
	sources: List[Dict[str, Any]],
	job_state: Dict[str, Dict[str, Any]],
) -> bool:
	for source in sources:
		if source.get("error"):
			return True
	for step_entries in job_state.values():
		if not isinstance(step_entries, dict):
			continue
		for entry in step_entries.values():
			if isinstance(entry, dict) and entry.get("status") == "failed":
				return True
	return False


def _is_safe_temp_dir(temp_dir: Path, output_dir: Path) -> bool:
	if temp_dir.name != "temp":
		return False
	try:
		temp_resolved = temp_dir.resolve(strict=False)
		output_resolved = output_dir.resolve(strict=False)
		return temp_resolved.is_relative_to(output_resolved)
	except Exception:
		return False


def _is_relative_to(path: Path, base: Path) -> bool:
	try:
		return path.resolve(strict=False).is_relative_to(base.resolve(strict=False))
	except Exception:
		return False


def _collect_files(root: Path) -> list[Path]:
	if not root.exists():
		return []
	return [path for path in root.rglob("*") if path.is_file()]


def _delete_files_with_retries(paths: list[Path]) -> tuple[int, list[dict[str, str]]]:
	deleted = 0
	failed: list[dict[str, str]] = []
	for path in paths:
		result = _retry_remove_file(path)
		if result is None:
			deleted += 1
		else:
			failed.append({"path": str(path), "reason": result})
	return deleted, failed


def _retry_remove_file(path: Path, retries: int = 3, delay_seconds: float = 0.2) -> str | None:
	for attempt in range(retries):
		try:
			path.unlink()
			return None
		except FileNotFoundError:
			return "not found"
		except Exception as exc:  # noqa: BLE001
			if attempt == retries - 1:
				return str(exc)
			time.sleep(delay_seconds)
	return "unknown error"


def _remove_empty_dirs(
	*,
	directories: list[Path],
) -> tuple[list[str], list[dict[str, str]]]:
	removed: list[str] = []
	skipped: list[dict[str, str]] = []
	for directory in directories:
		reason = _retry_remove_dir(directory)
		if reason is None:
			removed.append(str(directory))
		else:
			skipped.append({"path": str(directory), "reason": reason})
	return removed, skipped


def _retry_remove_dir(
	path: Path,
	retries: int = 3,
	delay_seconds: float = 0.2,
) -> str | None:
	for attempt in range(retries):
		if not path.exists():
			return "not found"
		if not _is_dir_empty(path):
			return "not empty"
		try:
			path.rmdir()
			return None
		except Exception as exc:  # noqa: BLE001
			if attempt == retries - 1:
				return str(exc)
			time.sleep(delay_seconds)
	return "unknown error"


def _is_dir_empty(path: Path) -> bool:
	try:
		next(path.iterdir())
		return False
	except StopIteration:
		return True
	except FileNotFoundError:
		return True


def _sorted_dirs_descending(directories: list[Path]) -> list[Path]:
	collected: list[Path] = []
	for directory in directories:
		if directory.exists():
			collected.append(directory)
			collected.extend([path for path in directory.rglob("*") if path.is_dir()])
	return sorted(collected, key=lambda path: len(path.parts), reverse=True)


def _build_prompt(quality: Dict[str, float | bool]) -> str:
	return (
		"You are evaluating a representative frame from a short video clip for a child's growth memory slideshow. "
		"Return ONLY JSON. Do NOT output anything else. "
		"No extra text, no explanations, no markdown. "
		"The JSON MUST match this schema exactly, with no extra keys: "
		f"{json.dumps(SCHEMA_TEMPLATE, ensure_ascii=True)} "
		"Tags must be at most 5 items, all lowercase snake_case English words. "
		"Caption must be short Japanese (15-25 characters). "
		"Score must be between 0.0 and 1.0. "
		"If the image is inappropriate or cannot be judged, still return JSON with a low score. "
		"Background blur is acceptable. Focus on whether the subject looks sharp. "
		"Set risks.blur true when the subject or hands show motion blur. "
		"Set risks.out_of_focus true when the subject is not in focus. "
		"Consider the provided quality hints, including center and lower-area sharpness and exposure. "
		f"Quality hints: {json.dumps(quality, ensure_ascii=True)}"
	)


def _validate_analysis(analysis: Dict[str, Any]) -> Dict[str, Any]:
	return normalize_analysis(analysis)


def _apply_risk_penalties(score: float, risks: Dict[str, Any]) -> float:
	penalty_scale = 0.5
	if bool(risks.get("blur")):
		score -= 0.25 * penalty_scale
	if bool(risks.get("out_of_focus")):
		score -= 0.25 * penalty_scale
	if bool(risks.get("dark")):
		score -= 0.15 * penalty_scale
	if bool(risks.get("overexposed")):
		score -= 0.15 * penalty_scale

	if score < 0.0:
		return 0.0
	if score > 1.0:
		return 1.0
	return float(score)


def _process_sources(
	records: List[Dict[str, Any]],
	paths: VideoOutputPaths,
	max_source_seconds: int,
	preset: str,
	concat_in_digest_folder: bool,
	use_hwaccel: bool,
	job: JobContext,
	dedupe_enabled: bool,
	dedupe_hamming_threshold: int,
	dedupe_scope: str,
	max_selected_clips: int,
	target_digest_seconds: int,
) -> List[Dict[str, Any]]:
	grouped: dict[str, list[Dict[str, Any]]] = {}
	for record in records:
		source = record.get("source_path")
		if not isinstance(source, str):
			continue
		grouped.setdefault(source, []).append(record)

	results: list[Dict[str, Any]] = []
	shared_hashes: list[int] = []
	for source_path, source_records in grouped.items():
		result = _process_single_source(
			source_path,
			source_records,
			paths=paths,
			max_source_seconds=max_source_seconds,
			preset=preset,
			concat_in_digest_folder=concat_in_digest_folder,
			use_hwaccel=use_hwaccel,
			job=job,
			dedupe_enabled=dedupe_enabled,
			dedupe_hamming_threshold=dedupe_hamming_threshold,
			dedupe_scope=dedupe_scope,
			max_selected_clips=max_selected_clips,
			target_digest_seconds=target_digest_seconds,
			shared_hashes=shared_hashes,
		)
		results.append(result)

	return results


def _process_single_source(
	source_path: str,
	records: List[Dict[str, Any]],
	paths: VideoOutputPaths,
	max_source_seconds: int,
	preset: str,
	concat_in_digest_folder: bool,
	use_hwaccel: bool,
	job: JobContext,
	dedupe_enabled: bool,
	dedupe_hamming_threshold: int,
	dedupe_scope: str,
	max_selected_clips: int,
	target_digest_seconds: int,
	shared_hashes: list[int],
) -> Dict[str, Any]:
	source = Path(source_path)
	selected: list[Dict[str, Any]] = []
	selected_manifest: list[Dict[str, Any]] = []
	digest_path: Path | None = None
	return_error: str | None = None

	try:
		existing_hashes = shared_hashes if dedupe_scope == "global" else []
		selected, stats = _select_clips_for_source(
			records,
			max_source_seconds=max_source_seconds,
			max_selected_clips=max_selected_clips,
			target_digest_seconds=target_digest_seconds,
			dedupe_enabled=dedupe_enabled,
			hamming_threshold=dedupe_hamming_threshold,
			existing_hashes=existing_hashes,
		)
		job.record("select", source_path, "ok")
		log_event(
			"plain",
			level="info",
			event_type="video_dedupe",
			file_path=source_path,
			message="video dedupe summary",
			extra=stats,
		)
	except Exception as exc:  # noqa: BLE001
		return_error = str(exc)
		job.record("select", source_path, "failed", return_error)
		log_event(
			"plain",
			level="error",
			event_type="select_failed",
			file_path=source_path,
			message=return_error,
		)
		selected = []

	selected_sorted = sorted(selected, key=lambda item: float(item.get("start", 0.0)))
	selected_clips_dir = digest_clips_source_dir(paths, source.stem)
	selected_clips_dir.mkdir(parents=True, exist_ok=True)

	copied_paths: list[Path] = []
	for idx, record in enumerate(selected_sorted, start=1):
		clip_path = Path(record["clip_path"])
		destination = selected_clips_dir / f"clip_{idx:04d}{clip_path.suffix}"
		try:
			shutil.copy2(clip_path, destination)
			copied_paths.append(destination)
			selected_manifest.append(
				{
					"path": str(destination),
					"start": record.get("start"),
					"end": record.get("end"),
					"score": record.get("score_final"),
					"has_speech": record.get("has_speech"),
				}
			)
			job.record("select", str(clip_path), "ok")
		except Exception as exc:  # noqa: BLE001
			message = str(exc)
			job.record("select", str(clip_path), "failed", message)
			log_event(
				"plain",
				level="error",
				event_type="copy_failed",
				file_path=str(clip_path),
				message=message,
			)

	if preset != "clips_only":
		if copied_paths:
			digest_path = final_digest_path(paths, source.stem)
			log_event(
				"plain",
				level="info",
				event_type="concat_plan",
				file_path=str(digest_path),
				message="concatenating selected clips",
				extra={
					"total_clips": len(records),
					"selected_clips_count": len(selected_sorted),
					"concatenating_count": len(copied_paths),
					"output_file_path": str(digest_path),
				},
			)
			try:
				_concat_clips_reencode(
					copied_paths,
					digest_path,
					use_hwaccel,
					concat_list_path(paths, f"{source.stem}_root"),
				)
				job.record("concat", str(digest_path), "ok")
			except Exception as exc:  # noqa: BLE001
				message = str(exc)
				return_error = return_error or message
				job.record("concat", str(digest_path), "failed", message)
				log_event(
					"plain",
					level="error",
					event_type="concat_failed",
					file_path=str(digest_path),
					message=message,
				)
		else:
			log_event(
				"plain",
				level="info",
				event_type="concat_skipped",
				file_path=str(final_digest_path(paths, source.stem)),
				message="no selected clips to concatenate",
				extra={
					"total_clips": len(records),
					"selected_clips_count": len(selected_sorted),
					"concatenating_count": 0,
					"output_file_path": str(final_digest_path(paths, source.stem)),
				},
			)

	if concat_in_digest_folder:
		folder_concat = selected_clips_dir / "digest.mp4"
		if copied_paths:
			log_event(
				"plain",
				level="info",
				event_type="concat_plan",
				file_path=str(folder_concat),
				message="concatenating selected clips",
				extra={
					"total_clips": len(records),
					"selected_clips_count": len(selected_sorted),
					"concatenating_count": len(copied_paths),
					"output_file_path": str(folder_concat),
				},
			)
			try:
				_concat_clips_reencode(
					copied_paths,
					folder_concat,
					use_hwaccel,
					concat_list_path(paths, f"{source.stem}_folder"),
				)
				job.record("concat", str(folder_concat), "ok")
			except Exception as exc:  # noqa: BLE001
				message = str(exc)
				return_error = return_error or message
				job.record("concat", str(folder_concat), "failed", message)
				log_event(
					"plain",
					level="error",
					event_type="concat_failed",
					file_path=str(folder_concat),
					message=message,
				)
		else:
			log_event(
				"plain",
				level="info",
				event_type="concat_skipped",
				file_path=str(folder_concat),
				message="no selected clips to concatenate",
				extra={
					"total_clips": len(records),
					"selected_clips_count": len(selected_sorted),
					"concatenating_count": 0,
					"output_file_path": str(folder_concat),
				},
			)

	return {
		"source_video": source_path,
		"selected_clips": selected_manifest,
		"digest_path": str(digest_path) if digest_path else None,
		"total_duration": float(sum(float(item.get("duration", 0.0)) for item in selected)),
		"error": return_error,
	}


def _select_clips_for_source(
	records: List[Dict[str, Any]],
	*,
	max_source_seconds: int,
	max_selected_clips: int,
	target_digest_seconds: int,
	dedupe_enabled: bool,
	hamming_threshold: int,
	existing_hashes: list[int],
) -> tuple[list[Dict[str, Any]], dict[str, float | int | None]]:
	eligible = [
		record
		for record in records
		if not record.get("error")
		and isinstance(record.get("score_final"), (int, float))
		and _passes_quality_gate(record)
	]
	ordered = sorted(eligible, key=lambda item: item["score_final"], reverse=True)
	limit_seconds = min(max_source_seconds, target_digest_seconds)

	selected: list[Dict[str, Any]] = []
	total = 0.0
	removed_duplicates = 0
	for record in ordered:
		if len(selected) >= max_selected_clips:
			break
		duration = float(record.get("duration", 0.0))
		if total + duration > limit_seconds:
			continue
		candidate_hash = hash_to_int(record.get("frame_hash"))
		if (
			dedupe_enabled
			and candidate_hash is not None
			and is_near_duplicate(candidate_hash, existing_hashes, hamming_threshold)
		):
			removed_duplicates += 1
			continue
		selected.append(record)
		if candidate_hash is not None:
			existing_hashes.append(candidate_hash)
		total += duration
		if total >= limit_seconds:
			break

	score_stats = _score_stats([float(item["score_final"]) for item in eligible])
	stats: dict[str, float | int | None] = {
		"total_clips": len(records),
		"scored": len(eligible),
		"removed_duplicates": removed_duplicates,
		"selected": len(selected),
		"total_selected_seconds": round(total, 3),
		"score_min": score_stats.get("min"),
		"score_median": score_stats.get("median"),
		"score_p90": score_stats.get("p90"),
		"score_max": score_stats.get("max"),
	}
	return selected, stats


def _score_stats(scores: list[float]) -> dict[str, float | None]:
	if not scores:
		return {"min": None, "median": None, "p90": None, "max": None}
	sorted_scores = sorted(scores)
	count = len(sorted_scores)
	median_index = count // 2
	if count % 2 == 0:
		median = (sorted_scores[median_index - 1] + sorted_scores[median_index]) / 2.0
	else:
		median = sorted_scores[median_index]
	p90_index = int(0.9 * (count - 1))
	return {
		"min": sorted_scores[0],
		"median": median,
		"p90": sorted_scores[p90_index],
		"max": sorted_scores[-1],
	}


def _passes_quality_gate(record: Dict[str, Any]) -> bool:
	quality = record.get("quality")
	if not isinstance(quality, dict):
		return False
	brightness = quality.get("brightness")
	if isinstance(brightness, (int, float)) and brightness < MIN_BRIGHTNESS_GATE:
		return False
	return True


def _concat_clips_reencode(
	clips: List[Path],
	output_path: Path,
	use_hwaccel: bool,
	list_path: Path,
) -> None:
	list_path.parent.mkdir(parents=True, exist_ok=True)
	lines = [f"file '{clip.as_posix()}'" for clip in clips]
	list_path.write_text("\n".join(lines), encoding="utf-8")

	command = _build_concat_command(list_path, output_path, use_hwaccel)
	result = subprocess.run(command, capture_output=True, text=True, check=False)
	if result.returncode != 0:
		raise RuntimeError(result.stderr.strip() or "ffmpeg concat failed")


def _build_concat_command(list_path: Path, output_path: Path, use_hwaccel: bool) -> list[str]:
	if use_hwaccel:
		return [
			"ffmpeg",
			"-y",
			"-f",
			"concat",
			"-safe",
			"0",
			"-i",
			str(list_path),
			"-c:v",
			"h264_nvenc",
			"-preset",
			"p4",
			"-rc",
			"vbr",
			"-cq",
			"19",
			"-b:v",
			"10M",
			"-maxrate",
			"20M",
			"-bufsize",
			"20M",
			"-pix_fmt",
			"yuv420p",
			"-profile:v",
			"high",
			"-c:a",
			"aac",
			"-b:a",
			"128k",
			"-movflags",
			"+faststart",
			str(output_path),
		]

	return [
		"ffmpeg",
		"-y",
		"-f",
		"concat",
		"-safe",
		"0",
		"-i",
		str(list_path),
		"-c:v",
		"libx264",
		"-pix_fmt",
		"yuv420p",
		"-profile:v",
		"-c:a",
		"aac",
		"-movflags",
		"+faststart",
		str(output_path),
	]
