from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from photo_selector.analyzer import (
	analyze_quality,
	apply_quality_corrections,
	encode_image_base64,
	get_image_info,
)
from photo_selector.audio_analyzer import AudioAnalysis, analyze_audio
from photo_selector.frame_extractor import extract_representative_frame
from photo_selector.ollama_client import OllamaClient
from photo_selector.output_paths import (
	concat_list_path,
	digest_clips_source_dir,
	final_digest_path,
	get_video_paths,
	VideoOutputPaths,
)
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


def run_video_digest(
	input_path: Path,
	output_dir: Path,
	max_source_seconds: int,
	min_clip: int,
	max_clip: int,
	model: str,
	base_url: str,
	keep_temp: bool,
	preset: str,
	concat_in_digest_folder: bool,
	use_hwaccel: bool,
) -> DigestResult:
	output_dir.mkdir(parents=True, exist_ok=True)
	paths = get_video_paths(output_dir)
	temp_dir = paths.temp_dir
	clip_dir = temp_dir / "clips"
	frame_dir = temp_dir / "frames"
	audio_dir = temp_dir / "audio"

	video_paths = collect_video_paths(input_path)
	clips: list[ClipInfo] = []
	for video_path in video_paths:
		video_clip_dir = clip_dir / video_path.stem
		clips.extend(
			split_video(
				video_path,
				video_clip_dir,
				min_clip,
				max_clip,
				use_hwaccel=use_hwaccel,
			)
		)

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
		except Exception as exc:  # noqa: BLE001
			record.update(
				{
					"analysis": None,
					"quality": None,
					"has_speech": None,
					"audio_rms": None,
					"score_final": None,
					"error": str(exc),
				}
			)

		clip_records.append(record)

	source_results = _process_sources(
		clip_records,
			paths=paths,
		max_source_seconds=max_source_seconds,
		preset=preset,
		concat_in_digest_folder=concat_in_digest_folder,
		use_hwaccel=use_hwaccel,
	)

	if not keep_temp:
		shutil.rmtree(temp_dir, ignore_errors=True)

	return DigestResult(sources=source_results)


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
	if not isinstance(analysis, dict):
		raise ValueError("Analysis is not a JSON object")

	analysis.setdefault("caption", "")
	analysis.setdefault("tags", [])
	analysis.setdefault("risks", {})

	if "score" not in analysis:
		raise ValueError("Missing key in analysis: score")

	if not isinstance(analysis.get("tags"), list):
		analysis["tags"] = []
	if not isinstance(analysis.get("risks"), dict):
		analysis["risks"] = {}

	if not isinstance(analysis.get("score"), (int, float)):
		raise ValueError("Score must be a number")

	risks = analysis.get("risks", {})
	for risk_key in ("blur", "dark", "overexposed", "out_of_focus"):
		risks.setdefault(risk_key, False)

	return analysis


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
) -> List[Dict[str, Any]]:
	grouped: dict[str, list[Dict[str, Any]]] = {}
	for record in records:
		source = record.get("source_path")
		if not isinstance(source, str):
			continue
		grouped.setdefault(source, []).append(record)

	results: list[Dict[str, Any]] = []
	for source_path, source_records in grouped.items():
		result = _process_single_source(
			source_path,
			source_records,
			paths=paths,
			max_source_seconds=max_source_seconds,
			preset=preset,
			concat_in_digest_folder=concat_in_digest_folder,
			use_hwaccel=use_hwaccel,
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
) -> Dict[str, Any]:
	source = Path(source_path)
	selected: list[Dict[str, Any]] = []
	selected_manifest: list[Dict[str, Any]] = []
	digest_path: Path | None = None
	return_error: str | None = None

	try:
		selected = _select_clips_for_source(records, max_source_seconds)
		selected_sorted = sorted(selected, key=lambda item: float(item.get("start", 0.0)))
		selected_clips_dir = digest_clips_source_dir(paths, source.stem)
		selected_clips_dir.mkdir(parents=True, exist_ok=True)

		copied_paths: list[Path] = []
		for idx, record in enumerate(selected_sorted, start=1):
			clip_path = Path(record["clip_path"])
			destination = selected_clips_dir / f"clip_{idx:04d}{clip_path.suffix}"
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

		if preset != "clips_only" and copied_paths:
			digest_path = final_digest_path(paths, source.stem)
			_concat_clips_reencode(
				copied_paths,
				digest_path,
				use_hwaccel,
				concat_list_path(paths, f"{source.stem}_root"),
			)

		if concat_in_digest_folder and copied_paths:
			folder_concat = selected_clips_dir / "digest.mp4"
			_concat_clips_reencode(
				copied_paths,
				folder_concat,
				use_hwaccel,
				concat_list_path(paths, f"{source.stem}_folder"),
			)
	except Exception as exc:  # noqa: BLE001
		return_error = str(exc)


	return {
		"source_video": source_path,
		"selected_clips": selected_manifest,
		"digest_path": str(digest_path) if digest_path else None,
		"total_duration": float(sum(float(item.get("duration", 0.0)) for item in selected)),
		"error": return_error,
	}


def _select_clips_for_source(records: List[Dict[str, Any]], max_source_seconds: int) -> List[Dict[str, Any]]:
	eligible = [
		record
		for record in records
		if not record.get("error")
		and isinstance(record.get("score_final"), (int, float))
		and _passes_quality_gate(record)
	]
	ordered = sorted(eligible, key=lambda item: item["score_final"], reverse=True)

	selected: list[Dict[str, Any]] = []
	total = 0.0
	for record in ordered:
		duration = float(record.get("duration", 0.0))
		if total + duration > max_source_seconds:
			continue
		selected.append(record)
		total += duration
		if total >= max_source_seconds:
			break

	return selected


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
