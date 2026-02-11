from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

from photo_selector.analyzer import collect_image_paths, compute_image_hash
from photo_selector.resume_db import ScoreStore
from photo_selector.video_splitter import collect_video_paths


def build_execution_plan(
	plan_type: str,
	*,
	input_path: Path,
	output_dir: Path,
	resume: bool = False,
	force: bool = False,
	preset: str | None = None,
	concat_in_digest_folder: bool = False,
) -> Dict[str, Any]:
	if plan_type == "photo":
		return _build_photo_plan(
			input_path=input_path,
			output_dir=output_dir,
			resume=resume,
			force=force,
		)
	if plan_type == "video":
		return _build_video_plan(
			input_path=input_path,
			output_dir=output_dir,
			preset=preset,
			concat_in_digest_folder=concat_in_digest_folder,
		)
	raise ValueError(f"Unknown plan type: {plan_type}")


def _build_photo_plan(
	*,
	input_path: Path,
	output_dir: Path,
	resume: bool,
	force: bool,
) -> Dict[str, Any]:
	resume_enabled = bool(resume) and not bool(force)
	db_path = output_dir / "photo_scores.sqlite"
	score_store = ScoreStore(db_path, create=False) if resume_enabled else None

	image_paths = collect_image_paths(input_path)
	files_to_process: list[str] = []
	files_to_skip: list[str] = []

	for path in image_paths:
		file_hash = compute_image_hash(path)
		hex_hash = f"{file_hash:016x}"
		if resume_enabled and score_store is not None:
			cached = score_store.get(str(path), hex_hash)
			if cached is not None:
				files_to_skip.append(str(path))
				continue
		files_to_process.append(str(path))

	selected_dir = output_dir / "selected"
	estimated_outputs: list[str] = [
		str(output_dir / "manifest.photos.json"),
		str(selected_dir),
	]
	estimated_outputs.extend(
		str(selected_dir / Path(path).name) for path in files_to_process
	)

	return {
		"type": "photo",
		"resume": resume_enabled,
		"files_to_process": files_to_process,
		"files_to_skip": files_to_skip,
		"estimated_output_paths": _dedupe(estimated_outputs),
	}


def _build_video_plan(
	*,
	input_path: Path,
	output_dir: Path,
	preset: str | None,
	concat_in_digest_folder: bool,
) -> Dict[str, Any]:
	video_paths = collect_video_paths(input_path)
	files_to_process = [str(path) for path in video_paths]
	files_to_skip: list[str] = []

	estimated_outputs: list[str] = [
		str(output_dir / "manifest.videos.json"),
		str(output_dir / "digest_clips"),
		str(output_dir / "per_source"),
		str(output_dir / "temp"),
	]

	for path in video_paths:
		stem = path.stem
		estimated_outputs.append(
			str(output_dir / "digest_clips" / stem / "clip_*.mp4")
		)
		if preset != "clips_only":
			estimated_outputs.append(
				str(output_dir / "per_source" / f"{stem}_digest.mp4")
			)
		if concat_in_digest_folder:
			estimated_outputs.append(
				str(output_dir / "digest_clips" / stem / "digest.mp4")
			)

	return {
		"type": "video",
		"preset": preset,
		"concat_in_digest_folder": bool(concat_in_digest_folder),
		"files_to_process": files_to_process,
		"files_to_skip": files_to_skip,
		"estimated_output_paths": _dedupe(estimated_outputs),
	}


def _dedupe(items: Iterable[str]) -> list[str]:
	seen: set[str] = set()
	result: list[str] = []
	for item in items:
		if item in seen:
			continue
		seen.add(item)
		result.append(item)
	return result
