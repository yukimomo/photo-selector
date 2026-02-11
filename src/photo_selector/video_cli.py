from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

from photo_selector.dependency_check import DependencyError, validate_dependencies
from photo_selector.execution_plan import build_execution_plan
from photo_selector.config_loader import coerce_bool, get_value, load_config
from photo_selector.log_utils import log_event
from photo_selector.manifest_video import save_manifest
from photo_selector.output_paths import get_video_paths
from photo_selector.video_digest import run_video_digest
from photo_selector.video_splitter import collect_video_paths


def main() -> int:
	load_dotenv()
	args = _parse_args()
	try:
		return _run(args)
	except DependencyError as exc:
		log_event(
			args.log_format,
			level="error",
			event_type="dependency_error",
			message=str(exc),
		)
		return 1
	except Exception as exc:  # noqa: BLE001
		if args.debug:
			raise
		log_event(
			args.log_format,
			level="error",
			event_type="error",
			message=str(exc),
		)
		return 1


def _run(args: argparse.Namespace) -> int:
	_apply_config(args)
	validate_dependencies(
		base_url=args.ollama_base_url,
		require_ffmpeg=True,
		require_ollama=True,
		use_hwaccel=args.use_hwaccel,
	)

	start_time = time.monotonic()
	input_path = Path(args.input).expanduser().resolve()
	output_dir = Path(args.output).expanduser().resolve()

	if args.dry_run:
		plan = build_execution_plan(
			"video",
			input_path=input_path,
			output_dir=output_dir,
			preset=args.preset,
			concat_in_digest_folder=args.concat_in_digest_folder,
		)
		print(json.dumps(plan, ensure_ascii=True, indent=2))
		_summary_from_plan(args.log_format, plan, start_time)
		return 0

	video_paths = collect_video_paths(input_path)

	result = run_video_digest(
		input_path=input_path,
		output_dir=output_dir,
		max_source_seconds=args.max_source_seconds,
		min_clip=args.min_clip,
		max_clip=args.max_clip,
		model=args.model,
		base_url=args.ollama_base_url,
		keep_temp=args.keep_temp,
		delete_split_files=args.delete_split_files,
		preset=args.preset,
		concat_in_digest_folder=args.concat_in_digest_folder,
		use_hwaccel=args.use_hwaccel,
		dedupe_enabled=args.video_dedupe,
		dedupe_hamming_threshold=args.video_dedupe_hamming_threshold,
		dedupe_scope=args.video_dedupe_scope,
		max_selected_clips=args.video_max_selected_clips,
		target_digest_seconds=args.video_target_digest_seconds,
	)

	paths = get_video_paths(output_dir)
	manifest = {
		"input": str(input_path),
		"max_source_seconds": args.max_source_seconds,
		"min_clip": args.min_clip,
		"max_clip": args.max_clip,
		"model": args.model,
		"preset": args.preset,
		"sources": result.sources,
		"job_state": result.job_state,
	}

	save_manifest(paths.manifest_path, manifest)
	failed = sum(1 for source in result.sources if source.get("error"))
	_summary(
		args.log_format,
		total_files=len(video_paths),
		processed=len(video_paths),
		skipped=0,
		failed=failed,
		start_time=start_time,
	)
	return 0


def _summary_from_plan(log_format: str, plan: Dict[str, Any], start_time: float) -> None:
	files_to_process = plan.get("files_to_process") or []
	files_to_skip = plan.get("files_to_skip") or []
	_summary(
		log_format,
		total_files=len(files_to_process) + len(files_to_skip),
		processed=len(files_to_process),
		skipped=len(files_to_skip),
		failed=0,
		start_time=start_time,
	)


def _summary(
	log_format: str,
	*,
	total_files: int,
	processed: int,
	skipped: int,
	failed: int,
	start_time: float,
) -> None:
	duration = time.monotonic() - start_time
	log_event(
		log_format,
		level="info",
		event_type="summary",
		message="summary",
		extra={
			"total_files": total_files,
			"processed": processed,
			"skipped": skipped,
			"failed": failed,
			"duration_seconds": round(duration, 3),
		},
	)


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Video digest MVP")
	parser.add_argument("--input", help="Input video file or folder")
	parser.add_argument("--output", help="Output directory")
	parser.add_argument("--max-source-seconds", type=int)
	parser.add_argument("--min-clip", default=2, type=int)
	parser.add_argument("--max-clip", default=6, type=int)
	parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL"))
	parser.add_argument("--config", help="Path to config.yaml")
	parser.add_argument("--video-dedupe", dest="video_dedupe", action="store_true")
	parser.add_argument("--no-video-dedupe", dest="video_dedupe", action="store_false")
	parser.set_defaults(video_dedupe=None)
	parser.add_argument("--video-dedupe-hamming-threshold", type=int)
	parser.add_argument(
		"--video-dedupe-scope",
		choices=["global", "per_source_video"],
	)
	parser.add_argument("--video-max-selected-clips", type=int)
	parser.add_argument("--video-target-digest-seconds", type=int)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Print an execution plan without writing files",
	)
	parser.add_argument(
		"--debug",
		action="store_true",
		help="Show stack traces on errors",
	)
	parser.add_argument(
		"--log-format",
		choices=["plain", "json"],
		default="plain",
		help="Log format",
	)
	parser.add_argument(
		"--ollama-base-url",
		default=os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434",
		help="Ollama base URL",
	)
	parser.add_argument("--keep-temp", action="store_true")
	parser.add_argument("--delete-split-files", action="store_true")
	parser.add_argument("--concat-in-digest-folder", action="store_true")
	parser.add_argument("--use-hwaccel", action="store_true")
	parser.add_argument(
		"--preset",
		choices=["youtube16x9", "shorts9x16", "clips_only"],
	)
	return parser.parse_args()


def _apply_config(args: argparse.Namespace) -> None:
	config: dict[str, object] = {}
	if args.config:
		config = load_config(Path(args.config).expanduser().resolve())

	model = args.model or get_value(config, "model") or os.getenv("OLLAMA_MODEL")
	if not isinstance(model, str) or not model:
		raise ValueError("Missing model. Set --model, config model, or OLLAMA_MODEL.")

	base_url = (
		args.ollama_base_url
		or get_value(config, "base_url")
		or os.getenv("OLLAMA_BASE_URL")
		or "http://localhost:11434"
	)
	if not isinstance(base_url, str) or not base_url:
		raise ValueError("Invalid base_url in config or args")

	preset = args.preset or get_value(config, "preset", "video") or "youtube16x9"
	if preset not in {"youtube16x9", "shorts9x16", "clips_only"}:
		raise ValueError("Invalid preset in config or args")

	hwaccel = coerce_bool(get_value(config, "use_hwaccel", "video"))
	if hwaccel is None:
		hwaccel = coerce_bool(get_value(config, "hwaccel", "video"))
	if hwaccel and not args.use_hwaccel:
		args.use_hwaccel = True

	concat_in_digest_folder = coerce_bool(
		get_value(config, "concat_in_digest_folder", "video")
	)
	if concat_in_digest_folder is None:
		concat_in_digest_folder = coerce_bool(get_value(config, "concat_in_digest_folder"))
	if concat_in_digest_folder and not args.concat_in_digest_folder:
		args.concat_in_digest_folder = True

	keep_temp = coerce_bool(get_value(config, "keep_temp", "video"))
	if keep_temp and not args.keep_temp:
		args.keep_temp = True

	delete_split_files = coerce_bool(get_value(config, "delete_split_files", "video"))
	if delete_split_files is None:
		delete_split_files = coerce_bool(get_value(config, "delete_split_files"))
	if delete_split_files and not args.delete_split_files:
		args.delete_split_files = True

	args.model = model
	args.ollama_base_url = base_url
	args.preset = preset

	max_source_seconds = args.max_source_seconds
	if max_source_seconds is None:
		max_source_seconds = get_value(config, "max_source_seconds", "video")
	if max_source_seconds is None:
		max_source_seconds = get_value(config, "max_source_seconds")
	if max_source_seconds is None:
		raise ValueError(
			"Missing max_source_seconds. Set --max-source-seconds or config max_source_seconds."
		)
	try:
		max_source_seconds = int(max_source_seconds)
	except (TypeError, ValueError):
		raise ValueError("max_source_seconds must be an integer")
	args.max_source_seconds = max_source_seconds

	video_dedupe_enabled = args.video_dedupe
	if video_dedupe_enabled is None:
		video_dedupe_enabled = coerce_bool(get_value(config, "dedupe_enabled", "video"))
	if video_dedupe_enabled is None:
		video_dedupe_enabled = True

	video_dedupe_threshold = args.video_dedupe_hamming_threshold
	if video_dedupe_threshold is None:
		video_dedupe_threshold = get_value(
			config, "dedupe_hamming_threshold", "video"
		)
	if video_dedupe_threshold is None:
		video_dedupe_threshold = 6
	try:
		video_dedupe_threshold = int(video_dedupe_threshold)
	except (TypeError, ValueError):
		raise ValueError("video.dedupe_hamming_threshold must be an integer")

	video_dedupe_scope = args.video_dedupe_scope or get_value(
		config, "dedupe_scope", "video"
	)
	if not video_dedupe_scope:
		video_dedupe_scope = "per_source_video"
	if video_dedupe_scope not in {"global", "per_source_video"}:
		raise ValueError("video.dedupe_scope must be global or per_source_video")

	max_selected_clips = args.video_max_selected_clips
	if max_selected_clips is None:
		max_selected_clips = get_value(config, "max_selected_clips", "video")
	if max_selected_clips is None:
		max_selected_clips = 20
	try:
		max_selected_clips = int(max_selected_clips)
	except (TypeError, ValueError):
		raise ValueError("video.max_selected_clips must be an integer")

	target_digest_seconds = args.video_target_digest_seconds
	if target_digest_seconds is None:
		target_digest_seconds = get_value(config, "target_digest_seconds", "video")
	if target_digest_seconds is None:
		target_digest_seconds = 90
	try:
		target_digest_seconds = int(target_digest_seconds)
	except (TypeError, ValueError):
		raise ValueError("video.target_digest_seconds must be an integer")

	args.video_dedupe = bool(video_dedupe_enabled)
	args.video_dedupe_hamming_threshold = video_dedupe_threshold
	args.video_dedupe_scope = video_dedupe_scope
	args.video_max_selected_clips = max_selected_clips
	args.video_target_digest_seconds = target_digest_seconds

	input_path = args.input or get_value(config, "input")
	output_path = args.output or get_value(config, "output")
	if not isinstance(input_path, str) or not input_path:
		raise ValueError("Missing input. Set --input or config input.")
	if not isinstance(output_path, str) or not output_path:
		raise ValueError("Missing output. Set --output or config output.")
	args.input = input_path
	args.output = output_path


if __name__ == "__main__":
	raise SystemExit(main())
