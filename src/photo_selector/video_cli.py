from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from photo_selector.dependency_check import DependencyError, validate_dependencies
from photo_selector.execution_plan import build_execution_plan
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
		preset=args.preset,
		concat_in_digest_folder=args.concat_in_digest_folder,
		use_hwaccel=args.use_hwaccel,
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
	env_model = os.getenv("OLLAMA_MODEL")
	env_base_url = os.getenv("OLLAMA_BASE_URL")
	parser = argparse.ArgumentParser(description="Video digest MVP")
	parser.add_argument("--input", required=True, help="Input video file or folder")
	parser.add_argument("--output", required=True, help="Output directory")
	parser.add_argument("--max-source-seconds", required=True, type=int)
	parser.add_argument("--min-clip", default=2, type=int)
	parser.add_argument("--max-clip", default=6, type=int)
	parser.add_argument("--model", default=env_model, required=env_model is None)
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
		default=env_base_url or "http://localhost:11434",
		help="Ollama base URL",
	)
	parser.add_argument("--keep-temp", action="store_true")
	parser.add_argument("--concat-in-digest-folder", action="store_true")
	parser.add_argument("--use-hwaccel", action="store_true")
	parser.add_argument(
		"--preset",
		default="youtube16x9",
		choices=["youtube16x9", "shorts9x16", "clips_only"],
	)
	return parser.parse_args()


if __name__ == "__main__":
	raise SystemExit(main())
