from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from photo_selector.execution_plan import build_execution_plan
from photo_selector.manifest_video import save_manifest
from photo_selector.video_digest import run_video_digest


def main() -> int:
	load_dotenv()
	args = _parse_args()

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
		return 0

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

	manifest = {
		"input": str(input_path),
		"max_source_seconds": args.max_source_seconds,
		"min_clip": args.min_clip,
		"max_clip": args.max_clip,
		"model": args.model,
		"preset": args.preset,
		"sources": result.sources,
	}

	save_manifest(output_dir / "manifest.videos.json", manifest)
	return 0


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
