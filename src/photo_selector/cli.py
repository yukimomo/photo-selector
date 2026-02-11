from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from tqdm import tqdm

from photo_selector.analyzer import (
	analyze_quality,
	apply_quality_corrections,
	compute_image_hash,
	collect_image_paths,
	encode_image_base64,
	get_image_info,
)
from photo_selector.dependency_check import DependencyError, validate_dependencies
from photo_selector.execution_plan import build_execution_plan
from photo_selector.log_utils import log_event
from photo_selector.manifest import save_manifest
from photo_selector.ollama_client import OllamaClient
from photo_selector.output_paths import get_photo_paths
from photo_selector.resume_db import ScoreStore
from photo_selector.config_loader import coerce_bool, get_value, load_config
from photo_selector.score_schema import normalize_analysis
from photo_selector.selector import select_photos_with_dedupe


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
	)

	start_time = time.monotonic()
	input_dir = Path(args.input).expanduser().resolve()
	output_dir = Path(args.output).expanduser().resolve()
	paths = get_photo_paths(output_dir)

	if args.dry_run:
		plan = build_execution_plan(
			"photo",
			input_path=input_dir,
			output_dir=output_dir,
			resume=args.resume,
			force=args.force,
		)
		print(json.dumps(plan, ensure_ascii=True, indent=2))
		_summary_from_plan(args.log_format, plan, start_time)
		return 0

	output_dir.mkdir(parents=True, exist_ok=True)
	paths.selected_dir.mkdir(parents=True, exist_ok=True)
	paths.scores_dir.mkdir(parents=True, exist_ok=True)
	score_store = ScoreStore(paths.db_path)

	image_paths = collect_image_paths(input_dir)
	client = OllamaClient(base_url=args.ollama_base_url)

	photos: list[Dict[str, Any]] = []
	resume_enabled = bool(args.resume) and not bool(args.force)

	skipped = 0
	for path in tqdm(image_paths, desc="Analyzing", unit="image"):
		record: Dict[str, Any] = {
			"path": str(path),
		}
		try:
			info = get_image_info(path)
			image_hash = compute_image_hash(path)
			record.update(
				{
					"width": info.width,
					"height": info.height,
					"orientation": info.orientation,
					"hash": f"{image_hash:016x}",
				}
			)

			cached = None
			if resume_enabled:
				cached = score_store.get(record["path"], record["hash"])

			if cached is not None:
				skipped += 1
				analysis = cached.analysis or {"score": cached.score}
				record["analysis"] = _validate_analysis(analysis)
				record["quality"] = cached.quality
				record["error"] = None
			else:
				quality = analyze_quality(path)
				image_b64 = encode_image_base64(path)
				prompt = _build_prompt(quality)
				analysis = client.chat(args.model, image_b64, prompt)
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

				record["analysis"] = analysis
				record["quality"] = quality
				record["error"] = None
				score_store.upsert(
					record["path"],
					record["hash"],
					float(analysis["score"]),
					analysis,
					quality,
				)
		except Exception as exc:  # noqa: BLE001
			record["analysis"] = None
			record["quality"] = None
			record["error"] = str(exc)

		photos.append(record)

	scored_photos = [
		photo
		for photo in photos
		if not photo.get("error")
		and isinstance(photo.get("analysis"), dict)
		and isinstance(photo["analysis"].get("score"), (int, float))
	]
	selected_before_dedupe = min(args.target_count, len(scored_photos))
	selected = select_photos_with_dedupe(
		photos,
		args.target_count,
		args.photo_dedupe_hamming_threshold,
		args.photo_dedupe,
	)
	removed_duplicates = (
		selected_before_dedupe - len(selected) if args.photo_dedupe else 0
	)
	log_event(
		args.log_format,
		level="info",
		event_type="photo_dedupe",
		message="photo dedupe summary",
		extra={
			"total_photos": len(image_paths),
			"scored": len(scored_photos),
			"selected_before_dedupe": selected_before_dedupe,
			"removed_as_duplicate": removed_duplicates,
			"final_selected": len(selected),
		},
	)
	selected_paths = {item["path"] for item in selected}

	for record in photos:
		record["selected"] = record.get("path") in selected_paths

	for record in selected:
		try:
			source = Path(record["path"])
			destination = paths.selected_dir / source.name
			shutil.copy2(source, destination)
		except Exception as exc:  # noqa: BLE001
			record["error"] = record.get("error") or str(exc)
			record["selected"] = False

	save_manifest(paths.manifest_path, {"photos": photos})
	failed = sum(1 for photo in photos if photo.get("error"))
	processed = len(photos) - skipped
	_summary(
		args.log_format,
		total_files=len(image_paths),
		processed=processed,
		skipped=skipped,
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


def _build_prompt(quality: Dict[str, float | bool]) -> str:
	return (
		"You are evaluating a photo for a child's growth memory slideshow. "
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


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Photo selector MVP")
	parser.add_argument("--input", help="Input directory")
	parser.add_argument("--output", help="Output directory")
	parser.add_argument("--target-count", type=int)
	parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL"))
	parser.add_argument("--config", help="Path to config.yaml")
	parser.add_argument("--photo-dedupe", dest="photo_dedupe", action="store_true")
	parser.add_argument("--no-photo-dedupe", dest="photo_dedupe", action="store_false")
	parser.set_defaults(photo_dedupe=None)
	parser.add_argument("--photo-dedupe-hamming-threshold", type=int)
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
		"--resume",
		action="store_true",
		help="Skip already processed files based on stored hashes",
	)
	parser.add_argument(
		"--force",
		action="store_true",
		help="Recompute scores even if cached",
	)
	parser.add_argument(
		"--ollama-base-url",
		default=os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434",
		help="Ollama base URL",
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

	target_count = args.target_count
	if target_count is None:
		target_count = get_value(config, "target_count")
	if target_count is None:
		raise ValueError("Missing target_count. Set --target-count or config target_count.")
	try:
		target_count = int(target_count)
	except (TypeError, ValueError):
		raise ValueError("target_count must be an integer")

	dedupe_enabled = args.photo_dedupe
	if dedupe_enabled is None:
		dedupe_enabled = coerce_bool(get_value(config, "dedupe_enabled", "photo"))
	if dedupe_enabled is None:
		dedupe_enabled = True

	dedupe_threshold = args.photo_dedupe_hamming_threshold
	if dedupe_threshold is None:
		dedupe_threshold = get_value(config, "dedupe_hamming_threshold", "photo")
	if dedupe_threshold is None:
		dedupe_threshold = 6
	try:
		dedupe_threshold = int(dedupe_threshold)
	except (TypeError, ValueError):
		raise ValueError("photo.dedupe_hamming_threshold must be an integer")

	args.model = model
	args.ollama_base_url = base_url
	args.target_count = target_count
	args.photo_dedupe = bool(dedupe_enabled)
	args.photo_dedupe_hamming_threshold = dedupe_threshold

	input_path = args.input or get_value(config, "input")
	output_path = args.output or get_value(config, "output")
	if not isinstance(input_path, str) or not input_path:
		raise ValueError("Missing input. Set --input or config input.")
	if not isinstance(output_path, str) or not output_path:
		raise ValueError("Missing output. Set --output or config output.")
	args.input = input_path
	args.output = output_path

	hwaccel = coerce_bool(get_value(config, "hwaccel"))
	if hwaccel and not args.use_hwaccel:
		args.use_hwaccel = True


if __name__ == "__main__":
	raise SystemExit(main())
