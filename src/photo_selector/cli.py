from __future__ import annotations

import argparse
import json
import os
import shutil
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
from photo_selector.manifest import save_manifest
from photo_selector.ollama_client import OllamaClient
from photo_selector.selector import select_top_photos


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

	input_dir = Path(args.input).expanduser().resolve()
	output_dir = Path(args.output).expanduser().resolve()
	selected_dir = output_dir / "selected"
	manifest_path = output_dir / "manifest.photos.json"

	output_dir.mkdir(parents=True, exist_ok=True)
	selected_dir.mkdir(parents=True, exist_ok=True)

	image_paths = collect_image_paths(input_dir)
	client = OllamaClient(base_url=args.ollama_base_url)

	photos: list[Dict[str, Any]] = []
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
		except Exception as exc:  # noqa: BLE001
			record["analysis"] = None
			record["quality"] = None
			record["error"] = str(exc)

		photos.append(record)

	selected = select_top_photos(photos, args.target_count)
	selected_paths = {item["path"] for item in selected}

	for record in photos:
		record["selected"] = record.get("path") in selected_paths

	for record in selected:
		try:
			source = Path(record["path"])
			destination = selected_dir / source.name
			shutil.copy2(source, destination)
		except Exception as exc:  # noqa: BLE001
			record["error"] = record.get("error") or str(exc)
			record["selected"] = False

	save_manifest(manifest_path, {"photos": photos})
	return 0


def _build_prompt(quality: Dict[str, float | bool]) -> str:
	return (
		"You are evaluating a photo for a child's growth memory slideshow. "
		"Return ONLY JSON. No extra text, no explanations, no markdown. "
		"The JSON MUST match this schema exactly, with no extra keys: "
		f"{json.dumps(SCHEMA_TEMPLATE, ensure_ascii=True)} "
		"Tags must be at most 5 items, all lowercase snake_case English words. "
		"Use realistic scores from 0.0 to 1.0. "
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


def _parse_args() -> argparse.Namespace:
	env_model = os.getenv("OLLAMA_MODEL")
	env_base_url = os.getenv("OLLAMA_BASE_URL")
	parser = argparse.ArgumentParser(description="Photo selector MVP")
	parser.add_argument("--input", required=True, help="Input directory")
	parser.add_argument("--output", required=True, help="Output directory")
	parser.add_argument("--target-count", required=True, type=int)
	parser.add_argument("--model", default=env_model, required=env_model is None)
	parser.add_argument(
		"--ollama-base-url",
		default=env_base_url or "http://localhost:11434",
		help="Ollama base URL",
	)
	return parser.parse_args()


if __name__ == "__main__":
	raise SystemExit(main())
