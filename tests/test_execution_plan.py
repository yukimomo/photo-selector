from __future__ import annotations

from pathlib import Path

from PIL import Image

from photo_selector.analyzer import compute_image_hash
from photo_selector.execution_plan import build_execution_plan
from photo_selector.output_paths import get_photo_paths
from photo_selector.resume_db import ScoreStore


def _write_image(path: Path) -> None:
	image = Image.new("RGB", (10, 10), color=(255, 0, 0))
	image.save(path)


def test_execution_plan_resume_skips_cached(tmp_path: Path) -> None:
	input_dir = tmp_path / "input"
	output_dir = tmp_path / "output"
	input_dir.mkdir()

	image_path = input_dir / "photo.jpg"
	_write_image(image_path)

	paths = get_photo_paths(output_dir)
	paths.scores_dir.mkdir(parents=True, exist_ok=True)
	store = ScoreStore(paths.db_path)
	image_hash = compute_image_hash(image_path)
	hex_hash = f"{image_hash:016x}"
	store.upsert(str(image_path), hex_hash, 0.5, {"score": 0.5}, {})

	plan = build_execution_plan(
		"photo",
		input_path=input_dir,
		output_dir=output_dir,
		resume=True,
		force=False,
	)

	assert plan["files_to_process"] == []
	assert plan["files_to_skip"] == [str(image_path)]


def test_execution_plan_without_resume_processes_all(tmp_path: Path) -> None:
	input_dir = tmp_path / "input"
	output_dir = tmp_path / "output"
	input_dir.mkdir()

	image_path = input_dir / "photo.jpg"
	_write_image(image_path)

	plan = build_execution_plan(
		"photo",
		input_path=input_dir,
		output_dir=output_dir,
		resume=False,
		force=False,
	)

	assert plan["files_to_process"] == [str(image_path)]
	assert plan["files_to_skip"] == []
