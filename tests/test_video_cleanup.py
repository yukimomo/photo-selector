from __future__ import annotations

from pathlib import Path

from photo_selector.video_digest import _cleanup_temp_artifacts


def _setup_temp_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
	output_dir = tmp_path / "output"
	temp_dir = output_dir / "temp"
	clip_root = temp_dir / "clips"
	clip_dir = clip_root / "source"
	clip_dir.mkdir(parents=True)
	(clip_dir / "clip_0001.mp4").write_text("clip", encoding="utf-8")
	(frame_dir := temp_dir / "frames" / "source").mkdir(parents=True)
	(frame_dir / "frame.jpg").write_text("frame", encoding="utf-8")
	return output_dir, temp_dir, clip_root


def test_cleanup_removes_temp_when_keep_temp_false(tmp_path: Path) -> None:
	output_dir, temp_dir, clip_root = _setup_temp_tree(tmp_path)
	sources = [{"error": None}]
	job_state = {
		"split": {"a": {"status": "ok"}},
		"score": {},
		"select": {},
		"concat": {},
	}

	_cleanup_temp_artifacts(
		output_dir=output_dir,
		clip_dir=clip_root,
		temp_dir=temp_dir,
		keep_temp=False,
		delete_split_files=True,
		sources=sources,
		job_state=job_state,
	)

	assert not temp_dir.exists()


def test_cleanup_skips_when_keep_temp_true(tmp_path: Path) -> None:
	output_dir, temp_dir, clip_root = _setup_temp_tree(tmp_path)
	sources = [{"error": None}]
	job_state = {
		"split": {"a": {"status": "ok"}},
		"score": {},
		"select": {},
		"concat": {},
	}

	_cleanup_temp_artifacts(
		output_dir=output_dir,
		clip_dir=clip_root,
		temp_dir=temp_dir,
		keep_temp=True,
		delete_split_files=True,
		sources=sources,
		job_state=job_state,
	)

	assert temp_dir.exists()


def test_cleanup_skips_on_failure(tmp_path: Path) -> None:
	output_dir, temp_dir, clip_root = _setup_temp_tree(tmp_path)
	sources = [{"error": "boom"}]
	job_state = {
		"split": {"a": {"status": "failed"}},
		"score": {},
		"select": {},
		"concat": {},
	}

	_cleanup_temp_artifacts(
		output_dir=output_dir,
		clip_dir=clip_root,
		temp_dir=temp_dir,
		keep_temp=False,
		delete_split_files=True,
		sources=sources,
		job_state=job_state,
	)

	assert temp_dir.exists()


def test_cleanup_safety_outside_temp_root(tmp_path: Path) -> None:
	output_dir = tmp_path / "output"
	unsafe_temp = tmp_path / "unsafe"
	clip_root = unsafe_temp / "clips"
	clip_root.mkdir(parents=True)
	(clip_root / "clip_0001.mp4").write_text("clip", encoding="utf-8")

	sources = [{"error": None}]
	job_state = {
		"split": {"a": {"status": "ok"}},
		"score": {},
		"select": {},
		"concat": {},
	}

	_cleanup_temp_artifacts(
		output_dir=output_dir,
		clip_dir=clip_root,
		temp_dir=unsafe_temp,
		keep_temp=False,
		delete_split_files=True,
		sources=sources,
		job_state=job_state,
	)

	assert unsafe_temp.exists()
