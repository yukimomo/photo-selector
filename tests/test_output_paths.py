from __future__ import annotations

from pathlib import Path

from photo_selector.output_paths import (
	concat_list_path,
	digest_clips_source_dir,
	final_digest_path,
	get_photo_paths,
	get_video_paths,
)


def test_photo_output_paths(tmp_path: Path) -> None:
	paths = get_photo_paths(tmp_path)
	assert paths.output_dir == tmp_path
	assert paths.selected_dir == tmp_path / "selected"
	assert paths.scores_dir == tmp_path / "scores"
	assert paths.manifest_path == tmp_path / "scores" / "manifest.photos.json"
	assert paths.db_path == tmp_path / "scores" / "photo_scores.sqlite"


def test_video_output_paths(tmp_path: Path) -> None:
	paths = get_video_paths(tmp_path)
	assert paths.output_dir == tmp_path
	assert paths.scores_dir == tmp_path / "scores"
	assert paths.temp_dir == tmp_path / "temp"
	assert paths.digest_clips_dir == tmp_path / "digest_clips"
	assert paths.manifest_path == tmp_path / "scores" / "manifest.videos.json"
	assert digest_clips_source_dir(paths, "source") == tmp_path / "digest_clips" / "source"
	assert final_digest_path(paths, "source") == tmp_path / "source_digest.mp4"
	assert concat_list_path(paths, "source") == tmp_path / "temp" / "concat" / "source.txt"
