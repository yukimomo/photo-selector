from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PhotoOutputPaths:
	output_dir: Path
	selected_dir: Path
	scores_dir: Path
	manifest_path: Path
	db_path: Path


@dataclass(frozen=True)
class VideoOutputPaths:
	output_dir: Path
	scores_dir: Path
	temp_dir: Path
	digest_clips_dir: Path
	manifest_path: Path


def get_photo_paths(output_dir: Path) -> PhotoOutputPaths:
	scores_dir = output_dir / "scores"
	return PhotoOutputPaths(
		output_dir=output_dir,
		selected_dir=output_dir / "selected",
		scores_dir=scores_dir,
		manifest_path=scores_dir / "manifest.photos.json",
		db_path=scores_dir / "photo_scores.sqlite",
	)


def get_video_paths(output_dir: Path) -> VideoOutputPaths:
	scores_dir = output_dir / "scores"
	return VideoOutputPaths(
		output_dir=output_dir,
		scores_dir=scores_dir,
		temp_dir=output_dir / "temp",
		digest_clips_dir=output_dir / "digest_clips",
		manifest_path=scores_dir / "manifest.videos.json",
	)


def digest_clips_source_dir(paths: VideoOutputPaths, source_stem: str) -> Path:
	return paths.digest_clips_dir / source_stem


def final_digest_path(paths: VideoOutputPaths, source_stem: str) -> Path:
	return paths.output_dir / f"{source_stem}_digest.mp4"


def concat_list_path(paths: VideoOutputPaths, label: str) -> Path:
	return paths.temp_dir / "concat" / f"{label}.txt"
