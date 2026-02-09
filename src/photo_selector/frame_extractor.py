from __future__ import annotations

import subprocess
from pathlib import Path

from photo_selector.video_splitter import get_video_metadata


def extract_representative_frame(clip_path: Path, output_path: Path) -> Path:
	metadata = get_video_metadata(clip_path)
	timestamp = max(0.0, metadata.duration / 2.0)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	command = [
		"ffmpeg",
		"-y",
		"-ss",
		f"{timestamp:.3f}",
		"-i",
		str(clip_path),
		"-frames:v",
		"1",
		"-q:v",
		"2",
		str(output_path),
	]
	result = subprocess.run(command, capture_output=True, text=True, check=False)
	if result.returncode != 0:
		raise RuntimeError(result.stderr.strip() or "ffmpeg frame extract failed")

	return output_path
