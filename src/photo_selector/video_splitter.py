from __future__ import annotations

import subprocess
import json
from dataclasses import dataclass
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


@dataclass
class ClipInfo:
	source_path: Path
	clip_path: Path
	start: float
	end: float
	duration: float
	width: int
	height: int
	fps: float


@dataclass
class VideoMetadata:
	duration: float
	width: int
	height: int
	fps: float


def collect_video_paths(input_path: Path) -> list[Path]:
	if input_path.is_file():
		return [input_path] if input_path.suffix.lower() in VIDEO_EXTENSIONS else []

	paths: list[Path] = []
	for path in input_path.rglob("*"):
		if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
			paths.append(path)
	return paths


def get_video_metadata(path: Path) -> VideoMetadata:
	command = [
		"ffprobe",
		"-v",
		"error",
		"-select_streams",
		"v:0",
		"-show_entries",
		"stream=width,height,avg_frame_rate",
		"-show_entries",
		"format=duration",
		"-of",
		"json",
		str(path),
	]
	result = subprocess.run(command, capture_output=True, text=True, check=False)
	if result.returncode != 0:
		raise RuntimeError(result.stderr.strip() or "ffprobe failed")
	data = json.loads(result.stdout)
	stream = (data.get("streams") or [{}])[0]
	format_info = data.get("format") or {}

	duration = float(format_info.get("duration") or 0.0)
	width = int(stream.get("width") or 0)
	height = int(stream.get("height") or 0)
	fps = _parse_fps(stream.get("avg_frame_rate"))

	return VideoMetadata(duration=duration, width=width, height=height, fps=fps)


def split_video(
	path: Path,
	output_dir: Path,
	min_clip: int,
	max_clip: int,
	use_hwaccel: bool = False,
) -> list[ClipInfo]:
	output_dir.mkdir(parents=True, exist_ok=True)

	pattern = output_dir / "clip_%04d.mp4"
	_command_ffmpeg_segment(path, pattern, max_clip, use_hwaccel)

	clips: list[ClipInfo] = []
	clip_paths = sorted(output_dir.glob("clip_*.mp4"))
	start = 0.0
	for clip_path in clip_paths:
		metadata = get_video_metadata(clip_path)
		if metadata.duration < min_clip:
			continue

		end = start + metadata.duration
		clips.append(
			ClipInfo(
				source_path=path,
				clip_path=clip_path,
				start=start,
				end=end,
				duration=metadata.duration,
				width=metadata.width,
				height=metadata.height,
				fps=metadata.fps,
			)
		)
		start = end

	return clips


def _command_ffmpeg_segment(
	input_path: Path,
	output_pattern: Path,
	segment_time: int,
	use_hwaccel: bool,
) -> None:
	command = _build_segment_command(input_path, output_pattern, segment_time, use_hwaccel)
	result = subprocess.run(command, capture_output=True, text=True, check=False)
	if result.returncode != 0:
		raise RuntimeError(result.stderr.strip() or "ffmpeg segment failed")


def _build_segment_command(
	input_path: Path,
	output_pattern: Path,
	segment_time: int,
	use_hwaccel: bool,
) -> list[str]:
	if use_hwaccel:
		return [
			"ffmpeg",
			"-y",
			"-hwaccel",
			"cuda",
			"-i",
			str(input_path),
			"-c:v",
			"h264_nvenc",
			"-preset",
			"p4",
			"-rc",
			"vbr",
			"-cq",
			"19",
			"-b:v",
			"10M",
			"-maxrate",
			"20M",
			"-bufsize",
			"20M",
			"-pix_fmt",
			"yuv420p",
			"-force_key_frames",
			f"expr:gte(t,n_forced*{segment_time})",
			"-c:a",
			"aac",
			"-b:a",
			"128k",
			"-f",
			"segment",
			"-segment_time",
			str(segment_time),
			"-reset_timestamps",
			"1",
			str(output_pattern),
		]

	return [
		"ffmpeg",
		"-y",
		"-i",
		str(input_path),
		"-c:v",
		"libx264",
		"-crf",
		"22",
		"-preset",
		"veryfast",
		"-pix_fmt",
		"yuv420p",
		"-force_key_frames",
		f"expr:gte(t,n_forced*{segment_time})",
		"-c:a",
		"aac",
		"-b:a",
		"128k",
		"-f",
		"segment",
		"-segment_time",
		str(segment_time),
		"-reset_timestamps",
		"1",
		str(output_pattern),
	]


def _parse_fps(value: str | None) -> float:
	if not value or value == "0/0":
		return 0.0
	if "/" in value:
		numerator, denominator = value.split("/", 1)
		try:
			return float(numerator) / float(denominator)
		except (ValueError, ZeroDivisionError):
			return 0.0
	try:
		return float(value)
	except ValueError:
		return 0.0
