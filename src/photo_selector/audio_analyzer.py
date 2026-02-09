from __future__ import annotations

import audioop
import os
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

SPEECH_RMS_THRESHOLD = 0.02
RMS_WINDOW_MS = 20
RMS_ACTIVE_RATIO = 0.2
VAD_SPEECH_RATIO = 0.15
USE_WEBRTC_VAD_ENV = "USE_WEBRTC_VAD"

try:
	import webrtcvad  # type: ignore

	HAS_WEBRTC_VAD = True
except Exception:
	HAS_WEBRTC_VAD = False


@dataclass
class AudioAnalysis:
	has_speech: bool
	rms: float


def analyze_audio(
	clip_path: Path,
	temp_dir: Path,
	use_vad: bool | None = None,
) -> AudioAnalysis:
	temp_dir.mkdir(parents=True, exist_ok=True)
	wav_path = temp_dir / f"{clip_path.stem}.wav"
	_extract_audio(clip_path, wav_path)

	with wave.open(str(wav_path), "rb") as audio_file:
		sample_rate = audio_file.getframerate()
		channels = audio_file.getnchannels()
		sample_width = audio_file.getsampwidth()
		n_frames = audio_file.getnframes()
		if n_frames == 0:
			return AudioAnalysis(has_speech=False, rms=0.0)

		if _should_use_vad(use_vad) and sample_rate in {8000, 16000, 32000, 48000}:
			return _analyze_with_vad(audio_file, sample_rate)

		return _analyze_with_rms(audio_file, sample_rate, channels, sample_width)


def _extract_audio(input_path: Path, output_path: Path) -> None:
	command = [
		"ffmpeg",
		"-y",
		"-i",
		str(input_path),
		"-vn",
		"-ac",
		"1",
		"-ar",
		"16000",
		"-sample_fmt",
		"s16",
		"-f",
		"wav",
		str(output_path),
	]
	result = subprocess.run(command, capture_output=True, text=True, check=False)
	if result.returncode != 0:
		raise RuntimeError(result.stderr.strip() or "ffmpeg audio extract failed")


def _should_use_vad(use_vad: bool | None) -> bool:
	if use_vad is not None:
		return use_vad and HAS_WEBRTC_VAD
	flag = os.getenv(USE_WEBRTC_VAD_ENV, "").strip().lower()
	return flag in {"1", "true", "yes"} and HAS_WEBRTC_VAD


def _analyze_with_rms(
	audio_file: wave.Wave_read,
	sample_rate: int,
	channels: int,
	sample_width: int,
) -> AudioAnalysis:
	window_frames = max(int(sample_rate * RMS_WINDOW_MS / 1000), 1)
	bytes_per_frame = sample_width * channels
	chunk_size = window_frames
	max_possible = float((1 << (8 * sample_width - 1)) - 1)

	rms_values: list[float] = []
	above_threshold = 0
	while True:
		frames = audio_file.readframes(chunk_size)
		if not frames:
			break
		rms = audioop.rms(frames, sample_width)
		normalized = float(rms) / max_possible
		rms_values.append(normalized)
		if normalized >= SPEECH_RMS_THRESHOLD:
			above_threshold += 1

	if not rms_values:
		return AudioAnalysis(has_speech=False, rms=0.0)

	mean_rms = sum(rms_values) / len(rms_values)
	active_ratio = above_threshold / len(rms_values)
	has_speech = active_ratio >= RMS_ACTIVE_RATIO
	return AudioAnalysis(has_speech=has_speech, rms=float(mean_rms))


def _analyze_with_vad(audio_file: wave.Wave_read, sample_rate: int) -> AudioAnalysis:
	if not HAS_WEBRTC_VAD:
		return AudioAnalysis(has_speech=False, rms=0.0)

	vad = webrtcvad.Vad(2)
	window_frames = max(int(sample_rate * RMS_WINDOW_MS / 1000), 1)
	bytes_per_frame = audio_file.getsampwidth() * audio_file.getnchannels()

	speech_frames = 0
	total_frames = 0
	while True:
		frames = audio_file.readframes(window_frames)
		if not frames or len(frames) < window_frames * bytes_per_frame:
			break
		is_speech = vad.is_speech(frames, sample_rate)
		total_frames += 1
		if is_speech:
			speech_frames += 1

	if total_frames == 0:
		return AudioAnalysis(has_speech=False, rms=0.0)

	ratio = speech_frames / total_frames
	return AudioAnalysis(has_speech=ratio >= VAD_SPEECH_RATIO, rms=0.0)
