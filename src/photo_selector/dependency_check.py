from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

import requests


@dataclass
class DependencyStatus:
	ffmpeg_ok: bool
	nvenc_available: bool
	ollama_ok: bool


class DependencyError(RuntimeError):
	pass


def validate_dependencies(
	*,
	base_url: str,
	require_ffmpeg: bool,
	require_ollama: bool,
	use_hwaccel: bool = False,
) -> DependencyStatus:
	ffmpeg_ok = _check_ffmpeg() if require_ffmpeg else True
	if require_ffmpeg and not ffmpeg_ok:
		raise DependencyError("ffmpeg not found in PATH.")

	nvenc_available = _check_nvenc() if require_ffmpeg else False
	if use_hwaccel and not nvenc_available:
		raise DependencyError("NVENC encoder not available in ffmpeg.")

	ollama_ok = _check_ollama(base_url) if require_ollama else True
	if require_ollama and not ollama_ok:
		raise DependencyError(
			"Cannot reach Ollama server. Check OLLAMA_BASE_URL or --ollama-base-url."
		)

	return DependencyStatus(
		ffmpeg_ok=ffmpeg_ok,
		nvenc_available=nvenc_available,
		ollama_ok=ollama_ok,
	)


def _check_ffmpeg() -> bool:
	return shutil.which("ffmpeg") is not None


def _check_nvenc() -> bool:
	command = ["ffmpeg", "-hide_banner", "-encoders"]
	result = subprocess.run(command, capture_output=True, text=True, check=False)
	if result.returncode != 0:
		return False
	return "h264_nvenc" in result.stdout


def _check_ollama(base_url: str) -> bool:
	url = f"{base_url.rstrip('/')}/api/tags"
	try:
		response = requests.get(url, timeout=3)
		return response.status_code == 200
	except Exception:
		return False
