from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, cast

from PIL import Image, ImageFilter, ImageStat

try:
	import pillow_heif  # type: ignore

	pillow_heif.register_heif_opener()
except Exception:
	# HEIC support is optional; if not installed, HEIC files are skipped/unsupported.
	pass


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic"}
HASH_SIZE = 8
MIN_SHORT_SIDE = 720
DARK_PENALTY = 0.2
LOW_RESOLUTION_PENALTY = 0.2
BLUR_PENALTY = 0.25
BLUR_STRONG_PENALTY = 0.4
QUALITY_PENALTY_SCALE = 0.5
MIN_EDGE_VARIANCE = 140.0
MIN_CENTER_EDGE_VARIANCE = 220.0
MIN_LOWER_EDGE_VARIANCE = 180.0
MIN_STRONG_CENTER_VARIANCE = 140.0


@dataclass
class ImageInfo:
	width: int
	height: int
	orientation: str


def collect_image_paths(input_dir: Path) -> list[Path]:
	paths: list[Path] = []
	for path in input_dir.rglob("*"):
		if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
			paths.append(path)
	return paths


def get_image_info(path: Path) -> ImageInfo:
	with Image.open(path) as image:
		width, height = image.size
	orientation = _compute_orientation(width, height)
	return ImageInfo(width=width, height=height, orientation=orientation)


def encode_image_base64(path: Path) -> str:
	with Image.open(path) as image:
		output_format = _choose_output_format(path, image.format)
		safe_image = image
		if output_format == "JPEG" and image.mode in {"RGBA", "LA", "P"}:
			safe_image = image.convert("RGB")

		with BytesIO() as buffer:
			safe_image.save(buffer, format=output_format)
			return base64.b64encode(buffer.getvalue()).decode("ascii")


def analyze_quality(path: Path) -> Dict[str, float | bool]:
	with Image.open(path) as image:
		grayscale = image.convert("L")
		stat = ImageStat.Stat(grayscale)
		brightness = float(stat.mean[0])
		width, height = image.size
		edge_variance = _edge_variance(grayscale)
		center_variance = _center_edge_variance(grayscale)
		lower_variance = _lower_edge_variance(grayscale)

	resolution = float(width * height)
	dark = brightness < 50.0
	overexposed = brightness > 205.0
	blur = edge_variance < MIN_EDGE_VARIANCE
	blur_center = center_variance < MIN_CENTER_EDGE_VARIANCE
	blur_lower = lower_variance < MIN_LOWER_EDGE_VARIANCE
	blur_strong = center_variance < MIN_STRONG_CENTER_VARIANCE

	return {
		"brightness": brightness,
		"resolution": resolution,
		"edge_variance": edge_variance,
		"center_edge_variance": center_variance,
		"lower_edge_variance": lower_variance,
		"dark": dark,
		"overexposed": overexposed,
		"blur": blur,
		"blur_center": blur_center,
		"blur_lower": blur_lower,
		"blur_strong": blur_strong,
	}


def compute_image_hash(path: Path, hash_size: int = HASH_SIZE) -> int:
	with Image.open(path) as image:
		grayscale = image.convert("L")
		resized = grayscale.resize((hash_size, hash_size), Image.Resampling.BILINEAR)
		pixels = list(_get_flattened_pixels(resized))
	avg = sum(pixels) / len(pixels)

	hash_value = 0
	for idx, pixel in enumerate(pixels):
		if pixel >= avg:
			hash_value |= 1 << idx

	return hash_value


def _get_flattened_pixels(image: Image.Image) -> Iterable[int]:
	get_flattened = getattr(image, "get_flattened_data", None)
	if callable(get_flattened):
		return cast(Iterable[int], get_flattened())
	return cast(Iterable[int], image.getdata())


def apply_quality_corrections(
	score: float,
	quality: Dict[str, float | bool],
	width: int,
	height: int,
) -> float:
	if score > 1.0:
		score = score / 100.0

	if bool(quality.get("dark")):
		score -= DARK_PENALTY * QUALITY_PENALTY_SCALE
	if bool(quality.get("blur_strong")):
		score -= BLUR_STRONG_PENALTY * QUALITY_PENALTY_SCALE
	if bool(quality.get("blur_center")):
		score -= BLUR_PENALTY * QUALITY_PENALTY_SCALE
	if bool(quality.get("blur_lower")):
		score -= BLUR_PENALTY * 0.6 * QUALITY_PENALTY_SCALE

	short_side = min(width, height)
	if short_side < MIN_SHORT_SIDE:
		score -= LOW_RESOLUTION_PENALTY * QUALITY_PENALTY_SCALE

	return _clamp_score(score)


def _compute_orientation(width: int, height: int) -> str:
	if width == height:
		return "square"
	if width > height:
		return "landscape"
	return "portrait"


def _choose_output_format(path: Path, image_format: str | None) -> str:
	if path.suffix.lower() == ".heic":
		return "JPEG"
	if image_format is None:
		return "JPEG"
	format_upper = image_format.upper()
	if format_upper in {"JPEG", "JPG", "PNG"}:
		return "PNG" if format_upper == "PNG" else "JPEG"
	return "JPEG"


def _clamp_score(score: float) -> float:
	if score < 0.0:
		return 0.0
	if score > 1.0:
		return 1.0
	return float(score)


def _edge_variance(grayscale: Image.Image) -> float:
	edges = grayscale.filter(ImageFilter.FIND_EDGES)
	edge_stat = ImageStat.Stat(edges)
	return float(edge_stat.var[0])


def _center_edge_variance(grayscale: Image.Image) -> float:
	width, height = grayscale.size
	left = int(width * 0.25)
	right = int(width * 0.75)
	top = int(height * 0.25)
	bottom = int(height * 0.75)
	center = grayscale.crop((left, top, right, bottom))
	return _edge_variance(center)


def _lower_edge_variance(grayscale: Image.Image) -> float:
	width, height = grayscale.size
	left = int(width * 0.1)
	right = int(width * 0.9)
	top = int(height * 0.5)
	bottom = int(height * 0.95)
	lower = grayscale.crop((left, top, right, bottom))
	return _edge_variance(lower)
