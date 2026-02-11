from __future__ import annotations

from typing import Any


def hash_to_int(value: Any) -> int | None:
	if isinstance(value, int):
		return value
	if isinstance(value, str):
		try:
			return int(value, 16)
		except ValueError:
			return None
	return None


def hamming_distance(left: int, right: int) -> int:
	return (left ^ right).bit_count()


def is_near_duplicate(candidate: int, selected: list[int], threshold: int) -> bool:
	for existing in selected:
		if hamming_distance(candidate, existing) <= threshold:
			return True
	return False
