from __future__ import annotations

from photo_selector.selector import select_top_photos


def _photo(path: str, score: float, hash_hex: str) -> dict[str, object]:
	return {
		"path": path,
		"analysis": {"score": score},
		"hash": hash_hex,
		"error": None,
	}


def test_near_duplicate_selection_prefers_high_score() -> None:
	hash_hex = "ffffffffffffffff"
	photos = [
		_photo("a.jpg", 0.9, hash_hex),
		_photo("b.jpg", 0.8, hash_hex),
	]

	selected = select_top_photos(photos, target_count=1)

	assert len(selected) == 1
	assert selected[0]["path"] == "a.jpg"
