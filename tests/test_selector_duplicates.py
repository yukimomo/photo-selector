from __future__ import annotations

from photo_selector.selector import select_photos_with_dedupe


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

	selected = select_photos_with_dedupe(
		photos, target_count=1, hamming_threshold=6, dedupe_enabled=True
	)

	assert len(selected) == 1
	assert selected[0]["path"] == "a.jpg"


def test_dedupe_disabled_keeps_ordered() -> None:
	photos = [
		_photo("a.jpg", 0.9, "ffffffffffffffff"),
		_photo("b.jpg", 0.8, "ffffffffffffffff"),
	]

	selected = select_photos_with_dedupe(
		photos, target_count=2, hamming_threshold=6, dedupe_enabled=False
	)

	assert [item["path"] for item in selected] == ["a.jpg", "b.jpg"]
