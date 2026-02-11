from __future__ import annotations

from photo_selector.video_digest import _select_clips_for_source


def _record(path: str, score: float, duration: float, frame_hash: str) -> dict[str, object]:
	return {
		"clip_path": path,
		"score_final": score,
		"duration": duration,
		"frame_hash": frame_hash,
		"quality": {"brightness": 100.0},
		"error": None,
	}


def test_video_dedupe_keeps_best_scoring_clip() -> None:
	records = [
		_record("a.mp4", 0.9, 5.0, "ffffffffffffffff"),
		_record("b.mp4", 0.8, 5.0, "ffffffffffffffff"),
	]

	selected, stats = _select_clips_for_source(
		records,
		max_source_seconds=60,
		max_selected_clips=20,
		target_digest_seconds=90,
		dedupe_enabled=True,
		hamming_threshold=6,
		existing_hashes=[],
	)
	assert len(selected) == 1
	assert selected[0]["clip_path"] == "a.mp4"
	assert stats["removed_duplicates"] == 1




def test_video_caps_limit_selection() -> None:
	records = [
		_record("a.mp4", 0.9, 50.0, "ffffffffffffffff"),
		_record("b.mp4", 0.8, 50.0, "0f0f0f0f0f0f0f0f"),
		_record("c.mp4", 0.7, 50.0, "f0f0f0f0f0f0f0f0"),
	]
	selected, stats = _select_clips_for_source(
		records,
		max_source_seconds=300,
		max_selected_clips=2,
		target_digest_seconds=60,
		dedupe_enabled=False,
		hamming_threshold=6,
		existing_hashes=[],
	)
	assert len(selected) == 1
	assert stats["total_selected_seconds"] == 50.0
