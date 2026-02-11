from __future__ import annotations

from pathlib import Path

from photo_selector.output_paths import get_video_paths
from photo_selector.video_digest import JobContext, _process_single_source


def _record(path: Path, start: float) -> dict[str, object]:
	return {
		"clip_path": str(path),
		"score_final": 0.9,
		"duration": 5.0,
		"start": start,
		"end": start + 5.0,
		"frame_hash": "ffffffffffffffff",
		"quality": {"brightness": 100.0},
		"has_speech": None,
		"error": None,
	}


def test_concat_uses_selected_clips_only(tmp_path: Path, monkeypatch) -> None:
	paths = get_video_paths(tmp_path)
	source_path = tmp_path / "source.mp4"
	source_path.write_text("source", encoding="utf-8")

	clip_a = tmp_path / "a.mp4"
	clip_b = tmp_path / "b.mp4"
	clip_c = tmp_path / "c.mp4"
	for clip in (clip_a, clip_b, clip_c):
		clip.write_text("clip", encoding="utf-8")

	records = [_record(clip_a, 0.0), _record(clip_b, 5.0), _record(clip_c, 10.0)]
	selected_records = [records[1], records[2]]

	def fake_select(*_args, **_kwargs):
		stats = {
			"total_clips": 3,
			"selected_clips_count": 2,
			"concatenating_count": 2,
		}
		return selected_records, stats

	captured: dict[str, list[Path]] = {}

	def fake_concat(clips, output_path, use_hwaccel, list_path):
		captured["clips"] = list(clips)
		captured["output"] = output_path
		captured["list_path"] = list_path

	monkeypatch.setattr("photo_selector.video_digest._select_clips_for_source", fake_select)
	monkeypatch.setattr("photo_selector.video_digest._concat_clips_reencode", fake_concat)
	monkeypatch.setattr("shutil.copy2", lambda _src, _dst: None)

	_process_single_source(
		str(source_path),
		records,
		paths=paths,
		max_source_seconds=60,
		preset="youtube16x9",
		concat_in_digest_folder=False,
		use_hwaccel=False,
		job=JobContext(),
		dedupe_enabled=True,
		dedupe_hamming_threshold=6,
		dedupe_scope="per_source_video",
		max_selected_clips=20,
		target_digest_seconds=90,
		shared_hashes=[],
	)

	selected_dir = paths.digest_clips_dir / source_path.stem
	expected = [
		selected_dir / "clip_0001.mp4",
		selected_dir / "clip_0002.mp4",
	]
	assert captured["clips"] == expected
