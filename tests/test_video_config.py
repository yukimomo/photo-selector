from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from photo_selector.video_cli import _apply_config


def _base_args(config_path: Path):
	return SimpleNamespace(
		config=str(config_path),
		model=None,
		ollama_base_url=None,
		preset=None,
		use_hwaccel=False,
		concat_in_digest_folder=False,
		keep_temp=False,
		delete_split_files=False,
		max_source_seconds=None,
		input=None,
		output=None,
		video_dedupe=None,
		video_dedupe_hamming_threshold=None,
		video_dedupe_scope=None,
		video_max_selected_clips=None,
		video_target_digest_seconds=None,
	)


def test_config_top_level_concat_enables(tmp_path: Path) -> None:
	config_path = tmp_path / "config.yaml"
	config_path.write_text(
		"""
model: test
base_url: http://localhost:11434
input: in
output: out
max_source_seconds: 10
concat_in_digest_folder: true
""".strip(),
		encoding="utf-8",
	)
	args = _base_args(config_path)
	_apply_config(args)
	assert args.concat_in_digest_folder is True


def test_config_nested_concat_enables(tmp_path: Path) -> None:
	config_path = tmp_path / "config.yaml"
	config_path.write_text(
		"""
model: test
base_url: http://localhost:11434
input: in
output: out
max_source_seconds: 10
video:
  concat_in_digest_folder: true
""".strip(),
		encoding="utf-8",
	)
	args = _base_args(config_path)
	_apply_config(args)
	assert args.concat_in_digest_folder is True


def test_cli_concat_overrides_config(tmp_path: Path) -> None:
	config_path = tmp_path / "config.yaml"
	config_path.write_text(
		"""
model: test
base_url: http://localhost:11434
input: in
output: out
max_source_seconds: 10
concat_in_digest_folder: false
""".strip(),
		encoding="utf-8",
	)
	args = _base_args(config_path)
	args.concat_in_digest_folder = True
	_apply_config(args)
	assert args.concat_in_digest_folder is True


def test_config_nested_delete_split_files_enables(tmp_path: Path) -> None:
	config_path = tmp_path / "config.yaml"
	config_path.write_text(
		"""
model: test
base_url: http://localhost:11434
input: in
output: out
max_source_seconds: 10
video:
  delete_split_files: true
""".strip(),
		encoding="utf-8",
	)
	args = _base_args(config_path)
	_apply_config(args)
	assert args.delete_split_files is True


def test_config_top_level_delete_split_files_enables(tmp_path: Path) -> None:
	config_path = tmp_path / "config.yaml"
	config_path.write_text(
		"""
model: test
base_url: http://localhost:11434
input: in
output: out
max_source_seconds: 10
delete_split_files: true
""".strip(),
		encoding="utf-8",
	)
	args = _base_args(config_path)
	_apply_config(args)
	assert args.delete_split_files is True


def test_cli_delete_split_files_overrides_config(tmp_path: Path) -> None:
	config_path = tmp_path / "config.yaml"
	config_path.write_text(
		"""
model: test
base_url: http://localhost:11434
input: in
output: out
max_source_seconds: 10
delete_split_files: false
""".strip(),
		encoding="utf-8",
	)
	args = _base_args(config_path)
	args.delete_split_files = True
	_apply_config(args)
	assert args.delete_split_files is True
