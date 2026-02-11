from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def load_config(path: Path) -> Dict[str, Any]:
	if not path.exists():
		raise FileNotFoundError(f"Config file not found: {path}")

	try:
		import yaml  # type: ignore
	except Exception as exc:  # noqa: BLE001
		raise RuntimeError(
			"PyYAML is required for --config. Install with: pip install pyyaml"
		) from exc

	data = yaml.safe_load(path.read_text(encoding="utf-8"))
	if data is None:
		return {}
	if not isinstance(data, dict):
		raise ValueError("Config must be a YAML mapping")
	return data


def coerce_bool(value: Any) -> bool | None:
	if isinstance(value, bool):
		return value
	if isinstance(value, (int, float)):
		return bool(value)
	if isinstance(value, str):
		value = value.strip().lower()
		if value in {"1", "true", "yes", "on"}:
			return True
		if value in {"0", "false", "no", "off"}:
			return False
	return None
