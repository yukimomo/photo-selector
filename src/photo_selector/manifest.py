from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_manifest(path: Path) -> Dict[str, Any]:
	if not path.exists():
		return {"photos": []}
	with path.open("r", encoding="utf-8") as handle:
		return json.load(handle)


def save_manifest(path: Path, data: Dict[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as handle:
		json.dump(data, handle, ensure_ascii=True, indent=2)
