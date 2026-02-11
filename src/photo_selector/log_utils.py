from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict


LogFormat = str


def log_event(
	log_format: LogFormat,
	*,
	level: str,
	event_type: str,
	message: str,
	file_path: str | None = None,
	extra: Dict[str, Any] | None = None,
) -> None:
	timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
	payload: Dict[str, Any] = {
		"timestamp": timestamp,
		"level": level,
		"event_type": event_type,
		"file_path": file_path,
		"message": message,
	}
	if extra:
		payload.update(extra)

	if log_format == "json":
		print(json.dumps(payload, ensure_ascii=True))
		return

	message_parts = [f"{level.upper()}: {message}"]
	if file_path:
		message_parts.append(f"file={file_path}")
	for key in ("total_files", "processed", "skipped", "failed", "duration_seconds"):
		if key in payload:
			message_parts.append(f"{key}={payload[key]}")
	print(" ".join(message_parts))
