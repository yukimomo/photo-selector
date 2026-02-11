from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


@dataclass
class CachedScore:
	score: float
	analysis: Dict[str, Any] | None
	quality: Dict[str, Any] | None


class ScoreStore:
	def __init__(self, db_path: Path) -> None:
		self._db_path = db_path
		self._init_db()

	def get(self, file_path: str, file_hash: str) -> CachedScore | None:
		row = self._fetch_row(file_path)
		if row is None:
			return None

		stored_hash = row[1]
		if stored_hash != file_hash:
			return None

		analysis = self._load_json(row[3])
		quality = self._load_json(row[4])
		return CachedScore(score=float(row[2]), analysis=analysis, quality=quality)

	def upsert(
		self,
		file_path: str,
		file_hash: str,
		score: float,
		analysis: Dict[str, Any] | None,
		quality: Dict[str, Any] | None,
	) -> None:
		payload = (
			file_path,
			file_hash,
			float(score),
			self._dump_json(analysis),
			self._dump_json(quality),
			self._now_iso(),
		)
		with sqlite3.connect(self._db_path) as connection:
			connection.execute(
				"""
				INSERT INTO photo_scores (
					file_path,
					file_hash,
					score,
					analysis_json,
					quality_json,
					last_processed_at
				)
				VALUES (?, ?, ?, ?, ?, ?)
				ON CONFLICT(file_path) DO UPDATE SET
					file_hash = excluded.file_hash,
					score = excluded.score,
					analysis_json = excluded.analysis_json,
					quality_json = excluded.quality_json,
					last_processed_at = excluded.last_processed_at
				""",
				payload,
			)

	def _init_db(self) -> None:
		self._db_path.parent.mkdir(parents=True, exist_ok=True)
		with sqlite3.connect(self._db_path) as connection:
			connection.execute(
				"""
				CREATE TABLE IF NOT EXISTS photo_scores (
					file_path TEXT PRIMARY KEY,
					file_hash TEXT NOT NULL,
					score REAL NOT NULL,
					analysis_json TEXT,
					quality_json TEXT,
					last_processed_at TEXT NOT NULL
				)
				"""
			)

	def _fetch_row(self, file_path: str) -> tuple[Any, ...] | None:
		with sqlite3.connect(self._db_path) as connection:
			cursor = connection.execute(
				"""
				SELECT file_path, file_hash, score, analysis_json, quality_json, last_processed_at
				FROM photo_scores
				WHERE file_path = ?
				""",
				(file_path,),
			)
			return cursor.fetchone()

	@staticmethod
	def _dump_json(value: Dict[str, Any] | None) -> str | None:
		if value is None:
			return None
		return json.dumps(value, ensure_ascii=True)

	@staticmethod
	def _load_json(value: str | None) -> Dict[str, Any] | None:
		if not value:
			return None
		return json.loads(value)

	@staticmethod
	def _now_iso() -> str:
		return datetime.now(timezone.utc).isoformat(timespec="seconds")
