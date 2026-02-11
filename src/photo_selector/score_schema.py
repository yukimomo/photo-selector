from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class ScoreResult:
	overall_score: float
	sharpness: float
	subject_visibility: float
	composition: float
	duplication_penalty: float
	reasoning: str


def normalize_analysis(raw: Dict[str, Any]) -> Dict[str, Any]:
	if not isinstance(raw, dict):
		raise ValueError("Analysis is not a JSON object")

	analysis = dict(raw)
	analysis.setdefault("caption", "")
	analysis.setdefault("tags", [])
	analysis.setdefault("risks", {})

	if not isinstance(analysis.get("tags"), list):
		analysis["tags"] = []
	if not isinstance(analysis.get("risks"), dict):
		analysis["risks"] = {}

	for risk_key in ("blur", "dark", "overexposed", "out_of_focus"):
		analysis["risks"].setdefault(risk_key, False)

	score_result = parse_score_result(analysis)
	analysis["overall_score"] = score_result.overall_score
	analysis["sharpness"] = score_result.sharpness
	analysis["subject_visibility"] = score_result.subject_visibility
	analysis["composition"] = score_result.composition
	analysis["duplication_penalty"] = score_result.duplication_penalty
	analysis["reasoning"] = score_result.reasoning
	analysis["score"] = score_result.overall_score

	return analysis


def parse_score_result(raw: Dict[str, Any]) -> ScoreResult:
	overall = _coerce_float(raw.get("overall_score"))
	if overall is None:
		overall = _coerce_float(raw.get("score"))
	if overall is None:
		overall = 0.0

	sharpness = _coerce_float(raw.get("sharpness")) or 0.0
	subject_visibility = _coerce_float(raw.get("subject_visibility")) or 0.0
	composition = _coerce_float(raw.get("composition")) or 0.0
	duplication_penalty = _coerce_float(raw.get("duplication_penalty")) or 0.0
	reasoning = raw.get("reasoning") if isinstance(raw.get("reasoning"), str) else ""

	return ScoreResult(
		overall_score=_clamp01(overall),
		sharpness=_clamp01(sharpness),
		subject_visibility=_clamp01(subject_visibility),
		composition=_clamp01(composition),
		duplication_penalty=_clamp01(duplication_penalty),
		reasoning=reasoning.strip(),
	)


def _coerce_float(value: Any) -> float | None:
	if isinstance(value, (int, float)):
		return float(value)
	if isinstance(value, str):
		try:
			return float(value)
		except ValueError:
			return None
	return None


def _clamp01(value: float) -> float:
	if value < 0.0:
		return 0.0
	if value > 1.0:
		return 1.0
	return float(value)
