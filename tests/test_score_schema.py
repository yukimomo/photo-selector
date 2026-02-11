from __future__ import annotations

from photo_selector.score_schema import normalize_analysis, parse_score_result


def test_parse_score_result_defaults() -> None:
	result = parse_score_result({})
	assert result.overall_score == 0.0
	assert result.sharpness == 0.0
	assert result.subject_visibility == 0.0
	assert result.composition == 0.0
	assert result.duplication_penalty == 0.0
	assert result.reasoning == ""


def test_normalize_analysis_clamps_and_maps_score() -> None:
	raw = {
		"score": "1.2",
		"tags": "not-a-list",
		"risks": {"blur": True},
	}
	analysis = normalize_analysis(raw)

	assert analysis["overall_score"] == 1.0
	assert analysis["score"] == 1.0
	assert analysis["sharpness"] == 0.0
	assert analysis["subject_visibility"] == 0.0
	assert analysis["composition"] == 0.0
	assert analysis["duplication_penalty"] == 0.0
	assert analysis["reasoning"] == ""
	assert isinstance(analysis["tags"], list)
	assert isinstance(analysis["risks"], dict)
