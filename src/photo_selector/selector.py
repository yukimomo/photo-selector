from __future__ import annotations

from typing import Any, Dict, List


DEFAULT_HAMMING_THRESHOLD = 12
DEFAULT_NEAR_DUPLICATE_THRESHOLD = 8


def select_top_photos(photos: List[Dict[str, Any]], target_count: int) -> List[Dict[str, Any]]:
	eligible = [photo for photo in photos if _has_valid_score(photo)]
	ordered = sorted(eligible, key=lambda item: item["analysis"]["score"], reverse=True)

	clusters: List[List[Dict[str, Any]]] = []
	cluster_hashes: List[int] = []
	ungrouped: List[Dict[str, Any]] = []

	for photo in ordered:
		photo_hash = _hash_to_int(photo.get("hash"))
		if photo_hash is None:
			ungrouped.append(photo)
			continue

		assigned = False
		for idx, base_hash in enumerate(cluster_hashes):
			if _hamming_distance(photo_hash, base_hash) <= DEFAULT_NEAR_DUPLICATE_THRESHOLD:
				clusters[idx].append(photo)
				assigned = True
				break
		if not assigned:
			clusters.append([photo])
			cluster_hashes.append(photo_hash)

	cluster_best = [cluster[0] for cluster in clusters if cluster]
	cluster_best.extend(ungrouped)
	cluster_best = sorted(
		cluster_best, key=lambda item: item["analysis"]["score"], reverse=True
	)

	if target_count >= len(cluster_best):
		return cluster_best

	return cluster_best[:target_count]


def _has_valid_score(photo: Dict[str, Any]) -> bool:
	if photo.get("error"):
		return False
	analysis = photo.get("analysis")
	return isinstance(analysis, dict) and isinstance(analysis.get("score"), (int, float))


def _hash_to_int(value: Any) -> int | None:
	if isinstance(value, int):
		return value
	if isinstance(value, str):
		try:
			return int(value, 16)
		except ValueError:
			return None
	return None


def _is_similar(candidate: int, selected: List[int]) -> bool:
	for existing in selected:
		distance = _hamming_distance(candidate, existing)
		if distance <= DEFAULT_HAMMING_THRESHOLD:
			return True
	return False


def _hamming_distance(left: int, right: int) -> int:
	return (left ^ right).bit_count()
