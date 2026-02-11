from __future__ import annotations

from typing import Any, Dict, List

from photo_selector.dedupe_utils import hash_to_int, is_near_duplicate


DEFAULT_NEAR_DUPLICATE_THRESHOLD = 8


def select_top_photos(photos: List[Dict[str, Any]], target_count: int) -> List[Dict[str, Any]]:
	eligible = [photo for photo in photos if _has_valid_score(photo)]
	ordered = sorted(eligible, key=lambda item: item["analysis"]["score"], reverse=True)

	clusters: List[List[Dict[str, Any]]] = []
	cluster_hashes: List[int] = []
	ungrouped: List[Dict[str, Any]] = []

	for photo in ordered:
		photo_hash = hash_to_int(photo.get("hash"))
		if photo_hash is None:
			ungrouped.append(photo)
			continue

		assigned = False
		for idx, base_hash in enumerate(cluster_hashes):
			if is_near_duplicate(photo_hash, [base_hash], DEFAULT_NEAR_DUPLICATE_THRESHOLD):
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


def select_photos_with_dedupe(
	photos: List[Dict[str, Any]],
	target_count: int,
	hamming_threshold: int,
	dedupe_enabled: bool,
) -> List[Dict[str, Any]]:
	eligible = [photo for photo in photos if _has_valid_score(photo)]
	ordered = sorted(eligible, key=lambda item: item["analysis"]["score"], reverse=True)
	if not dedupe_enabled:
		return ordered[:target_count]

	kept: list[Dict[str, Any]] = []
	kept_hashes: list[int] = []
	for photo in ordered:
		photo_hash = hash_to_int(photo.get("hash"))
		if photo_hash is not None and is_near_duplicate(
			photo_hash, kept_hashes, hamming_threshold
		):
			continue
		kept.append(photo)
		if photo_hash is not None:
			kept_hashes.append(photo_hash)
		if len(kept) >= target_count:
			break

	return kept
