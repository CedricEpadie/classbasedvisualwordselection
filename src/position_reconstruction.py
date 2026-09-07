"""Per-position reconstruction pipeline for cnn_bovw_cvws / vit_cvws (2026-09
methodology update). Unlike bovw_cvws (whose local descriptors -- SIFT
keypoints -- have no fixed count or order per image, so its candidates feed a
single, position-agnostic final K-means, see `cvws_clustering.py`), CNN/ViT
local descriptors have a FIXED count and a FIXED, meaningful order per image:
row p is always "the activation/patch at spatial position p", the same p
across every image. That structure is exploited here instead of being
discarded:

    1. Steps 1-5 (preprocessing, extraction with provenance, UMAP, HDBSCAN,
       noise removal) are UNCHANGED -- see `cvws_clustering.reduce_and_cluster`.
       Every surviving descriptor still carries a global HDBSCAN cluster id,
       exactly as in bovw_cvws ("Toujours avec la logique d'identifiant de
       cluster").
    2. Descriptors are re-grouped into one "bag" per position
       (`group_rows_by_position`): position p's bag holds every surviving
       descriptor that was extracted at row p of its own image, across every
       image and every class.
    3. Each bag is scored per class exactly like step 6-7 of bovw_cvws
       (`compute_positional_stats` + the SAME `selection_strategies.py`
       GCF/ICC/CED strategies, unchanged), keeping each class's own top
       `position_top_n` cluster ids AT THAT POSITION
       (`rank_position_words`).
    4. For each class, `images_per_class` (y) synthetic training "images"
       are reconstructed: image j's descriptor at position p is the j-th
       entry of position p's ranked word list for that class, cycling
       through the ranking (highest-scoring first) once y exceeds the
       number of selected words -- `allocation.pick_with_duplication`
       (`build_reconstructed_images`). Each selected (position, cluster id)
       is represented by a single canonical vector: the ORIGINAL descriptor
       of that cluster's highest-HDBSCAN-probability member at that
       position (`build_position_representatives`) -- reused across every
       synthetic image that needs that word at that position, the same way
       bovw_cvws's step 8 reuses a cluster's nearest-medoid member.

    Steps 5 (encoding the reconstructed (y, n_positions, D) tensors into
    vector representations via the CNN/ViT "tail", i.e. everything from the
    hooked local layer onward) and 6 (classification) live in
    `tail_encoders.py` and `pipeline.PipelineRunner._run_position_cvws_variant`.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Set, Tuple

import numpy as np

from src.allocation import pick_with_duplication
from src.cvws_clustering import ClusteringResult
from src.selection_strategies import VisualWordSelectionStrategy
from src.vocabulary import VocabularyStats


# --------------------------------------------------------------------------- #
# Step 2: per-position bags
# --------------------------------------------------------------------------- #
def group_rows_by_position(clustering_result: ClusteringResult) -> Dict[int, np.ndarray]:
    """Bag of surviving-descriptor row indices per position, i.e.
    `clustering_result.local_indices` value (CNN/ViT local descriptors have
    one row per fixed spatial position per image, unlike SIFT's variable-
    length keypoint sets -- see the module docstring)."""
    bags: Dict[int, List[int]] = {}
    for row_idx, position in enumerate(clustering_result.local_indices):
        bags.setdefault(int(position), []).append(row_idx)
    return {p: np.asarray(rows, dtype=np.int64) for p, rows in bags.items()}


# --------------------------------------------------------------------------- #
# Step 3: per-position, per-class cluster-occurrence stats + ranking
# --------------------------------------------------------------------------- #
def compute_positional_stats(
    clustering_result: ClusteringResult,
    labels: Dict[str, str],
    position: int,
    position_rows: np.ndarray,
) -> VocabularyStats:
    """The step-6/7 `VocabularyStats` for ONE position: for every training
    image (every id appearing anywhere in `clustering_result.image_ids`),
    whether ITS OWN descriptor at `position` survived HDBSCAN and, if so,
    which cluster it landed in. An image whose descriptor at this position
    was dropped as noise contributes an all-zero row (same "absent word"
    convention as `cvws_clustering.compute_cluster_stats`) -- it is still
    counted in `n_images_per_class`, it simply has no word here.

    `position_rows`: `group_rows_by_position(clustering_result)[position]`,
    passed in rather than recomputed so the caller (which needs it for
    every position anyway, see `select_all_position_words`) only builds it
    once per position.
    """
    train_image_ids = sorted(set(clustering_result.image_ids))
    classes = sorted({labels[i] for i in train_image_ids})
    k = clustering_result.n_clusters
    stats = VocabularyStats(k=k, classes=classes)
    per_class_rows: Dict[str, List[np.ndarray]] = {c: [] for c in classes}
    for c in classes:
        stats.F[c] = np.zeros(k, dtype=np.float64)
        stats.DF[c] = np.zeros(k, dtype=np.float64)
        stats.n_images_per_class[c] = 0

    cluster_at_position_by_image: Dict[str, int] = {
        clustering_result.image_ids[row_idx]: int(clustering_result.cluster_labels[row_idx])
        for row_idx in position_rows
    }

    for image_id in train_image_ids:
        c = labels[image_id]
        hist = np.zeros(k, dtype=np.float64)
        if image_id in cluster_at_position_by_image:
            hist[cluster_at_position_by_image[image_id]] = 1.0
        stats.F[c] += hist
        stats.DF[c] += (hist > 0).astype(np.float64)
        stats.n_images_per_class[c] += 1
        per_class_rows[c].append(hist)

    for c in classes:
        stats.per_class_matrix[c] = (
            np.stack(per_class_rows[c], axis=0) if per_class_rows[c] else np.zeros((0, k), dtype=np.float64)
        )
    return stats


def rank_position_words(
    strategy: VisualWordSelectionStrategy, stats: VocabularyStats, class_label: str, top_n: int
) -> List[int]:
    """The (up to) `top_n` cluster ids with the highest score(v, class_label)
    at this position, ORDERED highest-scoring first -- unlike
    `strategy.select_top_n` (returns an unordered `Set`, fine for bovw_cvws's
    step 8 where only membership matters), the rank order here is exactly
    what `build_reconstructed_images`'s cyclic fill needs.

    A cluster with score 0 (never occurs at this position for this class,
    the overwhelmingly common case -- most classes have a handful of
    genuinely position-and-class-specific clusters, not `top_n` of them) is
    EXCLUDED rather than padded in: there is no real descriptor to
    represent it, so including it would just insert an arbitrary/empty
    word. It is normal, not an error, for this to return fewer than
    `top_n` ids, or even zero (see `build_reconstructed_images`'s handling
    of that case).
    """
    score_vector = strategy.score_matrix(stats)[class_label]
    order = np.argsort(-score_vector, kind="stable")
    return [int(i) for i in order[:top_n] if score_vector[i] > 0]


def select_all_position_words(
    strategy: VisualWordSelectionStrategy,
    clustering_result: ClusteringResult,
    labels: Dict[str, str],
    n_positions: int,
    top_n: int,
    logger: logging.Logger,
) -> Dict[int, Dict[str, List[int]]]:
    """Step 3 end to end: `{position: {class: ranked_cluster_ids}}` for
    every position `0..n_positions-1`."""
    position_bags = group_rows_by_position(clustering_result)
    result: Dict[int, Dict[str, List[int]]] = {}
    for position in range(n_positions):
        rows = position_bags.get(position, np.zeros((0,), dtype=np.int64))
        stats = compute_positional_stats(clustering_result, labels, position, rows)
        result[position] = {c: rank_position_words(strategy, stats, c, top_n) for c in stats.classes}
    n_empty = sum(1 for by_class in result.values() for ids in by_class.values() if not ids)
    logger.info(
        "Per-position selection (%s): %d positions x %d classes, %d (position, class) pairs with no selected word",
        strategy.name,
        n_positions,
        len(next(iter(result.values()), {})),
        n_empty,
    )
    return result


# --------------------------------------------------------------------------- #
# Step 4a: one canonical representative vector per selected (position, cluster)
# --------------------------------------------------------------------------- #
def build_position_representatives(
    clustering_result: ClusteringResult,
    descriptor_paths: Dict[str, str],
    position_words: Dict[int, Dict[str, List[int]]],
) -> Dict[Tuple[int, int], np.ndarray]:
    """For every (position, cluster id) selected by at least one class in
    `position_words`, the ORIGINAL (pre-UMAP) descriptor vector of that
    cluster's highest-HDBSCAN-probability member AT THAT POSITION -- reused
    for every synthetic image that needs that word at that position (see
    `build_reconstructed_images`)."""
    needed: Set[Tuple[int, int]] = set()
    for position, by_class in position_words.items():
        for ranked_ids in by_class.values():
            needed.update((position, cid) for cid in ranked_ids)
    if not needed:
        return {}

    position_bags = group_rows_by_position(clustering_result)
    needed_positions = {p for p, _ in needed}
    members_by_pair: Dict[Tuple[int, int], List[int]] = {}
    for position in needed_positions:
        for row_idx in position_bags.get(position, np.zeros((0,), dtype=np.int64)):
            cid = int(clustering_result.cluster_labels[row_idx])
            if (position, cid) in needed:
                members_by_pair.setdefault((position, cid), []).append(int(row_idx))

    loaded_by_image: Dict[str, np.ndarray] = {}
    representatives: Dict[Tuple[int, int], np.ndarray] = {}
    for pair, rows in members_by_pair.items():
        best_row = max(rows, key=lambda r: clustering_result.probabilities[r])
        image_id = clustering_result.image_ids[best_row]
        local_idx = int(clustering_result.local_indices[best_row])
        if image_id not in loaded_by_image:
            loaded_by_image[image_id] = np.load(descriptor_paths[image_id])
        representatives[pair] = loaded_by_image[image_id][local_idx]
    return representatives


# --------------------------------------------------------------------------- #
# Step 4b: reconstruct y synthetic images per class
# --------------------------------------------------------------------------- #
def build_reconstructed_images(
    position_words: Dict[int, Dict[str, List[int]]],
    representatives: Dict[Tuple[int, int], np.ndarray],
    classes: List[str],
    n_positions: int,
    descriptor_dim: int,
    images_per_class: int,
    logger: logging.Logger,
) -> Dict[str, np.ndarray]:
    """`{class: (images_per_class, n_positions, descriptor_dim) array}` --
    image j (row j) at position p is `representatives[(p, word_ids[j %
    len(word_ids)])]`, `word_ids = position_words[p][class]` (highest-
    scoring first, cycled via `allocation.pick_with_duplication` once
    `images_per_class` exceeds `len(word_ids)` -- the spec's "on répète les
    mots clé en ordre de pertinence jusqu'à atteindre y").

    A (position, class) pair with NO selected word (see
    `rank_position_words`'s docstring -- normal, not an error) leaves every
    synthetic image of that class with an all-zero vector at that position,
    the same "absent word" convention used throughout this framework.
    """
    n_missing = 0
    out: Dict[str, np.ndarray] = {}
    for c in classes:
        tensor = np.zeros((images_per_class, n_positions, descriptor_dim), dtype=np.float32)
        for position in range(n_positions):
            word_ids = position_words.get(position, {}).get(c, [])
            if not word_ids:
                n_missing += 1
                continue
            cycled = pick_with_duplication(word_ids, images_per_class)
            for j, cid in enumerate(cycled):
                vec = representatives.get((position, cid))
                if vec is not None:
                    tensor[j, position, :] = vec
        out[c] = tensor
    if n_missing:
        logger.warning(
            "%d (position, class) pairs had no selected word; left as all-zero for every synthetic image of "
            "that class at that position",
            n_missing,
        )
    return out
