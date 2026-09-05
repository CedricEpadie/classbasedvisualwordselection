"""CVWS candidate pipeline, steps 3-8 of "Pipeline de construction d'un
vocabulaire visuel par sélection de mots visuels basée sur la classe"
(C. Epadie, Aug. 2026):

    3. UMAP dimensionality reduction (d = vocabulary.umap.n_components)
    4. HDBSCAN clustering (cosine) on the UMAP embedding
    5. Noise removal (HDBSCAN label -1 dropped)
    6. Symbolic re-encoding: each training image -> per-cluster occurrence
       counts (reuses `vocabulary.VocabularyStats`/`selection_strategies.py`
       UNCHANGED -- GCF/ICC/CED score cluster ids exactly like they used
       to score raw candidate descriptors)
    7. Per-class Top-N cluster selection (`strategy.select_top_n`, already
       generic -- no changes needed there either)
    8. Weighted candidate reconstruction: a cluster selected by
       `selection_count` distinct classes contributes its
       `selection_count` highest-HDBSCAN-probability member descriptors
       (nearest its medoid) to the final candidate pool, in their
       ORIGINAL (pre-UMAP) descriptor space

Steps 1-2 (preprocessing, extraction+provenance) and 9-11 (final cosine
K-means, histogram encoding, classification) live in `vocabulary.py`
(`build_vocabulary_from_descriptors(..., metric="cosine")`,
`encode_histograms`) and `pipeline.PipelineRunner._run_cvws_variants`,
which orchestrates the full 11-step pipeline end to end.

Cosine distance is required by the spec for both the UMAP/HDBSCAN stage
(step 3-4) and the final K-means (step 9), but neither UMAP's internal
neighbor search as used here, HDBSCAN, nor K-means natively support it as
cheaply as Euclidean in scikit-learn (HDBSCAN's cosine mode forces an
O(N^2) brute-force search; K-means has no cosine variant at all). Both are
implemented via the standard, mathematically-equivalent trick of
L2-normalizing vectors and then running the plain Euclidean algorithm: for
unit vectors, ||a-b||^2 = 2 - 2*cos(a,b), so Euclidean nearest-neighbor /
clustering on normalized vectors ranks identically to cosine distance, at
a fraction of the cost (real KD-tree/ball-tree search stays available).
Descriptors are normalized once before UMAP (so its embedding reflects
cosine neighborhoods) and the UMAP OUTPUT is re-normalized before HDBSCAN
(UMAP's embedding doesn't itself preserve input norms) — see
`reduce_and_cluster`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import hdbscan
from tqdm import tqdm

from src.config import PipelineConfig
from src.selection_strategies import VisualWordSelectionStrategy
from src.utils.io_utils import path_exists_and_valid, read_pickle, atomic_write_pickle
from src.vocabulary import VocabularyStats, l2_normalize


@dataclass
class ClusteringResult:
    """Output of steps 3-5 (UMAP -> HDBSCAN -> noise removal), for the
    TRAINING descriptors only (per confirmed design: UMAP/HDBSCAN are fit
    on train descriptors exclusively, consistent with every other
    vocabulary-building step in this framework — test images are only
    ever encoded later, against the FINAL vocabulary from step 9).

    One row per SURVIVING (non-noise) training descriptor; noise (HDBSCAN
    label -1) is already dropped, and `cluster_labels` already remapped to
    a contiguous 0..n_clusters-1 range.

    `image_ids[i]` / `local_indices[i]` together give the "provenance"
    required by steps 2 and 9: `descriptor_paths[image_ids[i]]`, row
    `local_indices[i]`, is the ORIGINAL (pre-UMAP, pre-normalization)
    descriptor vector for row i here — this is how step 8/9 map selected
    candidates back to the original descriptor space without having to
    store a second copy of every descriptor.
    """

    n_clusters: int
    image_ids: List[str]  # len N_survivors
    local_indices: np.ndarray  # (N_survivors,) int32 -- row index into that image's own descriptor .npy
    cluster_labels: np.ndarray  # (N_survivors,) int32, contiguous 0..n_clusters-1
    probabilities: np.ndarray  # (N_survivors,) float32, HDBSCAN's own soft cluster-membership strength


def _load_descriptors_with_provenance(
    train_descriptor_paths: Dict[str, str], desc: str
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Load every training descriptor, keeping track of which image (and
    which row within that image's own .npy file) each one came from —
    step 2's "provenance image-descripteur" requirement."""
    image_ids: List[str] = []
    local_indices: List[int] = []
    chunks: List[np.ndarray] = []
    for image_id, path in tqdm(train_descriptor_paths.items(), desc=desc):
        d = np.load(path)
        if d.shape[0] == 0:
            continue
        chunks.append(d)
        image_ids.extend([image_id] * d.shape[0])
        local_indices.extend(range(d.shape[0]))
    stacked = np.concatenate(chunks, axis=0)
    return stacked, image_ids, np.asarray(local_indices, dtype=np.int32)


def reduce_and_cluster(
    train_descriptor_paths: Dict[str, str],
    cfg: PipelineConfig,
    logger: logging.Logger,
    out_path: Path,
) -> ClusteringResult:
    """Steps 3-5: UMAP -> HDBSCAN (cosine) -> drop noise.

    Fits on EVERY training descriptor (no subsampling, consistent with
    every other vocabulary-building step in this framework).
    """
    if path_exists_and_valid(out_path):
        logger.info("UMAP/HDBSCAN clustering cache hit: %s", out_path)
        return read_pickle(out_path)

    # Lazy import: umap-learn is an optional-at-import-time dependency
    # (like torch/torchvision for ViT), only required if this step
    # actually runs — mirrors `ViTExtractor`/`CnnExtractor`'s pattern.
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "The CVWS pipeline's UMAP step (step 3) requires the 'umap-learn' "
            "package: pip install umap-learn"
        ) from exc

    logger.info("Loading training descriptors (with provenance) for UMAP/HDBSCAN...")
    stacked, image_ids, local_indices = _load_descriptors_with_provenance(
        train_descriptor_paths, "loading descriptors (provenance)"
    )
    n_total = stacked.shape[0]
    logger.info("Loaded %d training descriptors", n_total)

    # Cosine, part 1/2: normalize the ORIGINAL descriptors before UMAP so
    # its (Euclidean) internal neighbor search approximates cosine.
    normalized = l2_normalize(stacked.astype(np.float32))

    n_components = cfg.vocabulary.umap.n_components
    logger.info("Fitting UMAP (n_components=%d) on %d descriptors...", n_components, n_total)
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=cfg.vocabulary.umap.n_neighbors,
        min_dist=cfg.vocabulary.umap.min_dist,
        metric="euclidean",  # input already L2-normalized -> ranks like cosine
        random_state=cfg.vocabulary.seed,
        n_jobs=1,  # required by umap-learn for a reproducible random_state
    )
    d_umap = reducer.fit_transform(normalized)

    # Cosine, part 2/2: UMAP's own embedding doesn't preserve input norms,
    # so re-normalize its output before HDBSCAN.
    d_umap_norm = l2_normalize(np.asarray(d_umap, dtype=np.float32))

    min_cluster_size = cfg.vocabulary.hdbscan.min_cluster_size
    min_samples = cfg.vocabulary.hdbscan.min_samples
    logger.info(
        "Fitting HDBSCAN (min_cluster_size=%d, min_samples=%d) on %d UMAP-reduced descriptors...",
        min_cluster_size,
        min_samples,
        n_total,
    )
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        leaf_size=80,
        approx_min_span_tree=True,
        cluster_selection_method='leaf'
    )
    raw_labels = np.asarray(clusterer.labels_)
    raw_probabilities = np.asarray(clusterer.probabilities_)

    # Step 5: drop noise (label == -1), then remap surviving labels to a
    # contiguous 0..n_clusters-1 range.
    keep_mask = raw_labels != -1
    unique_labels = sorted(set(raw_labels[keep_mask].tolist()))
    n_clusters_found = len(unique_labels)
    if n_clusters_found == 0:
        raise ValueError(
            "HDBSCAN found no clusters (every descriptor was labeled as noise) -- "
            "try lowering vocabulary.hdbscan.min_cluster_size/min_samples."
        )
    remap = {old: new for new, old in enumerate(unique_labels)}
    remapped_labels = np.array([remap[label] for label in raw_labels[keep_mask]], dtype=np.int32)
    keep_idx = np.where(keep_mask)[0]

    result = ClusteringResult(
        n_clusters=n_clusters_found,
        image_ids=[image_ids[i] for i in keep_idx],
        local_indices=local_indices[keep_idx],
        cluster_labels=remapped_labels,
        probabilities=raw_probabilities[keep_mask].astype(np.float32),
    )
    n_noise = int((~keep_mask).sum())
    logger.info(
        "HDBSCAN found %d clusters over %d descriptors (%d noise points dropped, %.1f%%)",
        n_clusters_found,
        n_total,
        n_noise,
        100.0 * n_noise / n_total,
    )
    atomic_write_pickle(out_path, result)
    return result


def compute_cluster_stats(
    clustering_result: ClusteringResult,
    labels: Dict[str, str],
    logger: logging.Logger,
) -> VocabularyStats:
    """Step 6 (symbolic re-encoding) + per-class aggregation in one pass:
    for every TRAINING image, count how many of its surviving descriptors
    fall into each cluster — directly from HDBSCAN's own `labels_`, no
    nearest-centroid search needed (unlike the old Mean-Shift candidates)
    since every training descriptor already has a hard cluster assignment.

    Returns the SAME `VocabularyStats` structure `selection_strategies.py`
    already consumes, unchanged: Methods 1-3 (GCF/ICC/CED) run exactly as
    before, just scored over HDBSCAN cluster ids instead of Mean-Shift
    candidate indices.
    """
    classes = sorted(set(labels.values()))
    k = clustering_result.n_clusters
    stats = VocabularyStats(k=k, classes=classes)
    per_class_rows: Dict[str, List[np.ndarray]] = {c: [] for c in classes}
    for c in classes:
        stats.F[c] = np.zeros(k, dtype=np.float64)
        stats.DF[c] = np.zeros(k, dtype=np.float64)
        stats.n_images_per_class[c] = 0

    # Group surviving descriptor rows by image.
    per_image_labels: Dict[str, List[int]] = {}
    for image_id, cluster_id in zip(clustering_result.image_ids, clustering_result.cluster_labels):
        per_image_labels.setdefault(image_id, []).append(int(cluster_id))

    for image_id, cluster_ids in tqdm(per_image_labels.items(), desc="aggregating cluster stats"):
        c = labels[image_id]
        hist = np.bincount(cluster_ids, minlength=k).astype(np.float64)
        stats.F[c] += hist
        stats.DF[c] += (hist > 0).astype(np.float64)
        stats.n_images_per_class[c] += 1
        per_class_rows[c].append(hist)

    for c in classes:
        stats.per_class_matrix[c] = (
            np.stack(per_class_rows[c], axis=0) if per_class_rows[c] else np.zeros((0, k), dtype=np.float64)
        )

    logger.info("Cluster stats computed over %d classes, %d clusters", len(classes), k)
    return stats


def select_clusters_and_build_candidates(
    clustering_result: ClusteringResult,
    stats: VocabularyStats,
    strategy: VisualWordSelectionStrategy,
    n_candidates_per_class: int,
    descriptor_paths: Dict[str, str],
    logger: logging.Logger,
) -> Tuple[np.ndarray, dict]:
    """Steps 7-8: per-class Top-N cluster selection (step 7, reusing
    `strategy.select_top_n` unchanged) + weighted candidate reconstruction
    (step 8): each cluster selected by `selection_count` distinct classes
    contributes its `selection_count` highest-probability member
    descriptors (their ORIGINAL, pre-UMAP vectors) to the final candidate
    pool. If a cluster's `selection_count` exceeds its own member count,
    every member is kept (confirmed edge-case handling: clamp, don't error).

    Returns `(candidate_descriptors, audit_info)`:
      candidate_descriptors: (M, D_orig) array, original-descriptor-space
          candidates feeding step 9's final K-means.
      audit_info: per-class selected cluster ids + per-cluster selection
          counts, persisted to `outputs/selection/*.json` for interpretability.
    """
    # --- Step 7: per-class Top-N cluster ids ---
    per_class_selected: Dict[str, List[int]] = {
        c: sorted(int(v) for v in strategy.select_top_n(stats, c, n_candidates_per_class)) for c in stats.classes
    }

    # --- Step 8a: selection_count per cluster (# distinct classes that picked it) ---
    selection_count: Dict[int, int] = {}
    for cluster_ids in per_class_selected.values():
        for cid in cluster_ids:
            selection_count[cid] = selection_count.get(cid, 0) + 1

    # --- Step 8b: group surviving descriptor rows by cluster id (only for
    # selected clusters -- no need to bucket the rest) ---
    cluster_to_rows: Dict[int, List[int]] = {}
    for row_idx, cid in enumerate(clustering_result.cluster_labels):
        cid = int(cid)
        if cid in selection_count:
            cluster_to_rows.setdefault(cid, []).append(row_idx)

    chosen_rows: List[int] = []
    per_cluster_audit: Dict[str, dict] = {}
    for cid, n in selection_count.items():
        member_rows = cluster_to_rows.get(cid, [])
        # Highest HDBSCAN membership probability first (nearest the medoid).
        member_rows_sorted = sorted(member_rows, key=lambda r: -clustering_result.probabilities[r])
        take = min(n, len(member_rows_sorted))  # clamp to cluster size, per confirmed edge-case handling
        picked = member_rows_sorted[:take]
        chosen_rows.extend(picked)
        per_cluster_audit[str(cid)] = {
            "selection_count": n,
            "cluster_size": len(member_rows_sorted),
            "n_taken": take,
        }

    # --- Gather the ORIGINAL (pre-UMAP) descriptor vectors for the chosen rows ---
    rows_by_image: Dict[str, List[int]] = {}
    for row_idx in chosen_rows:
        image_id = clustering_result.image_ids[row_idx]
        local_idx = int(clustering_result.local_indices[row_idx])
        rows_by_image.setdefault(image_id, []).append(local_idx)

    candidate_vectors: List[np.ndarray] = []
    for image_id, local_idxs in tqdm(rows_by_image.items(), desc="gathering original-space candidates"):
        d = np.load(descriptor_paths[image_id])
        candidate_vectors.append(d[local_idxs])
    candidate_descriptors = (
        np.concatenate(candidate_vectors, axis=0) if candidate_vectors else np.zeros((0, 0), dtype=np.float32)
    )

    logger.info(
        "CVWS candidate reconstruction (%s): %d clusters selected -> %d candidate descriptors",
        strategy.name,
        len(selection_count),
        candidate_descriptors.shape[0],
    )

    audit_info = {
        "strategy": strategy.name,
        "n_candidates_per_class": n_candidates_per_class,
        "n_total_clusters": clustering_result.n_clusters,
        "per_class_selected_clusters": per_class_selected,
        "per_cluster_selection": per_cluster_audit,
        "n_candidate_descriptors": int(candidate_descriptors.shape[0]),
    }
    return candidate_descriptors, audit_info
