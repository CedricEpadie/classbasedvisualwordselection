"""Visual vocabulary construction (K-means) and histogram encoding.

`build_vocabulary` clusters all local descriptors from the training set
into `K` visual words (cluster centers). `VocabularyStats` then holds the
per-image, per-class frequency statistics `f(w, I)`, `F(w, c)`, `DF(w, c)`
(and, for the CVWS candidate pipeline, the full per-image count matrix)
needed by the selection strategies (section 3 of the spec) without those
strategies having to touch raw descriptors again.

Also provides the two extra building blocks used ONLY by the CVWS
candidate pipeline (bovw_cvws / cnn_bovw_cvws — see
`pipeline.PipelineRunner._run_cvws_variants`), steps 2 and 4 of the 5-step
vocabulary construction:
    reduce_descriptors_mean_shift   step 2: Mean Shift redundancy reduction
    build_vocabulary_from_descriptors   step 4: final K-means on an
                                         in-memory descriptor matrix (the
                                         selected candidates), as opposed
                                         to `build_vocabulary`, which loads
                                         descriptors from per-image files.
bovw_baseline / cnn_bovw are untouched: they still call `build_vocabulary`
directly on the full descriptor set D, with no Mean Shift step.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.cluster import KMeans, MeanShift, MiniBatchKMeans, estimate_bandwidth
from tqdm import tqdm

from src.config import PipelineConfig
from src.utils.io_utils import atomic_write_npy, atomic_write_pickle, ensure_dir, path_exists_and_valid, read_pickle


@dataclass
class Vocabulary:
    """A fitted K-means visual vocabulary."""

    k: int
    cluster_centers: np.ndarray  # (K, D)
    seed: int

    def assign(self, descriptors: np.ndarray) -> np.ndarray:
        """Vector-quantize descriptors: return, for each descriptor, the
        index of its nearest visual word. `descriptors`: (N, D)."""
        if descriptors.shape[0] == 0:
            return np.zeros((0,), dtype=np.int64)
        # Brute-force nearest-center assignment (fine for K in the
        # hundreds/low-thousands range typical of BoVW vocabularies).
        dists = np.linalg.norm(
            descriptors[:, None, :] - self.cluster_centers[None, :, :], axis=2
        )
        return np.argmin(dists, axis=1)

    def histogram(self, descriptors: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Frequency histogram f(w, I) over the K visual words for one image."""
        hist = np.zeros(self.k, dtype=np.float64)
        if descriptors.shape[0] > 0:
            assignments = self.assign(descriptors)
            counts = np.bincount(assignments, minlength=self.k)
            hist[: len(counts)] = counts
        if normalize and hist.sum() > 0:
            hist = hist / hist.sum()
        return hist


def _load_all_descriptors(descriptor_paths: Dict[str, str], desc: str) -> np.ndarray:
    """Load and concatenate every non-empty descriptor file into one (N, D)
    matrix. Shared by `build_vocabulary` and `reduce_descriptors_mean_shift`
    (both need "every training descriptor", just feeding different
    algorithms downstream)."""
    all_descriptors: List[np.ndarray] = []
    for path in tqdm(descriptor_paths.values(), desc=desc):
        d = np.load(path)
        if d.shape[0] > 0:
            all_descriptors.append(d)
    return np.concatenate(all_descriptors, axis=0)


def _fit_kmeans(descriptors: np.ndarray, cfg: PipelineConfig, k: int) -> np.ndarray:
    """Fit K-means/MiniBatchKMeans with `k` clusters on `descriptors`
    (N, D), honoring cfg.vocabulary.{minibatch,minibatch_batch_size,
    max_iter,seed}. Returns the (k, D) cluster centers. Shared by
    `build_vocabulary` (fits on the full descriptor set D) and
    `build_vocabulary_from_descriptors` (fits on the CVWS pipeline's
    selected candidate subset)."""
    if cfg.vocabulary.minibatch:
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=cfg.vocabulary.seed,
            batch_size=cfg.vocabulary.minibatch_batch_size,
            max_iter=cfg.vocabulary.max_iter,
            n_init="auto",
        )
    else:
        model = KMeans(n_clusters=k, random_state=cfg.vocabulary.seed, max_iter=cfg.vocabulary.max_iter, n_init="auto")
    model.fit(descriptors)
    return model.cluster_centers_


def build_vocabulary(
    train_descriptor_paths: Dict[str, str],
    cfg: PipelineConfig,
    logger: logging.Logger,
    out_path: Path,
) -> Vocabulary:
    """Fit K-means over all training descriptors and persist the vocabulary
    (idempotent: skipped if `out_path` already holds a valid vocabulary).

    Used as-is (no Mean Shift, no candidate selection) by bovw_baseline /
    cnn_bovw: K-means runs directly on the full descriptor set D. The
    *_cvws approaches instead go through `reduce_descriptors_mean_shift`
    (step 2) and `build_vocabulary_from_descriptors` (step 4) — see the
    module docstring.
    """
    if path_exists_and_valid(out_path):
        logger.info("Vocabulary cache hit: %s", out_path)
        return read_pickle(out_path)

    logger.info("Loading training descriptors to fit vocabulary...")
    stacked = _load_all_descriptors(train_descriptor_paths, "loading descriptors")
    logger.info("Fitting K-means on %d descriptors (K=%d)...", stacked.shape[0], cfg.vocabulary.k)

    centers = _fit_kmeans(stacked, cfg, cfg.vocabulary.k)
    vocab = Vocabulary(k=cfg.vocabulary.k, cluster_centers=centers, seed=cfg.vocabulary.seed)
    atomic_write_pickle(out_path, vocab)
    logger.info("Vocabulary saved to %s", out_path)
    return vocab


# --------------------------------------------------------------------------- #
# CVWS candidate pipeline: steps 2 and 4 (bovw_cvws / cnn_bovw_cvws only)
# --------------------------------------------------------------------------- #
def reduce_descriptors_mean_shift(
    train_descriptor_paths: Dict[str, str],
    cfg: PipelineConfig,
    logger: logging.Logger,
    out_path: Path,
) -> "Vocabulary":
    """Step 2 of the CVWS pipeline: 'Réduction de redondance par
    clustering hiérarchique (Mean Shift)'.

    Runs Mean Shift over EVERY training descriptor (no subsampling, by
    design) with bandwidth auto-estimated via sklearn's
    `estimate_bandwidth(..., quantile=cfg.vocabulary.mean_shift.quantile)`.
    The resulting cluster centers are the reduced, representative
    "candidate" descriptors that step 3 (per-class discriminative scoring,
    see `selection_strategies.py`) operates on.

    Returned as a `Vocabulary` purely so the existing `.assign()` /
    `.histogram()` / `encode_histograms` machinery can be reused unchanged
    to compute per-image candidate-occurrence counts in step 3 — its `k`
    is however many clusters Mean Shift converges to, which has nothing to
    do with `cfg.vocabulary.k` (the FINAL vocabulary size, produced later
    by `build_vocabulary_from_descriptors` in step 4).
    """
    if path_exists_and_valid(out_path):
        logger.info("Mean Shift candidates cache hit: %s", out_path)
        return read_pickle(out_path)

    logger.info("Loading training descriptors for Mean Shift reduction...")
    stacked = _load_all_descriptors(train_descriptor_paths, "loading descriptors (mean shift)")

    quantile = cfg.vocabulary.mean_shift.quantile
    logger.info("Estimating Mean Shift bandwidth (quantile=%.3f) over %d descriptors...", quantile, stacked.shape[0])
    # n_samples left at its sklearn default (None -> uses every row of
    # `stacked`): no subsampling, per design.
    bandwidth = estimate_bandwidth(stacked, quantile=quantile)
    if bandwidth <= 0:
        raise ValueError(
            f"estimate_bandwidth returned a non-positive bandwidth ({bandwidth}) for "
            f"quantile={quantile} -- try raising vocabulary.mean_shift.quantile."
        )

    logger.info("Fitting Mean Shift (bandwidth=%.6f)...", bandwidth)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=cfg.vocabulary.mean_shift.bin_seeding)
    ms.fit(stacked)
    candidates = ms.cluster_centers_
    logger.info("Mean Shift reduced %d descriptors to %d candidates", stacked.shape[0], candidates.shape[0])

    candidate_vocab = Vocabulary(k=candidates.shape[0], cluster_centers=candidates, seed=cfg.vocabulary.seed)
    atomic_write_pickle(out_path, candidate_vocab)
    return candidate_vocab


def build_vocabulary_from_descriptors(
    descriptors: np.ndarray,
    cfg: PipelineConfig,
    logger: logging.Logger,
    out_path: Path,
    k: Optional[int] = None,
) -> Vocabulary:
    """Step 4 of the CVWS pipeline: final K-means quantization.

    Like `build_vocabulary`, but fits directly on an in-memory descriptor
    matrix (the union of per-class-selected candidates from step 3, see
    `pipeline.PipelineRunner._run_cvws_variants`) instead of loading
    per-image descriptor files. `k` defaults to `cfg.vocabulary.k` — the
    SAME final vocabulary size used by bovw_baseline/cnn_bovw's direct
    K-means, so approaches stay comparable at equal feature dimension.
    """
    if path_exists_and_valid(out_path):
        logger.info("Vocabulary cache hit: %s", out_path)
        return read_pickle(out_path)

    k = cfg.vocabulary.k if k is None else k
    if descriptors.shape[0] < k:
        raise ValueError(
            f"Cannot fit the final K-means with K={k} on only {descriptors.shape[0]} "
            "selected candidate descriptors -- raise selection.n_candidates_per_class "
            "or lower vocabulary.k."
        )
    logger.info("Fitting final K-means on %d selected candidates (K=%d)...", descriptors.shape[0], k)
    centers = _fit_kmeans(descriptors, cfg, k)
    vocab = Vocabulary(k=k, cluster_centers=centers, seed=cfg.vocabulary.seed)
    atomic_write_pickle(out_path, vocab)
    logger.info("Final vocabulary saved to %s", out_path)
    return vocab


def encode_histograms(
    descriptor_paths: Dict[str, str],
    vocabulary: Vocabulary,
    cfg: PipelineConfig,
    logger: logging.Logger,
    out_dir: Path,
) -> Dict[str, str]:
    """Encode every image's descriptors into a normalized histogram over
    the vocabulary and persist as `.npy`. Idempotent per-image."""
    ensure_dir(out_dir)
    outputs: Dict[str, str] = {}
    n_done, n_skipped = 0, 0
    for image_id, desc_path in tqdm(descriptor_paths.items(), desc="encoding histograms"):
        out_path = out_dir / f"{image_id}.npy"
        outputs[image_id] = str(out_path)
        if path_exists_and_valid(out_path):
            n_skipped += 1
            continue
        descriptors = np.load(desc_path)
        hist = vocabulary.histogram(descriptors, normalize=False)  # raw counts; strategies need F(w,c)
        atomic_write_npy(out_path, hist)
        n_done += 1
    logger.info("Histogram encoding complete: %d computed, %d skipped", n_done, n_skipped)
    return outputs


# --------------------------------------------------------------------------- #
# Vocabulary-level statistics used by the selection strategies
# --------------------------------------------------------------------------- #
@dataclass
class VocabularyStats:
    """Per-class aggregate statistics over a vocabulary (the CVWS
    candidate set in step 3, or a final vocabulary), computed once from
    the raw (unnormalized) histograms of the training set. Feeds
    `selection_strategies.py`.

    Attributes:
        k: vocabulary size.
        classes: sorted list of class labels.
        F: F[c] -> (K,) array, F(w, c) = S_GCF(w, c) = total count of word w in class c.
        DF: DF[c] -> (K,) array, DF(w, c) = number of images of c containing w.
        n_images_per_class: n_c for each class.
        per_class_matrix: per_class_matrix[c] -> (N_c, K) array, the raw
            per-image occurrence counts of every word for every image of
            class c. F/DF alone (aggregate-only) are NOT enough to
            recompute the entropy terms H_intra/H_inter used by
            `IntraClassCorverage`/`ClassExclusivityDiscriminative` — this
            is why this per-image matrix is kept, unlike the
            aggregate-only stats sufficient for `GlobalClassFrequency`.
    """

    k: int
    classes: List[str]
    F: Dict[str, np.ndarray] = field(default_factory=dict)
    DF: Dict[str, np.ndarray] = field(default_factory=dict)
    n_images_per_class: Dict[str, int] = field(default_factory=dict)
    per_class_matrix: Dict[str, np.ndarray] = field(default_factory=dict)

    def df_out_of_class(self, c: str) -> np.ndarray:
        """DF(w, c-bar) = sum over c' != c of DF(w, c')."""
        total = np.zeros(self.k, dtype=np.float64)
        for c2 in self.classes:
            if c2 != c:
                total += self.DF[c2]
        return total

    def n_out_of_class(self, c: str) -> int:
        return sum(n for c2, n in self.n_images_per_class.items() if c2 != c)

    def rho_out_of_class(self, c: str) -> np.ndarray:
        """rho(w, c-bar) = DF(w, c-bar) / sum_{c' != c} n_{c'}."""
        denom = max(self.n_out_of_class(c), 1)
        return self.df_out_of_class(c) / denom

    def n_classes_containing(self) -> np.ndarray:
        """(K,) array: for each word w, number of classes c' with DF(w,c') > 0.
        Used by the TF-IDF variant of Method 3."""
        counts = np.zeros(self.k, dtype=np.float64)
        for c in self.classes:
            counts += (self.DF[c] > 0).astype(np.float64)
        return counts


def compute_vocabulary_stats(
    histogram_paths: Dict[str, str],
    labels: Dict[str, str],
    k: int,
    logger: logging.Logger,
) -> VocabularyStats:
    """Aggregate raw per-image histograms (counts, NOT normalized) into
    per-class F(w,c)/DF(w,c) statistics, plus the full per-image count
    matrix per class (per_class_matrix) needed for H_intra/H_inter."""
    classes = sorted(set(labels.values()))
    stats = VocabularyStats(k=k, classes=classes)
    per_class_rows: Dict[str, List[np.ndarray]] = {c: [] for c in classes}
    for c in classes:
        stats.F[c] = np.zeros(k, dtype=np.float64)
        stats.DF[c] = np.zeros(k, dtype=np.float64)
        stats.n_images_per_class[c] = 0

    for image_id, hist_path in tqdm(histogram_paths.items(), desc="aggregating vocabulary stats"):
        c = labels[image_id]
        hist = np.load(hist_path)
        stats.F[c] += hist
        stats.DF[c] += (hist > 0).astype(np.float64)
        stats.n_images_per_class[c] += 1
        per_class_rows[c].append(hist)

    for c in classes:
        stats.per_class_matrix[c] = (
            np.stack(per_class_rows[c], axis=0) if per_class_rows[c] else np.zeros((0, k), dtype=np.float64)
        )

    logger.info("Vocabulary stats computed over %d classes", len(classes))
    return stats
