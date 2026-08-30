"""Tests for the CVWS candidate pipeline's building blocks in
`src/vocabulary.py`: step 2 (`reduce_descriptors_mean_shift`), step 4
(`build_vocabulary_from_descriptors`), and the per-image `per_class_matrix`
that `compute_vocabulary_stats` now also populates (needed by
`selection_strategies.py`'s entropy-based ICC/CED formulas).

Uses small, well-separated synthetic descriptor blobs so Mean Shift and
K-means both converge to an unambiguous, easy-to-assert result without
requiring real images/SIFT/CNN extraction.
"""
from pathlib import Path

import numpy as np
import pytest

from src.config import PathsConfig, PipelineConfig
from src.vocabulary import (
    build_vocabulary_from_descriptors,
    compute_vocabulary_stats,
    encode_histograms,
    reduce_descriptors_mean_shift,
)


@pytest.fixture
def cfg(tmp_path) -> PipelineConfig:
    cfg = PipelineConfig(paths=PathsConfig(dataset_dir=str(tmp_path), output_dir=str(tmp_path / "outputs")))
    cfg.vocabulary.seed = 0
    cfg.vocabulary.minibatch = False  # exact KMeans: deterministic on tiny toy data
    return cfg


def _write_descriptor_files(tmp_path: Path, per_image: dict) -> dict:
    """Write {image_id: (N,D) ndarray} to .npy files, return {image_id: path}."""
    paths = {}
    desc_dir = tmp_path / "descriptors"
    desc_dir.mkdir(exist_ok=True)
    for image_id, arr in per_image.items():
        p = desc_dir / f"{image_id}.npy"
        np.save(p, arr)
        paths[image_id] = str(p)
    return paths


def test_reduce_descriptors_mean_shift_finds_two_well_separated_blobs(cfg, tmp_path):
    """Two far-apart, tight 2D blobs of descriptors should Mean-Shift down
    to (approximately) two candidate centroids, each close to its blob's
    true center -- a basic sanity check that step 2 actually reduces
    redundancy rather than just returning every input point."""
    rng = np.random.RandomState(0)
    blob_a = rng.normal(loc=[0.0, 0.0], scale=0.05, size=(60, 2))
    blob_b = rng.normal(loc=[10.0, 10.0], scale=0.05, size=(60, 2))
    descriptor_paths = _write_descriptor_files(
        tmp_path, {"img_a": blob_a.astype(np.float32), "img_b": blob_b.astype(np.float32)}
    )

    candidate_vocab = reduce_descriptors_mean_shift(
        descriptor_paths, cfg, __import__("logging").getLogger("test"), tmp_path / "candidates.pkl"
    )

    # Tight, far-apart blobs -> Mean Shift should collapse each to ~1
    # candidate (a handful at most is still acceptable), never anywhere
    # close to the 120 raw input descriptors.
    assert 1 <= candidate_vocab.k <= 4
    # Every candidate should land near one of the two true blob centers.
    centers = candidate_vocab.cluster_centers
    dist_to_a = np.linalg.norm(centers - np.array([0.0, 0.0]), axis=1)
    dist_to_b = np.linalg.norm(centers - np.array([10.0, 10.0]), axis=1)
    assert np.all(np.minimum(dist_to_a, dist_to_b) < 1.0)


def test_build_vocabulary_from_descriptors_matches_requested_k(cfg, tmp_path):
    rng = np.random.RandomState(1)
    descriptors = np.concatenate(
        [
            rng.normal(loc=[0.0, 0.0], scale=0.1, size=(20, 2)),
            rng.normal(loc=[5.0, 5.0], scale=0.1, size=(20, 2)),
            rng.normal(loc=[-5.0, 5.0], scale=0.1, size=(20, 2)),
        ]
    ).astype(np.float32)

    vocab = build_vocabulary_from_descriptors(
        descriptors, cfg, __import__("logging").getLogger("test"), tmp_path / "final_vocab.pkl", k=3
    )
    assert vocab.k == 3
    assert vocab.cluster_centers.shape == (3, 2)


def test_build_vocabulary_from_descriptors_rejects_too_few_candidates(cfg, tmp_path):
    descriptors = np.zeros((2, 4), dtype=np.float32)  # only 2 points, asking for K=5
    with pytest.raises(ValueError, match="n_candidates_per_class"):
        build_vocabulary_from_descriptors(
            descriptors, cfg, __import__("logging").getLogger("test"), tmp_path / "final_vocab.pkl", k=5
        )


def test_compute_vocabulary_stats_populates_per_class_matrix(tmp_path):
    """`per_class_matrix[c]` must stack the exact raw per-image histograms
    for class c, in the order they were encountered, so downstream entropy
    computations (`selection_strategies._entropy_intra`) see genuine
    per-image counts rather than only aggregates."""
    hist_dir = tmp_path / "hist"
    hist_dir.mkdir()
    histograms = {}
    labels = {}
    data = {
        "cat_1": np.array([1.0, 0.0, 3.0]),
        "cat_2": np.array([2.0, 0.0, 1.0]),
        "dog_1": np.array([0.0, 5.0, 0.0]),
    }
    for image_id, hist in data.items():
        p = hist_dir / f"{image_id}.npy"
        np.save(p, hist)
        histograms[image_id] = str(p)
        labels[image_id] = "cat" if image_id.startswith("cat") else "dog"

    stats = compute_vocabulary_stats(histograms, labels, k=3, logger=__import__("logging").getLogger("test"))

    assert stats.per_class_matrix["cat"].shape == (2, 3)
    assert stats.per_class_matrix["dog"].shape == (1, 3)
    assert stats.F["cat"].tolist() == [3.0, 0.0, 4.0]
    assert stats.F["dog"].tolist() == [0.0, 5.0, 0.0]
    # per_class_matrix rows must sum to F for internal consistency.
    assert stats.per_class_matrix["cat"].sum(axis=0).tolist() == stats.F["cat"].tolist()
