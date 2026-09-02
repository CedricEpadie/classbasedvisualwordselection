"""Tests for the CVWS pipeline's final-quantization building block in
`src/vocabulary.py`: step 9 (`build_vocabulary_from_descriptors`,
including its `metric="cosine"` mode -- see `Vocabulary.assign`), and the
per-image `per_class_matrix` that `compute_vocabulary_stats` also
populates (needed by `selection_strategies.py`'s entropy-based ICC/CED
formulas). Steps 3-8 (UMAP/HDBSCAN/selection) live in
`cvws_clustering.py` -- see `tests/test_cvws_clustering.py`.

Uses small, well-separated synthetic descriptor blobs so K-means converges
to an unambiguous, easy-to-assert result without requiring real
images/SIFT/CNN extraction.
"""
from pathlib import Path

import numpy as np
import pytest

from src.config import PathsConfig, PipelineConfig
from src.vocabulary import Vocabulary, build_vocabulary_from_descriptors, compute_vocabulary_stats, l2_normalize


@pytest.fixture
def cfg(tmp_path) -> PipelineConfig:
    cfg = PipelineConfig(paths=PathsConfig(dataset_dir=str(tmp_path), output_dir=str(tmp_path / "outputs")))
    cfg.vocabulary.seed = 0
    cfg.vocabulary.minibatch = False  # exact KMeans: deterministic on tiny toy data
    return cfg


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
    assert vocab.metric == "euclidean"  # default, unchanged behavior


def test_build_vocabulary_from_descriptors_rejects_too_few_candidates(cfg, tmp_path):
    descriptors = np.zeros((2, 4), dtype=np.float32)  # only 2 points, asking for K=5
    with pytest.raises(ValueError, match="n_candidates_per_class"):
        build_vocabulary_from_descriptors(
            descriptors, cfg, __import__("logging").getLogger("test"), tmp_path / "final_vocab.pkl", k=5
        )


def test_l2_normalize_unit_norm_and_zero_safe():
    x = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]])
    out = l2_normalize(x)
    assert np.linalg.norm(out[0]) == pytest.approx(1.0)
    assert out[1].tolist() == [0.0, 0.0]  # zero row left as-is, no NaN
    assert out[2].tolist() == [1.0, 0.0]


def test_build_vocabulary_from_descriptors_cosine_metric(cfg, tmp_path):
    """metric="cosine" (CVWS pipeline step 9): two rays of points at very
    different magnitudes but the same two directions should still cluster
    by DIRECTION, not by norm -- something plain Euclidean K-means on the
    raw (unnormalized) vectors would get wrong."""
    rng = np.random.RandomState(2)
    # Direction A: near [1, 0], magnitudes from 1 to 50 (huge spread).
    dir_a = np.array([1.0, 0.0])
    mags_a = rng.uniform(1.0, 50.0, size=30)
    angles_a = rng.normal(0.0, 0.02, size=30)
    pts_a = np.stack([mags_a * np.cos(angles_a), mags_a * np.sin(angles_a)], axis=1)
    # Direction B: near [0, 1], same wide magnitude spread.
    mags_b = rng.uniform(1.0, 50.0, size=30)
    angles_b = np.pi / 2 + rng.normal(0.0, 0.02, size=30)
    pts_b = np.stack([mags_b * np.cos(angles_b), mags_b * np.sin(angles_b)], axis=1)
    descriptors = np.concatenate([pts_a, pts_b], axis=0).astype(np.float32)

    vocab = build_vocabulary_from_descriptors(
        descriptors,
        cfg,
        __import__("logging").getLogger("test"),
        tmp_path / "cosine_vocab.pkl",
        k=2,
        metric="cosine",
    )
    assert vocab.metric == "cosine"
    # Every point from direction A should be assigned to the same word,
    # and likewise for direction B, regardless of their wildly different
    # norms -- a direct check that assignment is norm-invariant.
    assign_a = vocab.assign(pts_a.astype(np.float32))
    assign_b = vocab.assign(pts_b.astype(np.float32))
    assert len(set(assign_a.tolist())) == 1
    assert len(set(assign_b.tolist())) == 1
    assert assign_a[0] != assign_b[0]


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