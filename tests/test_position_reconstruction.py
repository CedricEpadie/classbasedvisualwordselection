"""Tests for `src/position_reconstruction.py`, using a small hand-built
`ClusteringResult` (as `test_cvws_clustering.py` does for bovw_cvws's step
8) so the pure numpy/Python logic is verified without needing a real
UMAP/HDBSCAN/CNN/ViT run.

Scenario: 2 classes (cat, dog), 2 training images per class, 2 positions
(0 and 1), descriptor dim 2. dog_2's position-1 descriptor was dropped as
HDBSCAN noise (absent from `clustering_result` entirely), to exercise the
"no selected word at this position" path.
"""
import logging

import numpy as np
import pytest

from src.cvws_clustering import ClusteringResult
from src.position_reconstruction import (
    build_position_representatives,
    build_reconstructed_images,
    compute_positional_stats,
    group_rows_by_position,
    rank_position_words,
    select_all_position_words,
)
from src.selection_strategies import GlobalClassFrequency

LOGGER = logging.getLogger("test")


@pytest.fixture
def clustering_result() -> ClusteringResult:
    # row: image     position  cluster  probability
    #  0   cat_1        0         0        0.9
    #  1   cat_2        0         0        0.5
    #  2   dog_1        0         1        0.8
    #  3   dog_2        0         1        0.3
    #  4   cat_1        1         2        0.7
    #  5   cat_2        1         3        0.6
    #  6   dog_1        1         2        0.4
    #      dog_2        1        (dropped as HDBSCAN noise -- absent)
    return ClusteringResult(
        n_clusters=4,
        image_ids=["cat_1", "cat_2", "dog_1", "dog_2", "cat_1", "cat_2", "dog_1"],
        local_indices=np.array([0, 0, 0, 0, 1, 1, 1], dtype=np.int32),
        cluster_labels=np.array([0, 0, 1, 1, 2, 3, 2], dtype=np.int32),
        probabilities=np.array([0.9, 0.5, 0.8, 0.3, 0.7, 0.6, 0.4], dtype=np.float32),
    )


@pytest.fixture
def labels() -> dict:
    return {"cat_1": "cat", "cat_2": "cat", "dog_1": "dog", "dog_2": "dog"}


@pytest.fixture
def descriptor_paths(tmp_path) -> dict:
    vectors = {
        "cat_1": [[1, 1], [2, 2]],
        "cat_2": [[3, 3], [4, 4]],
        "dog_1": [[5, 5], [6, 6]],
        "dog_2": [[7, 7], [8, 8]],  # row 1 (position 1) never actually selected -- was noise
    }
    paths = {}
    for image_id, rows in vectors.items():
        p = tmp_path / f"{image_id}.npy"
        np.save(p, np.array(rows, dtype=np.float32))
        paths[image_id] = str(p)
    return paths


def test_group_rows_by_position(clustering_result):
    bags = group_rows_by_position(clustering_result)
    assert sorted(bags[0].tolist()) == [0, 1, 2, 3]
    assert sorted(bags[1].tolist()) == [4, 5, 6]


def test_compute_positional_stats_position_0(clustering_result, labels):
    rows = group_rows_by_position(clustering_result)[0]
    stats = compute_positional_stats(clustering_result, labels, 0, rows)
    assert stats.classes == ["cat", "dog"]
    assert stats.F["cat"].tolist() == [2.0, 0.0, 0.0, 0.0]
    assert stats.F["dog"].tolist() == [0.0, 2.0, 0.0, 0.0]
    assert stats.n_images_per_class == {"cat": 2, "dog": 2}


def test_compute_positional_stats_position_1_handles_dropped_descriptor(clustering_result, labels):
    rows = group_rows_by_position(clustering_result)[1]
    stats = compute_positional_stats(clustering_result, labels, 1, rows)
    # dog_2 has no surviving descriptor at position 1 -> all-zero row, still counted.
    assert stats.F["dog"].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert stats.n_images_per_class["dog"] == 2
    assert stats.per_class_matrix["dog"].shape == (2, 4)


def test_rank_position_words_excludes_zero_score(clustering_result, labels):
    rows0 = group_rows_by_position(clustering_result)[0]
    stats0 = compute_positional_stats(clustering_result, labels, 0, rows0)
    strategy = GlobalClassFrequency()
    # top_n=2 requested, but only cluster 0 has nonzero score for cat at
    # position 0 -> only 1 id returned, not padded to 2.
    assert rank_position_words(strategy, stats0, "cat", top_n=2) == [0]
    assert rank_position_words(strategy, stats0, "dog", top_n=2) == [1]


def test_rank_position_words_orders_by_score_descending(clustering_result, labels):
    rows1 = group_rows_by_position(clustering_result)[1]
    stats1 = compute_positional_stats(clustering_result, labels, 1, rows1)
    strategy = GlobalClassFrequency()
    # cat has one descriptor in cluster 2 (cat_1) and one in cluster 3
    # (cat_2) at position 1 -> tied score 1.0 each, ties broken by
    # ascending cluster id.
    assert rank_position_words(strategy, stats1, "cat", top_n=2) == [2, 3]


def test_select_all_position_words_end_to_end(clustering_result, labels):
    strategy = GlobalClassFrequency()
    result = select_all_position_words(strategy, clustering_result, labels, n_positions=2, top_n=2, logger=LOGGER)
    assert result[0] == {"cat": [0], "dog": [1]}
    assert result[1] == {"cat": [2, 3], "dog": [2]}


def test_build_position_representatives_picks_highest_probability_member(
    clustering_result, labels, descriptor_paths
):
    position_words = {0: {"cat": [0], "dog": [1]}, 1: {"cat": [2, 3], "dog": [2]}}
    reps = build_position_representatives(clustering_result, descriptor_paths, position_words)

    # (position 0, cluster 0): cat_1 (prob .9) beats cat_2 (prob .5).
    assert reps[(0, 0)].tolist() == [1.0, 1.0]
    # (position 0, cluster 1): dog_1 (prob .8) beats dog_2 (prob .3).
    assert reps[(0, 1)].tolist() == [5.0, 5.0]
    # (position 1, cluster 2): cat_1 (prob .7) beats dog_1 (prob .4).
    assert reps[(1, 2)].tolist() == [2.0, 2.0]
    # (position 1, cluster 3): only cat_2.
    assert reps[(1, 3)].tolist() == [4.0, 4.0]
    assert set(reps.keys()) == {(0, 0), (0, 1), (1, 2), (1, 3)}


def test_build_reconstructed_images_cycles_and_handles_missing_words(clustering_result, labels, descriptor_paths):
    position_words = {0: {"cat": [0], "dog": [1]}, 1: {"cat": [2, 3], "dog": [2]}}
    reps = build_position_representatives(clustering_result, descriptor_paths, position_words)

    images = build_reconstructed_images(
        position_words, reps, classes=["cat", "dog"], n_positions=2, descriptor_dim=2, images_per_class=3, logger=LOGGER
    )

    assert images["cat"].shape == (3, 2, 2)
    # position 0: only word 0 selected -> every synthetic cat image repeats it.
    assert images["cat"][:, 0, :].tolist() == [[1.0, 1.0]] * 3
    # position 1: words [2, 3] cycled over 3 images -> [2, 3, 2].
    assert images["cat"][0, 1, :].tolist() == [2.0, 2.0]
    assert images["cat"][1, 1, :].tolist() == [4.0, 4.0]
    assert images["cat"][2, 1, :].tolist() == [2.0, 2.0]

    assert images["dog"].shape == (3, 2, 2)
    assert images["dog"][:, 0, :].tolist() == [[5.0, 5.0]] * 3
    assert images["dog"][:, 1, :].tolist() == [[2.0, 2.0]] * 3


def test_build_reconstructed_images_all_zero_when_no_word_selected(clustering_result, labels, descriptor_paths):
    # A (position, class) pair with no selected word at all.
    position_words = {0: {"cat": [], "dog": [1]}, 1: {"cat": [2], "dog": [2]}}
    reps = build_position_representatives(clustering_result, descriptor_paths, position_words)

    images = build_reconstructed_images(
        position_words, reps, classes=["cat", "dog"], n_positions=2, descriptor_dim=2, images_per_class=2, logger=LOGGER
    )
    assert images["cat"][:, 0, :].tolist() == [[0.0, 0.0]] * 2
