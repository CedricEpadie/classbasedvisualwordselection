"""Tests for `src/cvws_clustering.py`'s steps 6-8: symbolic re-encoding +
per-class stats (`compute_cluster_stats`) and per-class cluster selection
+ weighted candidate reconstruction (`select_clusters_and_build_candidates`).

Steps 3-5 (`reduce_and_cluster`, the actual UMAP/HDBSCAN fit) are NOT unit
tested here: they require the `umap-learn`/scikit-learn HDBSCAN
dependencies and are best exercised with a real end-to-end run. Instead,
these tests hand-build a small `ClusteringResult` (as if steps 3-5 had
already run) so steps 6-8's own logic -- which is pure numpy/Python, no
external ML dependency -- can be verified precisely and quickly.

Since selection is independent per class (see `selection_strategies.py`'s
module docstring), the same cluster id can legitimately be selected by
several classes at once — these tests exercise that overlap directly,
rather than working around an (obsolete) argmax/partition assumption.
"""
import logging

import numpy as np
import pytest

from src.cvws_clustering import ClusteringResult, compute_cluster_stats, select_clusters_and_build_candidates
from src.selection_strategies import GlobalClassFrequency

LOGGER = logging.getLogger("test")


@pytest.fixture
def clustering_result() -> ClusteringResult:
    """3 clusters, 2 training images ("cat_1", "dog_1"), 5 surviving
    (non-noise) descriptors total:

        row  image   local_idx  cluster  probability
        0    cat_1    0          0        0.9
        1    cat_1    1          0        0.4   <- same cluster as row 0, lower probability
        2    cat_1    2          1        0.7
        3    dog_1    0          2        0.6
        4    dog_1    1          2        0.95  <- same cluster as row 3, higher probability

    So cluster 0 has 2 members (both from cat_1), cluster 1 has 1 member
    (cat_1), cluster 2 has 2 members (both from dog_1).

    Resulting per-class cluster-occurrence counts (see
    `test_compute_cluster_stats_counts_occurrences_per_image_and_class`):
    F_cat = [2, 1, 0], F_dog = [0, 0, 2].
    """
    return ClusteringResult(
        n_clusters=3,
        image_ids=["cat_1", "cat_1", "cat_1", "dog_1", "dog_1"],
        local_indices=np.array([0, 1, 2, 0, 1], dtype=np.int32),
        cluster_labels=np.array([0, 0, 1, 2, 2], dtype=np.int32),
        probabilities=np.array([0.9, 0.4, 0.7, 0.6, 0.95], dtype=np.float32),
    )


@pytest.fixture
def labels() -> dict:
    return {"cat_1": "cat", "dog_1": "dog"}


@pytest.fixture
def descriptor_paths(tmp_path) -> dict:
    """cat_1 has 3 descriptors (rows [0,1,2,3], [4,5,6,7], [8,9,10,11]);
    dog_1 has 2 ([100..103], [104..107]) -- distinct, recognizable values
    so gathered candidates can be matched back to a specific row exactly."""
    paths = {}
    cat_arr = np.arange(3 * 4).reshape(3, 4).astype(np.float32)
    dog_arr = (np.arange(2 * 4).reshape(2, 4) + 100).astype(np.float32)
    for image_id, arr in [("cat_1", cat_arr), ("dog_1", dog_arr)]:
        p = tmp_path / f"{image_id}.npy"
        np.save(p, arr)
        paths[image_id] = str(p)
    return paths


def test_compute_cluster_stats_counts_occurrences_per_image_and_class(clustering_result, labels):
    stats = compute_cluster_stats(clustering_result, labels, LOGGER)

    assert stats.k == 3
    assert stats.classes == ["cat", "dog"]
    # cat_1 contributes 2 descriptors to cluster 0, 1 to cluster 1, 0 to cluster 2.
    assert stats.F["cat"].tolist() == [2.0, 1.0, 0.0]
    # dog_1 contributes 0 to clusters 0/1, 2 to cluster 2.
    assert stats.F["dog"].tolist() == [0.0, 0.0, 2.0]
    assert stats.n_images_per_class["cat"] == 1
    assert stats.n_images_per_class["dog"] == 1
    # Only 1 image per class here, so per_class_matrix has exactly 1 row each.
    assert stats.per_class_matrix["cat"].tolist() == [[2.0, 1.0, 0.0]]
    assert stats.per_class_matrix["dog"].tolist() == [[0.0, 0.0, 2.0]]


def test_select_clusters_top1_no_overlap(clustering_result, labels, descriptor_paths):
    """n_candidates_per_class=1: cat's top-1 cluster is 0 (F=2, its
    highest); dog's top-1 is 2 (F=2, its only nonzero cluster). No overlap
    -> each selected cluster's selection_count is 1, and only its single
    highest-probability member is kept."""
    stats = compute_cluster_stats(clustering_result, labels, LOGGER)
    strategy = GlobalClassFrequency()

    candidates, audit = select_clusters_and_build_candidates(
        clustering_result, stats, strategy, n_candidates_per_class=1, descriptor_paths=descriptor_paths, logger=LOGGER
    )

    assert audit["per_class_selected_clusters"] == {"cat": [0], "dog": [2]}
    assert audit["per_cluster_selection"]["0"] == {"selection_count": 1, "cluster_size": 2, "n_taken": 1}
    assert audit["per_cluster_selection"]["2"] == {"selection_count": 1, "cluster_size": 2, "n_taken": 1}
    assert audit["n_candidate_descriptors"] == 2

    cat_1 = np.load(descriptor_paths["cat_1"])
    dog_1 = np.load(descriptor_paths["dog_1"])
    candidate_set = {tuple(row) for row in candidates}
    # Cluster 0's members are cat_1 rows 0 (prob 0.9) and 1 (prob 0.4) ->
    # only the higher-probability one (row 0) is kept.
    assert tuple(cat_1[0]) in candidate_set
    assert tuple(cat_1[1]) not in candidate_set
    # Cluster 2's members are dog_1 rows 0 (prob 0.6) and 1 (prob 0.95) ->
    # only row 1 is kept.
    assert tuple(dog_1[1]) in candidate_set
    assert tuple(dog_1[0]) not in candidate_set


def test_select_clusters_top2_produces_genuine_overlap(clustering_result, labels, descriptor_paths):
    """n_candidates_per_class=2: cat's top-2 is {0, 1} (F=[2,1,0]); dog's
    top-2 is {2, 0} (F=[0,0,2], ties on 0/1 broken by ascending index).
    Cluster 0 is therefore selected by BOTH classes -- exactly the "un
    même identifiant de cluster [...] sélectionné par plusieurs classes"
    premise of step 8 -- so its selection_count is 2, and BOTH of its
    members are kept (not just the highest-probability one)."""
    stats = compute_cluster_stats(clustering_result, labels, LOGGER)
    strategy = GlobalClassFrequency()

    candidates, audit = select_clusters_and_build_candidates(
        clustering_result, stats, strategy, n_candidates_per_class=2, descriptor_paths=descriptor_paths, logger=LOGGER
    )

    assert audit["per_class_selected_clusters"] == {"cat": [0, 1], "dog": [0, 2]}
    assert audit["per_cluster_selection"]["0"] == {"selection_count": 2, "cluster_size": 2, "n_taken": 2}
    assert audit["per_cluster_selection"]["1"] == {"selection_count": 1, "cluster_size": 1, "n_taken": 1}
    assert audit["per_cluster_selection"]["2"] == {"selection_count": 1, "cluster_size": 2, "n_taken": 1}
    # 2 (cluster 0, both members) + 1 (cluster 1) + 1 (cluster 2) = 4.
    assert audit["n_candidate_descriptors"] == 4

    cat_1 = np.load(descriptor_paths["cat_1"])
    dog_1 = np.load(descriptor_paths["dog_1"])
    candidate_set = {tuple(row) for row in candidates}
    # Cluster 0's selection_count (2) equals its own cluster_size (2), so
    # BOTH members are kept this time (contrast with the top-1 case above).
    assert tuple(cat_1[0]) in candidate_set
    assert tuple(cat_1[1]) in candidate_set
    assert tuple(cat_1[2]) in candidate_set  # cluster 1's only member
    assert tuple(dog_1[1]) in candidate_set  # cluster 2, still only its higher-probability member


def test_select_clusters_clamps_when_selection_count_exceeds_cluster_size(
    clustering_result, labels, descriptor_paths
):
    """Confirmed edge-case handling: if a cluster's selection_count ever
    exceeded its own member count, every member would be kept (clamped),
    not an error. Forced here by giving BOTH classes an inflated,
    identical score on cluster 1 (which only has 1 member) via a bespoke
    stats object, so both "select" it independently -- selection_count=2
    on a cluster of size 1."""
    stats = compute_cluster_stats(clustering_result, labels, LOGGER)
    stats.F["cat"] = np.array([0.0, 5.0, 0.0])
    stats.F["dog"] = np.array([0.0, 5.0, 0.0])
    strategy = GlobalClassFrequency()

    candidates, audit = select_clusters_and_build_candidates(
        clustering_result, stats, strategy, n_candidates_per_class=1, descriptor_paths=descriptor_paths, logger=LOGGER
    )

    assert audit["per_class_selected_clusters"] == {"cat": [1], "dog": [1]}
    info = audit["per_cluster_selection"]["1"]
    assert info["selection_count"] == 2  # both classes picked it
    assert info["cluster_size"] == 1  # but it only has 1 member
    assert info["n_taken"] == 1  # clamped, not an error
    assert audit["n_candidate_descriptors"] == 1
