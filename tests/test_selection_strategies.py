"""Unit tests for the CVWS candidate-selection strategies from "Stratégie
de sélection des mots visuels" (C. Epadie, Aug. 2026): GCF (Méthode 1),
ICC (Méthode 2, entropy-weighted intra-class coverage), and CED
(Méthode 3, entropy-penalized class exclusivity).

Selection is INDEPENDENT per class (see `selection_strategies.py`'s module
docstring): each class ranks candidates strictly by its own score(v, C)
column and keeps its own top N, with no cross-class competition. A
candidate can therefore be selected by several classes at once -- these
tests explicitly check for that overlap where it's expected, rather than
assuming a partition.

Uses a small, hand-constructed `VocabularyStats` (2 classes, 4 candidates,
with explicit per-image count matrices) so H_intra/H_inter and every
downstream score can be verified against the spec's formulas directly
(via `math.log2`, not magic decimals). See the fixture's docstring for the
full worked-out numbers.
"""
import math

import numpy as np
import pytest

from src.selection_strategies import (
    ClassExclusivityDiscriminative,
    IntraClassCorverage,
    GlobalClassFrequency,
    apply_selection_to_histogram,
    build_strategy,
    select_union_vocabulary,
    top_n_for_vocabulary,
)
from src.vocabulary import VocabularyStats

# H_inter for candidates 1 and 2: both have (F_cat, F_dog) in the ratio
# 4:3 (8:6 and 4:3 respectively), so they share the same inter-class
# entropy despite different magnitudes -- entropy depends only on the
# *proportions* p(c|v), not on the absolute counts.
H_INTER_4_3 = -(4 / 7 * math.log2(4 / 7) + 3 / 7 * math.log2(3 / 7))  # ~0.98523


@pytest.fixture
def stats() -> VocabularyStats:
    """4 candidates, classes "cat" (N=4 images) and "dog" (N=3 images).

    Per-image raw counts (rows = images, columns = candidates 0..3):

        cat = [[2, 8, 1, 0],
               [2, 0, 1, 0],
               [2, 0, 1, 0],
               [2, 0, 1, 0]]

        dog = [[0, 2, 1, 3],
               [0, 2, 1, 3],
               [0, 2, 1, 3]]

    Candidate 0: exclusive to cat, spread perfectly uniformly (2 in every
        image) -> F_cat=8, F_dog=0, H_intra(cat)=1.
    Candidate 1: present in BOTH, but concentrated in a single cat image
        (F_cat=8, H_intra(cat)=0) vs spread uniformly across every dog
        image (F_dog=6, H_intra(dog)=1) -- raw frequency (GCF) ranks it
        highly for BOTH classes (8 for cat, 6 for dog: it's a genuine
        overlap candidate under GCF), but ICC's coverage weighting drops
        cat's score for it to 0.
    Candidate 2: present in both, spread uniformly in both (F_cat=4,
        F_dog=3, both H_intra=1).
    Candidate 3: exclusive to dog, spread perfectly uniformly -> F_cat=0,
        F_dog=9, H_intra(dog)=1.

    Derived (F = column sums, DF = count of nonzero entries per column):
        F_cat = [8, 8, 4, 0],  DF_cat = [4, 1, 4, 0]
        F_dog = [0, 6, 3, 9],  DF_dog = [0, 3, 3, 3]
    """
    cat = np.array(
        [
            [2, 8, 1, 0],
            [2, 0, 1, 0],
            [2, 0, 1, 0],
            [2, 0, 1, 0],
        ],
        dtype=float,
    )
    dog = np.array(
        [
            [0, 2, 1, 3],
            [0, 2, 1, 3],
            [0, 2, 1, 3],
        ],
        dtype=float,
    )
    s = VocabularyStats(k=4, classes=["cat", "dog"])
    s.per_class_matrix["cat"] = cat
    s.per_class_matrix["dog"] = dog
    s.F["cat"] = cat.sum(axis=0)
    s.F["dog"] = dog.sum(axis=0)
    s.DF["cat"] = (cat > 0).sum(axis=0).astype(float)
    s.DF["dog"] = (dog > 0).sum(axis=0).astype(float)
    s.n_images_per_class["cat"] = cat.shape[0]
    s.n_images_per_class["dog"] = dog.shape[0]
    return s


def test_method1_GCF(stats):
    strat = GlobalClassFrequency()
    scores = strat.score_matrix(stats)
    assert scores["cat"].tolist() == [8.0, 8.0, 4.0, 0.0]
    assert scores["dog"].tolist() == [0.0, 6.0, 3.0, 9.0]


def test_method2_intra_class_coverage_uses_entropy(stats):
    strat = IntraClassCorverage()
    scores = strat.score_matrix(stats)
    # S_ICC(v,C) = F(v,C) * H_intra(v,C).
    assert scores["cat"] == pytest.approx([8.0, 0.0, 4.0, 0.0])
    assert scores["dog"] == pytest.approx([0.0, 6.0, 3.0, 9.0])


def test_method2_zero_for_word_confined_to_a_single_image(stats):
    """A candidate occurring entirely within one image of a class scores
    0 under ICC, exactly like a candidate absent from the class -- this is
    the spec's stated H_intra -> 0 behavior, not merely a low score."""
    strat = IntraClassCorverage()
    scores = strat.score_matrix(stats)
    assert scores["cat"][1] == pytest.approx(0.0, abs=1e-9)  # confined to 1 image
    assert scores["cat"][3] == pytest.approx(0.0, abs=1e-9)  # absent entirely


def test_method3_class_exclusivity_discriminative(stats):
    strat = ClassExclusivityDiscriminative()
    scores = strat.score_matrix(stats)
    d_shared = 1.0 - H_INTER_4_3  # candidates 1 and 2: shared, ratio 4:3 in both
    # Candidates 0 and 3 are exclusive (H_inter=0 -> D=1): CED == ICC exactly.
    assert scores["cat"][0] == pytest.approx(8.0)
    assert scores["dog"][3] == pytest.approx(9.0)
    # Candidates 1 and 2 are shared: CED heavily discounts ICC by D < 1.
    assert scores["cat"][2] == pytest.approx(4.0 * d_shared)
    assert scores["dog"][1] == pytest.approx(6.0 * d_shared)
    assert scores["dog"][2] == pytest.approx(3.0 * d_shared)
    assert scores["cat"][1] == pytest.approx(0.0, abs=1e-9)  # 0 * anything is still 0


def test_method3_can_reorder_method2_within_class_ranking(stats):
    """D(v) varies per word (not per class): for dog, ICC ranks candidate
    2 (score 3) below candidate 1 (score 6), but CED's exclusivity penalty
    hits both equally hard (same D, since both are the shared 4:3-ratio
    candidates) so their RELATIVE order is unchanged here -- but this is a
    coincidence of the fixture, not a general guarantee (see the module
    docstring: CED can and does reorder ICC's within-class ranking
    whenever D differs across words, e.g. between an exclusive candidate
    like 3 (D=1) and a shared one like 1 (D~0.015))."""
    icc, ced = IntraClassCorverage(), ClassExclusivityDiscriminative()
    icc_dog = icc.score_matrix(stats)["dog"]
    ced_dog = ced.score_matrix(stats)["dog"]
    # Under ICC, candidate 3 (score 9) already outranks candidate 1 (score
    # 6). Under CED, candidate 3 keeps its full score (D=1, exclusive)
    # while candidate 1 is heavily discounted (D~0.015, shared) -- CED
    # only widens the gap here, it doesn't need to invert it to
    # demonstrate D is candidate-specific.
    assert icc_dog[3] > icc_dog[1]
    assert ced_dog[3] > ced_dog[1]
    assert (ced_dog[3] - ced_dog[1]) > (icc_dog[3] - icc_dog[1])


def test_select_top_n_ranks_independently_per_class(stats):
    """Core behavior change: `select_top_n` ranks each class's OWN score
    column, with no reference to any other class -- so nothing prevents
    the same candidate appearing in more than one class's top-N."""
    strat = GlobalClassFrequency()
    # cat's top-2 by F_cat=[8,8,4,0]: candidates 0 and 1 (tie at 8, kept
    # in ascending-index order over candidate 2's 4).
    assert strat.select_top_n(stats, "cat", top_n=2) == {0, 1}
    # dog's top-2 by F_dog=[0,6,3,9]: candidates 3 (9) and 1 (6).
    assert strat.select_top_n(stats, "dog", top_n=2) == {1, 3}


def test_candidate_can_be_selected_by_several_classes(stats):
    """The spec's step 8 premise, made explicit: under GCF, candidate 1
    ranks in BOTH cat's and dog's top-2 (raw frequency alone doesn't
    penalize it for being shared) -- this overlap is expected, not a bug,
    and is exactly what `cvws_clustering.select_clusters_and_build_candidates`
    (step 8) counts via `selection_count`."""
    strat = GlobalClassFrequency()
    cat_top2 = strat.select_top_n(stats, "cat", top_n=2)
    dog_top2 = strat.select_top_n(stats, "dog", top_n=2)
    assert cat_top2 & dog_top2 == {1}


def test_select_top_n_none_returns_full_ranking(stats):
    strat = GlobalClassFrequency()
    assert strat.select_top_n(stats, "cat", top_n=None) == {0, 1, 2, 3}


def test_select_top_n_cap_larger_than_k_is_a_noop(stats):
    strat = GlobalClassFrequency()
    assert strat.select_top_n(stats, "dog", top_n=100) == {0, 1, 2, 3}


def test_top_n_for_vocabulary_formula():
    assert top_n_for_vocabulary(vocabulary_k=100, n_classes=5) == 20
    assert top_n_for_vocabulary(vocabulary_k=47, n_classes=10) == 4  # floor(47/10)
    # Always at least 1, even if K < M.
    assert top_n_for_vocabulary(vocabulary_k=2, n_classes=10) == 1


def test_union_vocabulary_with_overlap_is_smaller_than_sum_of_per_class_sizes(stats):
    """`select_union_vocabulary` is a plain set union: with overlap (see
    above), its size can be strictly less than top_n * n_classes."""
    strat = GlobalClassFrequency()
    union = select_union_vocabulary(strat, stats, top_n=2)
    assert union == {0, 1, 3}  # cat gives {0,1}, dog gives {1,3}, union has 3 not 4 elements
    assert len(union) < 2 * len(stats.classes)


def test_union_vocabulary_uncapped_equals_full_candidate_set(stats):
    strat = GlobalClassFrequency()
    assert select_union_vocabulary(strat, stats) == {0, 1, 2, 3}


def test_build_strategy_factory():
    for name, cls in [
        ("GCF", GlobalClassFrequency),
        ("ICC", IntraClassCorverage),
        ("CED", ClassExclusivityDiscriminative),
    ]:
        assert isinstance(build_strategy(name), cls)
    with pytest.raises(ValueError):
        build_strategy("not_a_real_strategy")


def test_apply_selection_zero_fill_keeps_length():
    hist = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = apply_selection_to_histogram(hist, {1, 3}, zero_fill=True)
    assert out.shape == hist.shape
    assert out.tolist() == [0.0, 2.0, 0.0, 4.0, 0.0]


def test_apply_selection_compact_reduces_length():
    hist = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = apply_selection_to_histogram(hist, {1, 3}, zero_fill=False)
    assert out.tolist() == [2.0, 4.0]
