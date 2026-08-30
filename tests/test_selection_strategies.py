"""Unit tests for the CVWS candidate-selection strategies from "Stratégie
de sélection des mots visuels" (C. Epadie, Aug. 2026): GCF (Méthode 1),
ICC (Méthode 2, entropy-weighted intra-class coverage), and CED
(Méthode 3, entropy-penalized class exclusivity).

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
        image (F_dog=6, H_intra(dog)=1) -- raw frequency favors cat (8>6),
        but coverage flips the ICC winner to dog.
    Candidate 2: present in both, spread uniformly in both (F_cat=4,
        F_dog=3, both H_intra=1) -- cat wins on raw magnitude alone.
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


def test_method1_global_class_frequency(stats):
    strat = GlobalClassFrequency()
    # F_cat=[8,8,4,0] vs F_dog=[0,6,3,9]: cat wins 0,1,2 ; dog wins 3.
    assert strat.select(stats, "cat") == {0, 1, 2}
    assert strat.select(stats, "dog") == {3}


def test_method2_intra_class_coverage_uses_entropy(stats):
    strat = IntraClassCorverage()
    scores = strat.score_matrix(stats)
    # S_ICC(v,C) = F(v,C) * H_intra(v,C).
    assert scores["cat"] == pytest.approx([8.0, 0.0, 4.0, 0.0])
    assert scores["dog"] == pytest.approx([0.0, 6.0, 3.0, 9.0])


def test_method2_flips_a_word_that_gcf_assigned_to_the_other_class(stats):
    """Candidate 1 has more raw occurrences in cat (8) than dog (6), so
    GCF assigns it to cat -- but it's confined to a single cat image
    (H_intra(cat)=0) versus uniformly spread across every dog image
    (H_intra(dog)=1), so ICC flips its winner to dog."""
    gcf, icc = GlobalClassFrequency(), IntraClassCorverage()
    assert gcf.assign(stats)[1] == stats.classes.index("cat")
    assert icc.assign(stats)[1] == stats.classes.index("dog")
    assert icc.select(stats, "cat") == {0}
    assert icc.select(stats, "dog") == {1, 2, 3}


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


def test_method3_never_flips_method2_ranking(stats):
    """D(v) = 1 - H_inter(v) is a single global scalar per candidate (not
    per class), so multiplying every class's ICC score by it can only
    rescale, never flip, which class wins a given candidate."""
    icc, ced = IntraClassCorverage(), ClassExclusivityDiscriminative()
    assert list(icc.assign(stats)) == list(ced.assign(stats))
    assert ced.select(stats, "cat") == {0, 2}
    assert ced.select(stats, "dog") == {1, 3}


def test_assignment_is_a_partition(stats):
    """Every candidate must be assigned to exactly one class: S(c1) and
    S(c2) are disjoint and their union is the full candidate set."""
    for strat in [
        GlobalClassFrequency(),
        IntraClassCorverage(),
        ClassExclusivityDiscriminative(),
    ]:
        s_cat = strat.select(stats, "cat")
        s_dog = strat.select(stats, "dog")
        assert s_cat.isdisjoint(s_dog)
        assert s_cat | s_dog == {0, 1, 2, 3}


def test_union_vocabulary_equals_full_candidate_set(stats):
    """Union across all classes of an UNCAPPED partition-style assignment
    is necessarily the whole candidate set."""
    strat = GlobalClassFrequency()
    union = select_union_vocabulary(strat, stats)
    assert union == {0, 1, 2, 3}


def test_top_n_for_vocabulary_formula():
    assert top_n_for_vocabulary(vocabulary_k=100, n_classes=5) == 20
    assert top_n_for_vocabulary(vocabulary_k=47, n_classes=10) == 4  # floor(47/10)
    # Always at least 1, even if K < M.
    assert top_n_for_vocabulary(vocabulary_k=2, n_classes=10) == 1


def test_select_top_n_caps_class_selection_by_score(stats):
    strat = GlobalClassFrequency()
    # S1(cat) = {0, 1, 2} uncapped, scores F_cat = [8, 8, 4, 0].
    full = strat.select(stats, "cat")
    assert full == {0, 1, 2}
    top2 = strat.select_top_n(stats, "cat", top_n=2)
    # Keep the 2 highest-scoring of {0,1,2} by F_cat: candidates 0 and 1
    # tie at 8 (kept in ascending-index order), candidate 2 (4) dropped.
    assert top2 == {0, 1}


def test_select_top_n_noop_when_cap_not_reached(stats):
    strat = GlobalClassFrequency()
    # Cap larger than the assigned set size -> behaves exactly like select().
    assert strat.select_top_n(stats, "dog", top_n=10) == strat.select(stats, "dog")


def test_top_n_union_can_be_smaller_than_full_candidate_set(stats):
    """The whole point of `n_candidates_per_class`: with n < |S(c)| for at
    least one class, the union across classes shrinks below the full
    candidate set -- this is what feeds the final K-means a reduced pool."""
    strat = GlobalClassFrequency()
    union = select_union_vocabulary(strat, stats, top_n=1)
    # cat's top-1 by score is candidate 0 (tie broken by ascending index
    # among {0,1} at score 8), dog's only assigned candidate is {3}.
    assert union == {0, 3}
    assert len(union) < stats.k


def test_build_strategy_factory():
    for name, cls in [
        ("global_class_frequency", GlobalClassFrequency),
        ("intra_class_corverage", IntraClassCorverage),
        ("class_exclusivity_discriminatve", ClassExclusivityDiscriminative),
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
