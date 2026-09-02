"""Per-class visual word selection strategies, implementing exactly the
three methods from "Stratégie de sélection des mots visuels" (C. Epadie,
Aug. 2026), used as step 7 of the CVWS pipeline (see `cvws_clustering.py`'s
module docstring and `pipeline.PipelineRunner._run_cvws_variants`):
clusters surviving UMAP+HDBSCAN (steps 3-5) are scored per class by one of
these three methods, and the `n_candidates_per_class` best-scoring ones
per class are kept.

Notation from the spec, mapped onto `vocabulary.VocabularyStats`:
    f(v, i)          occurrence count of word v in image i               -> stats.per_class_matrix[c][row, v]
    S_GCF(v, c)      sum_{i in c} f(v, i)                                 -> stats.F[c]
    N_c              number of images in class c                         -> stats.n_images_per_class[c]
    H_intra(v, c)    normalized Shannon entropy of v's distribution
                     across the images of c                              -> _entropy_intra(stats.per_class_matrix[c])
    H_inter(v)       normalized Shannon entropy of v's distribution
                     across classes (global, not per-class)              -> _entropy_inter(stats.F, stats.classes)
    D(v, c)          number of images of c containing word v (document
                     frequency) — NOT used by any of the three formulas
                     below (only kept in VocabularyStats/DF for
                     interpretability/audit)                             -> stats.DF[c]

Selection is INDEPENDENT per class: for each class C, `select_top_n`
ranks candidates strictly by C's own score(v, C) column and keeps C's own
top `n_candidates_per_class`, with NO comparison against any other
class's score (no argmax/exclusive-assignment step). A candidate can
therefore legitimately be selected by several classes at once — per the
spec's step 8: "Un même identifiant de cluster pouvant être sélectionné
par plusieurs classes [...] les ensembles de candidats retenus par classe
ne sont donc pas nécessairement disjoints." `cvws_clustering.py`'s step 8
(`select_clusters_and_build_candidates`) explicitly counts, per candidate,
how many distinct classes selected it (`selection_count`), and uses that
count to decide how many representative descriptors that candidate
contributes to the final K-means pool.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set

import numpy as np

from src.vocabulary import VocabularyStats


# --------------------------------------------------------------------------- #
# Shannon entropy helpers (shared by Méthode 2 and Méthode 3)
# --------------------------------------------------------------------------- #
def _entropy_intra(counts: np.ndarray) -> np.ndarray:
    """H_intra(v, C) for every word v, given `counts` = the (N_c, K) raw
    per-image count matrix for a single class C
    (`VocabularyStats.per_class_matrix[C]`):

        p(i|v,C) = f(v,i) / S_GCF(v,C)
        H_intra(v,C) = -1/log2(N_c) * sum_i p(i|v,C) * log2(p(i|v,C))

    Returns a (K,) array in [0, 1]: 1 if v is spread uniformly across
    every image of C, 0 if v occurs in only one image of C.

    Edge cases (convention, not in the spec's main formula): 0*log2(0) :=
    0; a word absent from C entirely (S_GCF(v,C)=0) gets H_intra=0; if
    N_c <= 1 there is no distribution to normalize by (log2(1)=0), so
    H_intra is defined as 0 for every word.
    """
    n_c = counts.shape[0]
    totals = counts.sum(axis=0)  # S_GCF(v, C), shape (K,)
    if n_c <= 1:
        return np.zeros_like(totals)
    safe_totals = np.where(totals > 0, totals, 1.0)
    p = counts / safe_totals  # (N_c, K), p(i|v,C)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(p > 0, np.log2(p), 0.0)
    raw_entropy = -(p * log_p).sum(axis=0)  # (K,)
    h = raw_entropy / np.log2(n_c)
    return np.where(totals > 0, h, 0.0)


def _entropy_inter(F: Dict[str, np.ndarray], classes: List[str]) -> np.ndarray:
    """H_inter(v) for every word v, given the per-class totals S_GCF(v,c)
    (`VocabularyStats.F`). This is GLOBAL across classes -- one (K,)
    vector, not per-class:

        p(c|v) = S_GCF(v,c) / sum_{c'} S_GCF(v,c')
        H_inter(v) = -1/log2(K) * sum_c p(c|v) * log2(p(c|v))
                     (K = number of classes, called M elsewhere in this codebase)

    Returns a (K_words,) array in [0, 1]: 1 if v is spread evenly across
    every class (generic, non-discriminant word), 0 if v is exclusive to
    one class. Same zero-handling convention as `_entropy_intra`.
    """
    n_classes = len(classes)
    totals_by_class = np.stack([F[c] for c in classes], axis=0)  # (M, K_words)
    class_sum = totals_by_class.sum(axis=0)  # (K_words,)
    if n_classes <= 1:
        return np.zeros_like(class_sum)
    safe_sum = np.where(class_sum > 0, class_sum, 1.0)
    p = totals_by_class / safe_sum  # (M, K_words), p(c|v)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(p > 0, np.log2(p), 0.0)
    raw_entropy = -(p * log_p).sum(axis=0)  # (K_words,)
    h = raw_entropy / np.log2(n_classes)
    return np.where(class_sum > 0, h, 0.0)


class VisualWordSelectionStrategy(ABC):
    """Common interface for the three selection methods (Strategy pattern).

    Subclasses only need to implement `score_matrix`, which returns, for
    every class, the (K,) score vector over the candidate set. `select_top_n`
    is shared and never needs to be reimplemented: it simply ranks
    `score_matrix(...)[class_label]` on its own (no comparison against
    other classes) and keeps the top N -- see the module docstring for why
    this is independent-per-class rather than an argmax/exclusive
    assignment.
    """

    name: str = "base"

    @abstractmethod
    def score_matrix(self, vocabulary_stats: VocabularyStats) -> Dict[str, np.ndarray]:
        """Return {class_label: score_vector} with score_vector shape (K,)."""
        raise NotImplementedError

    def select_top_n(
        self, vocabulary_stats: VocabularyStats, class_label: str, top_n: Optional[int]
    ) -> Set[int]:
        """The `top_n` candidates with the highest score(v, class_label)
        for `class_label` ALONE (ties broken by ascending index for
        determinism) -- no comparison against any other class's score, so
        the same candidate can end up in several classes' top-N sets at
        once (see the module docstring).

        `top_n=None` returns every candidate (class_label's full ranking,
        uncapped) -- in the CVWS pipeline `n_candidates_per_class` is
        always explicit, so this is mostly a convenience for tests/direct use.
        """
        score_vector = self.score_matrix(vocabulary_stats)[class_label]
        k = score_vector.shape[0]
        # argsort is stable: ties keep ascending-index order for determinism.
        order = np.argsort(-score_vector, kind="stable")
        if top_n is None or top_n >= k:
            return set(int(i) for i in order)
        return set(int(i) for i in order[:top_n])


class GlobalClassFrequency(VisualWordSelectionStrategy):
    """Méthode 1 — Global Class Frequency (GCF).

    S_GCF(v, c) = sum_{i in c} f(v, i)

    Measures the word's total abundance within the class, in absolute
    terms. Ranks each class's candidates by this alone.
    """

    name = "GCF"

    def score_matrix(self, vocabulary_stats: VocabularyStats) -> Dict[str, np.ndarray]:
        return {c: vocabulary_stats.F[c] for c in vocabulary_stats.classes}


class IntraClassCorverage(VisualWordSelectionStrategy):
    """Méthode 2 — Intra-Class Coverage (ICC).

    S_ICC(v, C) = S_GCF(v, C) * H_intra(v, C)

    Weights the raw frequency by how uniformly the word is spread across
    the class's own images (H_intra -> 1 for uniform spread, -> 0 for a
    word confined to a single image of the class): a word can have a high
    raw count yet score near zero here if that count comes almost
    entirely from one image.
    """

    name = "ICC"

    def score_matrix(self, vocabulary_stats: VocabularyStats) -> Dict[str, np.ndarray]:
        scores = {}
        for c in vocabulary_stats.classes:
            h_intra = _entropy_intra(vocabulary_stats.per_class_matrix[c])
            scores[c] = vocabulary_stats.F[c] * h_intra
        return scores


class ClassExclusivityDiscriminative(VisualWordSelectionStrategy):
    """Méthode 3 — Class Exclusivity Discriminative (CED).

    S_CED(v, C) = S_ICC(v, C) * D(v),    D(v) = 1 - H_inter(v)

    Penalizes words that are spread uniformly across every class in the
    dataset (H_inter -> 1, generic/non-discriminant word -> D(v) -> 0,
    fully discounting S_ICC regardless of class); a word exclusive to one
    class (H_inter -> 0) keeps its full S_ICC score (D(v) -> 1).

    Note D(v) varies per WORD (not per class): for a fixed class c,
    comparing two candidates v1 and v2, CED can reorder them relative to
    their ICC ranking whenever D(v1) != D(v2) — a candidate with high
    S_ICC but that's also spread evenly across every class (D(v) close to
    0) can rank below one with lower S_ICC but strong class exclusivity
    (D(v) close to 1). Unlike the old argmax-across-classes framing, this
    is exactly the point of Méthode 3: it is expected to change each
    class's own ranking, not merely rescale it.
    """

    name = "CED"

    def score_matrix(self, vocabulary_stats: VocabularyStats) -> Dict[str, np.ndarray]:
        icc_scores = IntraClassCorverage().score_matrix(vocabulary_stats)
        h_inter = _entropy_inter(vocabulary_stats.F, vocabulary_stats.classes)
        discriminability = 1.0 - h_inter  # (K,), global per word -- same for every class
        return {c: icc_scores[c] * discriminability for c in vocabulary_stats.classes}


# --------------------------------------------------------------------------- #
# Registry / factory
# --------------------------------------------------------------------------- #
_STRATEGIES = {
    "GCF": GlobalClassFrequency,
    "ICC": IntraClassCorverage,
    "CED": ClassExclusivityDiscriminative,
}


def build_strategy(strategy_name: str) -> VisualWordSelectionStrategy:
    cls = _STRATEGIES.get(strategy_name)
    if cls is None:
        raise ValueError(f"Unknown selection strategy: {strategy_name}. Available: {sorted(_STRATEGIES)}")
    return cls()


def top_n_for_vocabulary(vocabulary_k: int, n_classes: int) -> int:
    """top_n = taille_vocabulaire / nombre_de_classes (floor division, at
    least 1).

    NOTE: no longer called automatically anywhere in `pipeline.py`. It
    predates the CVWS pipeline, back when `top_n` capped the FINAL
    vocabulary's own words post-hoc; `n_candidates_per_class` (see
    `config.SelectionConfig`) now plays that role instead, operating on
    the (differently-sized) HDBSCAN cluster-id space, and is passed
    explicitly rather than auto-derived from K. Kept as a standalone,
    tested utility in case a K-derived heuristic is useful elsewhere.
    """
    return max(1, vocabulary_k // max(n_classes, 1))


# --------------------------------------------------------------------------- #
# Multi-class union (feeds the final K-means, step 9 of the CVWS pipeline)
# --------------------------------------------------------------------------- #
def select_union_vocabulary(
    strategy: VisualWordSelectionStrategy,
    vocabulary_stats: VocabularyStats,
    top_n: Optional[int] = None,
) -> Set[int]:
    """Union of every class's own `select_top_n` result.

    Since selection is independent per class (see the module docstring),
    this union routinely contains candidates selected by more than one
    class — that overlap is expected, not an edge case, and is exactly
    what `cvws_clustering.select_clusters_and_build_candidates` (step 8)
    counts via `selection_count` to decide how many representative
    descriptors each selected candidate contributes to the final K-means
    pool (step 9, `vocabulary.build_vocabulary_from_descriptors`).
    """
    union: Set[int] = set()
    for c in vocabulary_stats.classes:
        union |= strategy.select_top_n(vocabulary_stats, c, top_n)
    return union


def apply_selection_to_histogram(
    histogram: np.ndarray, selected_indices: Set[int], zero_fill: bool = True
) -> np.ndarray:
    """Reconstruct a histogram restricted to the selected visual words.

    If `zero_fill` is True, keep the original vector length K and zero out
    unselected words (so all approaches can share the same downstream
    classifier code with a fixed-size input). Otherwise, compact the vector
    to only the selected dimensions, sorted by index.

    NOTE: no longer called by `pipeline.py`. It predates the CVWS
    pipeline, back when selection was a post-hoc masking of an
    already-built final vocabulary's histogram; the CVWS pipeline now
    selects candidates *before* the final K-means (steps 7-8, ahead of
    step 9) instead, so every final histogram already has the reduced
    dimensionality baked in and needs no further masking. Kept as a
    standalone, tested utility for other post-hoc masking use cases.
    """
    if zero_fill:
        mask = np.zeros_like(histogram)
        idx = sorted(selected_indices)
        mask[idx] = 1
        return histogram * mask
    idx = sorted(selected_indices)
    return histogram[idx]
