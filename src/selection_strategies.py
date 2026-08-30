"""Per-class visual word selection strategies, implementing exactly the
three methods from "Stratégie de sélection des mots visuels" (C. Epadie,
Aug. 2026), used as step 3 of the CVWS candidate pipeline (see
`vocabulary.py`'s module docstring and
`pipeline.PipelineRunner._run_cvws_variants`): candidates surviving Mean
Shift (step 2) are scored per class by one of these three methods, and the
`n_candidates_per_class` best-scoring ones per class are kept (union
across classes feeds the final K-means, step 4).

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

All three methods work the same way: compute a per-(word, class) score,
then assign each visual word v to the single class c*(v) that maximizes
its score:
    c*(v) = argmax_{c in C} score(v, c)

The per-class selected set S(c) is then every word assigned to c, capped
to the `n_candidates_per_class` highest-scoring ones (see `select_top_n`):
without a cap this is a full *partition* of the candidate set (union of
S(c) over every class = all candidates); capping to an explicit n < |S(c)|
is what actually reduces the pool of descriptors handed to the final
K-means (step 4) below the full candidate count.
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
    every class, the (K,) score vector over the candidate set used to
    decide that class's ownership of each candidate. `select` and `assign`
    are shared (argmax logic) and never need to be reimplemented.
    """

    name: str = "base"

    @abstractmethod
    def score_matrix(self, vocabulary_stats: VocabularyStats) -> Dict[str, np.ndarray]:
        """Return {class_label: score_vector} with score_vector shape (K,)."""
        raise NotImplementedError

    def assign(self, vocabulary_stats: VocabularyStats) -> np.ndarray:
        """(K,) int array: for each visual word, the index (into
        `vocabulary_stats.classes`) of its winning class c*(v)."""
        scores = self.score_matrix(vocabulary_stats)
        # Stack in the same, fixed class order for a deterministic argmax.
        matrix = np.stack([scores[c] for c in vocabulary_stats.classes], axis=0)  # (M, K)
        return np.argmax(matrix, axis=0)  # (K,)

    def select(self, vocabulary_stats: VocabularyStats, class_label: str) -> Set[int]:
        """S(c) = {v : c*(v) = c} — every visual word assigned to `class_label`."""
        class_index = vocabulary_stats.classes.index(class_label)
        assignment = self.assign(vocabulary_stats)
        return set(int(i) for i in np.where(assignment == class_index)[0])

    def select_top_n(
        self, vocabulary_stats: VocabularyStats, class_label: str, top_n: Optional[int]
    ) -> Set[int]:
        """Same as `select`, but capped to the `top_n` highest-scoring words
        within the class's own assigned set S(class_label) (ties broken by
        the original word index for determinism).

        This is what actually reduces the pool of candidates handed to the
        final K-means (step 4): `select` alone returns a *partition* of
        the candidate set, so the union of `select(c)` over every class is
        always the full candidate set. Capping each class's slice to
        `top_n` (= `selection.n_candidates_per_class`) words means the
        summed (disjoint) selection sizes, and therefore the union, become
        <= top_n * M <= K in general.

        `top_n=None` disables the cap (falls back to the full assignment,
        i.e. behaves exactly like `select`).
        """
        assigned = self.select(vocabulary_stats, class_label)
        if top_n is None or len(assigned) <= top_n:
            return assigned
        scores = self.score_matrix(vocabulary_stats)[class_label]
        assigned_sorted = np.array(sorted(assigned))
        assigned_scores = scores[assigned_sorted]
        # argsort is stable: ties keep ascending-index order for determinism.
        order = np.argsort(-assigned_scores, kind="stable")
        top = assigned_sorted[order][:top_n]
        return set(int(i) for i in top)


class GlobalClassFrequency(VisualWordSelectionStrategy):
    """Méthode 1 — Global Class Frequency (GCF).

    S_GCF(v, c) = sum_{i in c} f(v, i)

    Measures the word's total abundance within the class, in absolute
    terms. Assigns each word to the class where it occurs most often.
    """

    name = "global_class_frequency"

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

    name = "intra_class_corverage"

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

    Note D(v) is a single, class-INVARIANT scalar per word (it does not
    depend on C): multiplying every class's S_ICC(v, ·) by the same
    nonnegative constant never changes which class wins a given word
    (unless D(v)=0, in which case every class ties at score 0) — CED can
    only rescale S_ICC's per-word class ranking, never flip it.
    """

    name = "class_exclusivity_discriminatve"

    def score_matrix(self, vocabulary_stats: VocabularyStats) -> Dict[str, np.ndarray]:
        icc_scores = IntraClassCorverage().score_matrix(vocabulary_stats)
        h_inter = _entropy_inter(vocabulary_stats.F, vocabulary_stats.classes)
        discriminability = 1.0 - h_inter  # (K,), global per word -- same for every class
        return {c: icc_scores[c] * discriminability for c in vocabulary_stats.classes}


# --------------------------------------------------------------------------- #
# Registry / factory
# --------------------------------------------------------------------------- #
_STRATEGIES = {
    "global_class_frequency": GlobalClassFrequency,
    "intra_class_corverage": IntraClassCorverage,
    "class_exclusivity_discriminatve": ClassExclusivityDiscriminative,
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
    predates the CVWS candidate pipeline, back when `top_n` capped the
    FINAL vocabulary's own words post-hoc; `n_candidates_per_class` (see
    `config.SelectionConfig`) now plays that role instead, operating on
    the (differently-sized) Mean-Shift candidate set, and is passed
    explicitly rather than auto-derived from K. Kept as a standalone,
    tested utility in case a K-derived heuristic is useful elsewhere.
    """
    return max(1, vocabulary_k // max(n_classes, 1))


# --------------------------------------------------------------------------- #
# Multi-class union (feeds the final K-means, step 4 of the CVWS pipeline)
# --------------------------------------------------------------------------- #
def select_union_vocabulary(
    strategy: VisualWordSelectionStrategy,
    vocabulary_stats: VocabularyStats,
    top_n: Optional[int] = None,
) -> Set[int]:
    """Union of S(c) (or its top-N-capped version) across all classes.

    Without a cap (`top_n=None`), the three methods define a *partition* of
    the candidate set (every candidate assigned to exactly one class), so
    this union is mathematically the full candidate set. Passing `top_n`
    (`selection.n_candidates_per_class` in the CVWS pipeline) restricts
    each class to its `top_n` highest-scoring assigned candidates, which
    is what actually shrinks the union handed to the final K-means (step
    4, `vocabulary.build_vocabulary_from_descriptors`) below the full
    candidate count.
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

    NOTE: no longer called by `pipeline.py`. It predates the CVWS candidate
    pipeline, back when selection was a post-hoc masking of an
    already-built final vocabulary's histogram; the CVWS pipeline now
    selects candidates *before* the final K-means (step 3, ahead of step
    4) instead, so every final histogram already has the reduced
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
