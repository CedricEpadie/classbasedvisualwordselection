"""Proportional integer allocation, shared by:

* `cvws_clustering.select_clusters_and_build_weighted_candidates` (bovw_cvws's
  updated step 8: split N/|C| vectors per class across each class's selected
  clusters, proportional to their selection score).
* `position_reconstruction.py` (cnn_bovw_cvws/vit_cvws's per-position word
  allocation).

Both need the same operation: given a set of non-negative scores and a
target integer total, split the total across the scores' buckets so that
(a) each bucket gets a share proportional to its score and (b) the shares
are integers that sum EXACTLY to the target (plain `round()` on each share
independently does not guarantee this -- see `allocate_counts_proportional`).
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np


def allocate_counts_proportional(scores: Sequence[float], total: int) -> List[int]:
    """Split `total` (a non-negative int) across `len(scores)` buckets,
    proportional to `scores`, using the largest-remainder method (a.k.a.
    Hamilton's apportionment method) so the result sums EXACTLY to `total`.

    nb(i) = floor(total * score(i) / sum(scores)), then the `total -
    sum(floor shares)` remaining units go one-by-one to the buckets with the
    largest fractional remainder (ties broken by ascending index, for
    determinism) -- e.g. this reproduces the worked example from the spec:
    scores=[0.8, 0.7, 0.6, 0.5, 0.4], total=250 -> [67, 58, 50, 42, 33].

    If every score is 0 (or `scores` is empty), falls back to an even split
    (still via the largest-remainder method, on uniform weights) rather than
    dividing by zero.

    Raises `ValueError` on a negative `total` or a negative score.
    """
    n = len(scores)
    if total < 0:
        raise ValueError(f"total must be >= 0, got {total}")
    if n == 0:
        if total != 0:
            raise ValueError("Cannot allocate a nonzero total across zero buckets")
        return []
    scores_arr = np.asarray(scores, dtype=np.float64)
    if np.any(scores_arr < 0):
        raise ValueError("scores must all be non-negative")

    score_sum = scores_arr.sum()
    if score_sum <= 0:
        # No signal to weight by -- split as evenly as possible instead of
        # dividing by zero.
        scores_arr = np.ones(n, dtype=np.float64)
        score_sum = float(n)

    exact_shares = total * scores_arr / score_sum
    floor_shares = np.floor(exact_shares).astype(np.int64)
    remainder = int(total - floor_shares.sum())

    if remainder > 0:
        fractional = exact_shares - floor_shares
        # Largest fractional remainder first; stable sort keeps ascending
        # index order among ties, for determinism.
        order = np.argsort(-fractional, kind="stable")
        for idx in order[:remainder]:
            floor_shares[idx] += 1

    return [int(v) for v in floor_shares]


def pick_with_duplication(ordered_indices: Sequence[int], count: int) -> List[int]:
    """Return exactly `count` indices drawn from `ordered_indices` (already
    sorted by preference, most-preferred first): every distinct index is
    used at least once (up to availability) before any is repeated, and
    repeats cycle through `ordered_indices` in the same preference order --
    e.g. `pick_with_duplication([a, b], 5) == [a, b, a, b, a]`.

    Used whenever a bucket needs more picks than it has distinct members
    available (a small HDBSCAN cluster asked to contribute more vectors
    than it contains, or a position's ranked word list asked to fill more
    synthetic images than it has distinct selected words) -- per the
    spec's confirmed rule: duplicate rather than error.

    `count=0` returns an empty list; `ordered_indices` must be non-empty
    whenever `count > 0`.
    """
    if count == 0:
        return []
    n = len(ordered_indices)
    if n == 0:
        raise ValueError("Cannot pick from an empty sequence")
    if count <= n:
        return list(ordered_indices[:count])
    full_cycles, remainder = divmod(count, n)
    return list(ordered_indices) * full_cycles + list(ordered_indices[:remainder])
