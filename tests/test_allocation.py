import pytest

from src.allocation import allocate_counts_proportional, pick_with_duplication


def test_allocate_counts_matches_spec_worked_example():
    """scores=[0.8,0.7,0.6,0.5,0.4], total=250 -> [67,58,50,42,33], exactly
    the worked example from the 2026-09 methodology revision."""
    result = allocate_counts_proportional([0.8, 0.7, 0.6, 0.5, 0.4], 250)
    assert result == [67, 58, 50, 42, 33]
    assert sum(result) == 250


def test_allocate_counts_always_sums_exactly_to_total():
    # A case where naive independent rounding would NOT sum to the target.
    scores = [1.0, 1.0, 1.0]
    result = allocate_counts_proportional(scores, 10)
    assert sum(result) == 10
    assert result == [4, 3, 3]  # largest-remainder: first bucket wins the tie (ascending index)


def test_allocate_counts_all_zero_scores_falls_back_to_even_split():
    result = allocate_counts_proportional([0.0, 0.0, 0.0, 0.0], 10)
    assert sum(result) == 10
    assert result == [3, 3, 2, 2]


def test_allocate_counts_empty_scores_zero_total():
    assert allocate_counts_proportional([], 0) == []


def test_allocate_counts_rejects_nonzero_total_on_empty_scores():
    with pytest.raises(ValueError):
        allocate_counts_proportional([], 5)


def test_allocate_counts_rejects_negative_total():
    with pytest.raises(ValueError):
        allocate_counts_proportional([1.0], -1)


def test_allocate_counts_rejects_negative_score():
    with pytest.raises(ValueError):
        allocate_counts_proportional([1.0, -0.5], 10)


def test_allocate_counts_zero_total_gives_all_zero_buckets():
    assert allocate_counts_proportional([0.8, 0.2], 0) == [0, 0]


def test_pick_with_duplication_no_repeat_needed():
    assert pick_with_duplication(["a", "b", "c"], 2) == ["a", "b"]


def test_pick_with_duplication_cycles_in_order():
    assert pick_with_duplication(["a", "b"], 5) == ["a", "b", "a", "b", "a"]


def test_pick_with_duplication_exact_fit():
    assert pick_with_duplication(["a", "b"], 2) == ["a", "b"]


def test_pick_with_duplication_zero_count():
    assert pick_with_duplication(["a"], 0) == []


def test_pick_with_duplication_empty_source_raises_when_count_positive():
    with pytest.raises(ValueError):
        pick_with_duplication([], 3)
