"""Metric definitions.

These are the numbers the ablation table reports, so the conventions matter more
than the code: nDCG has several forms in the wild, and a table is only
comparable to another if both used the same one. Each test below pins one
convention choice that `metrics.py` documents in prose.
"""

from __future__ import annotations

import math

import pytest

from regsearch.eval.metrics import (
    dcg,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


# ------------------------------------------------------------------ recall
def test_recall_denominator_is_relevant_count_not_k():
    """With fewer relevant docs than k, a perfect ranking must still score 1.0.

    Using k as the denominator is the classic bug here: it would cap this at
    2/50 and make every arm look broken.
    """
    assert recall_at_k([7, 3, 9], {7, 3}, k=50) == 1.0


def test_recall_partial():
    assert recall_at_k([1, 2, 3], {1, 9}, k=50) == 0.5


def test_recall_respects_cutoff():
    # The relevant doc sits at rank 3, outside k=2.
    assert recall_at_k([8, 9, 1], {1}, k=2) == 0.0
    assert recall_at_k([8, 9, 1], {1}, k=3) == 1.0


def test_recall_with_no_relevant_docs_is_zero_not_nan():
    """Guards the empty-qrels division. A NaN here would poison the mean."""
    assert recall_at_k([1, 2], set(), k=50) == 0.0


def test_recall_ignores_duplicate_retrievals():
    """Set intersection means a repeated doc cannot count twice."""
    assert recall_at_k([1, 1, 1], {1, 2}, k=50) == 0.5


# --------------------------------------------------------------- precision
def test_precision_denominator_is_k():
    assert precision_at_k([1, 2, 3, 4], {1, 3}, k=4) == 0.5


def test_precision_k_zero_is_zero_not_nan():
    assert precision_at_k([1], {1}, k=0) == 0.0


# ---------------------------------------------------------- reciprocal rank
def test_reciprocal_rank_is_one_based():
    """A hit at position 1 scores 1.0, not 1/0 or 0.5."""
    assert reciprocal_rank([4, 5], {4}) == 1.0
    assert reciprocal_rank([5, 4], {4}) == 0.5
    assert reciprocal_rank([5, 6, 4], {4}) == pytest.approx(1 / 3)


def test_reciprocal_rank_uses_first_hit_only():
    assert reciprocal_rank([9, 1, 2], {1, 2}) == 0.5


def test_reciprocal_rank_no_hit_is_zero():
    assert reciprocal_rank([1, 2], {9}) == 0.0


# ---------------------------------------------------------------- dcg/nDCG
def test_dcg_rank_one_is_undiscounted():
    """log2(i+1) with i 1-based => the first gain divides by log2(2) == 1."""
    assert dcg([5.0]) == 5.0


def test_dcg_discount_matches_closed_form():
    assert dcg([3.0, 2.0]) == pytest.approx(3.0 + 2.0 / math.log2(3))


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k([1, 2], {1: 1, 2: 1}, k=10) == 1.0


def test_ndcg_uses_linear_gain_not_exponential():
    """Gain is the graded relevance itself, not 2**rel - 1.

    Both conventions are common and they disagree numerically, so this pins the
    one the README's table is computed under. Under exponential gain the same
    ranking would score ~0.7602.
    """
    got = ndcg_at_k([2, 1], {1: 3, 2: 1}, k=10)
    expected = (1.0 + 3.0 / math.log2(3)) / (3.0 + 1.0 / math.log2(3))
    assert got == pytest.approx(expected)
    assert got == pytest.approx(0.7967, abs=1e-4)


def test_ndcg_ideal_is_capped_at_k():
    """IDCG truncates to k as well.

    If the ideal ranking summed all judged items while the actual was cut at k,
    a query with more relevant docs than k could never reach 1.0 and the metric
    would silently penalise recall-heavy queries.
    """
    rel = {1: 1, 2: 1, 3: 1}
    assert ndcg_at_k([1, 2], rel, k=2) == 1.0


def test_ndcg_unjudged_docs_score_zero_gain():
    # Doc 99 is not in the qrels; it contributes nothing but still consumes rank 1.
    got = ndcg_at_k([99, 1], {1: 1}, k=10)
    assert got == pytest.approx(1.0 / math.log2(3))


def test_ndcg_empty_relevance_is_zero_not_nan():
    assert ndcg_at_k([1, 2], {}, k=10) == 0.0


def test_ndcg_all_zero_relevance_is_zero_not_nan():
    """IDCG is 0 here, so the normalisation would divide by zero."""
    assert ndcg_at_k([1, 2], {1: 0, 2: 0}, k=10) == 0.0


# ------------------------------------------------------------- percentile
def test_percentile_nearest_rank():
    vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert percentile(vals, 50) == 5
    assert percentile(vals, 95) == 10
    assert percentile(vals, 100) == 10


def test_percentile_does_not_interpolate():
    """Nearest-rank returns an observed value, never a synthetic midpoint.

    p95 latency should be a request that actually happened.
    """
    assert percentile([10.0, 20.0], 95) == 20.0
    assert percentile([10.0, 20.0], 50) == 10.0


def test_percentile_p0_does_not_underflow_index():
    """ceil(0)-1 == -1 would wrap to the *slowest* value. Clamped to 0."""
    assert percentile([1, 2, 3], 0) == 1


def test_percentile_empty_is_zero():
    assert percentile([], 50) == 0.0


def test_percentile_sorts_input():
    assert percentile([9.0, 1.0, 5.0], 50) == 5.0
