"""Ranking metrics.

Implemented directly rather than pulled from a library so the definitions are
inspectable -- nDCG in particular has several conventions in the wild and it
matters which one a reported number used.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(retrieved: Sequence[int], relevant: set[int], k: int) -> float:
    """Fraction of relevant items appearing in the top k.

    Denominator is |relevant|, not k -- with fewer than k relevant docs, a
    perfect ranking must still score 1.0.
    """
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: Sequence[int], relevant: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / k


def reciprocal_rank(retrieved: Sequence[int], relevant: set[int]) -> float:
    """1/rank of the first relevant hit; 0 if none. MRR is the mean of these."""
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def dcg(gains: Sequence[float]) -> float:
    # log2(i+1) with i 1-based, so the rank-1 gain is undiscounted.
    return sum(g / math.log2(i + 1) for i, g in enumerate(gains, start=1))


def ndcg_at_k(
    retrieved: Sequence[int], relevance: dict[int, int], k: int
) -> float:
    """Normalised DCG with the standard linear-gain convention.

    Gain is the graded relevance itself (not 2^rel - 1). Both conventions are
    common; the linear form is used here and stated so the number is
    interpretable. The ideal ranking sorts all judged items by relevance
    descending, so an unreachable perfect ranking is what we normalise against.
    """
    if not relevance:
        return 0.0
    gains = [float(relevance.get(doc, 0)) for doc in retrieved[:k]]
    ideal = sorted((float(v) for v in relevance.values()), reverse=True)[:k]
    idcg = dcg(ideal)
    return (dcg(gains) / idcg) if idcg > 0 else 0.0


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile. p in [0, 100]."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, math.ceil(p / 100.0 * len(ordered)) - 1)
    return ordered[idx]
