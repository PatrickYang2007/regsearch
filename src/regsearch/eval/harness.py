"""Evaluation harness: run every retrieval arm over the same query set.

The whole point of the project is this comparison, so the harness is written to
make an unfair comparison hard:
  * every arm sees the identical query set and identical qrels;
  * metrics are computed at the DOCUMENT level, because a passage-level hit
    would let a chunkier arm win by returning three passages from one paper;
  * latency is wall-clock per query, reported at p50/p95, not averaged (a mean
    hides the tail that actually determines whether a service feels slow).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from regsearch.db import client as db
from regsearch.eval.metrics import (
    ndcg_at_k,
    percentile,
    recall_at_k,
    reciprocal_rank,
)
from regsearch.retrieve.search import Arm, search

log = logging.getLogger(__name__)


@dataclass
class ArmResult:
    arm: str
    n_queries: int
    recall_at_50: float
    ndcg_at_10: float
    mrr: float
    latency_p50_ms: float
    latency_p95_ms: float
    per_query: list[dict[str, Any]] = field(default_factory=list)


def load_eval_set(split: str = "test", origin: str = "manual") -> list[dict[str, Any]]:
    """Load queries plus their graded qrels.

    origin='manual' by default so headline numbers come from hand-judged
    queries only. Citation-derived labels train the reranker; letting them into
    the eval set would report the model's own training signal back as accuracy.
    """
    with db.connection() as conn:
        rows = conn.execute(
            """
            SELECT q.query_id, q.query_text,
                   COALESCE(
                     json_object_agg(r.doc_id, r.relevance)
                       FILTER (WHERE r.doc_id IS NOT NULL),
                     '{}'::json
                   ) AS qrels
            FROM eval_queries q
            LEFT JOIN eval_qrels r USING (query_id)
            WHERE q.split = %s AND q.origin = %s
            GROUP BY q.query_id, q.query_text
            ORDER BY q.query_id
            """,
            (split, origin),
        ).fetchall()

    out = []
    for r in rows:
        qrels = {int(k): int(v) for k, v in (r["qrels"] or {}).items()}
        if qrels:  # a query with no judgements cannot score anything
            out.append(
                {
                    "query_id": r["query_id"],
                    "query_text": r["query_text"],
                    "qrels": qrels,
                }
            )
    return out


def evaluate_arm(
    arm: Arm,
    eval_set: list[dict[str, Any]],
    k_recall: int = 50,
    k_ndcg: int = 10,
    canonical_map: dict[int, int] | None = None,
) -> ArmResult:
    """Score one arm.

    `canonical_map` collapses duplicate records (preprint/published twins) onto
    a cluster representative before scoring. Without it an arm that returns the
    preprint when the qrel names the published version is charged a miss for
    finding the right paper. Pass None to score raw doc_ids.
    """
    cmap = canonical_map or {}

    recalls: list[float] = []
    ndcgs: list[float] = []
    rrs: list[float] = []
    latencies: list[float] = []
    per_query: list[dict[str, Any]] = []

    for item in eval_set:
        # Two twins can both be judged relevant; collapsing them must keep the
        # stronger grade rather than whichever happened to be iterated last.
        qrels: dict[int, int] = {}
        for d, rel in item["qrels"].items():
            c = cmap.get(d, d)
            qrels[c] = max(qrels.get(c, 0), rel)
        relevant = {d for d, rel in qrels.items() if rel > 0}

        t0 = time.perf_counter()
        resp = search(item["query_text"], arm=arm, k=k_recall)
        latency = (time.perf_counter() - t0) * 1000.0

        # Passages -> documents, keeping first-appearance order. Without this
        # an arm returning 3 passages from one paper would look like 3 hits.
        # Canonicalising here also means a preprint and its published twin
        # collapse to one rank slot instead of occupying two.
        seen: set[int] = set()
        doc_ranking: list[int] = []
        for h in resp.hits:
            doc = cmap.get(h.doc_id, h.doc_id)
            if doc not in seen:
                seen.add(doc)
                doc_ranking.append(doc)

        r = recall_at_k(doc_ranking, relevant, k_recall)
        n = ndcg_at_k(doc_ranking, qrels, k_ndcg)
        rr = reciprocal_rank(doc_ranking, relevant)

        recalls.append(r)
        ndcgs.append(n)
        rrs.append(rr)
        latencies.append(latency)
        per_query.append(
            {
                "query_id": item["query_id"],
                "query": item["query_text"],
                f"recall@{k_recall}": round(r, 4),
                f"ndcg@{k_ndcg}": round(n, 4),
                "rr": round(rr, 4),
                "latency_ms": round(latency, 2),
            }
        )

    n_q = max(len(eval_set), 1)
    return ArmResult(
        arm=arm,
        n_queries=len(eval_set),
        recall_at_50=sum(recalls) / n_q,
        ndcg_at_10=sum(ndcgs) / n_q,
        mrr=sum(rrs) / n_q,
        latency_p50_ms=percentile(latencies, 50),
        latency_p95_ms=percentile(latencies, 95),
        per_query=per_query,
    )


def run_ablation(
    arms: list[Arm] | None = None,
    split: str = "test",
    origin: str = "manual",
    canonicalize: bool = True,
) -> list[ArmResult]:
    arms = arms or ["fts", "dense", "hybrid", "hybrid_rerank"]
    eval_set = load_eval_set(split=split, origin=origin)
    if not eval_set:
        raise RuntimeError(
            f"no judged queries for split={split!r} origin={origin!r}. "
            "Run `regsearch build-evalset` first."
        )

    cmap = db.load_canonical_map() if canonicalize else {}
    if canonicalize and not cmap:
        log.warning(
            "canonicalize=True but no duplicate clusters are recorded; "
            "run `regsearch dedup-docs` or numbers will be deflated by "
            "preprint/published twins scoring as misses"
        )
    log.info(
        "evaluating %d arms over %d queries (canonicalize=%s, %d merged docs)",
        len(arms), len(eval_set), canonicalize, len(cmap),
    )
    return [evaluate_arm(a, eval_set, canonical_map=cmap) for a in arms]
