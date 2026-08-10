"""Evaluation harness: run every retrieval arm over the same query set.

The whole point of the project is this comparison, so the harness is written to
make an unfair comparison hard:
  * every arm sees the identical query set and identical qrels;
  * metrics are computed at the DOCUMENT level, because a passage-level hit
    would let a chunkier arm win by returning three passages from one paper;
  * latency is wall-clock per query, reported at p50/p95, not averaged (a mean
    hides the tail that actually determines whether a service feels slow).

One place it does not manage that, stated up front because the number is
published: "Recall@50" is not recall over 50 documents. Each arm is asked for 50
PASSAGES, and those collapse to however many distinct documents they happen to
cover -- a different count for each arm. See the comment at the collapse in
`evaluate_arm` for the measured per-arm slot counts and which way the resulting
bias runs. nDCG@10 is not affected.
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
    """One arm's scores over the whole query set.

    `recall_at_50` is named for the 50 passages retrieved, not 50 documents;
    the field name is kept because published files and scripts parse it. See
    the caveat in `evaluate_arm` for what it actually counts.
    """

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

    `k_recall` is a PASSAGE depth, not a document depth: it is what `search` is
    asked for, and the collapse to documents happens after. Read the caveat at
    that collapse before quoting `recall_at_50` as recall over 50 documents.
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
        #
        # CAVEAT -- this is why "Recall@50" is a loose label. `search` was asked
        # for k_recall=50 PASSAGES and the collapse happens here, afterwards, so
        # `doc_ranking` is NOT 50 documents: it is however many distinct
        # documents those 50 passages happened to cover. The denominator of the
        # recall below is still |relevant| (see metrics.recall_at_k), but the
        # candidate list it searches is shorter than 50 and a different length
        # for every arm -- and the `k_recall` handed to recall_at_k is therefore
        # never the binding cutoff, it slices a list already shorter than that.
        #
        # Distinct documents inside the top-50 passages, over the 171 test
        # queries, counted from the cached runs under `data/`:
        #
        #     dense              mean 33.3   min 18   rrf_tuning_runs.json
        #     fts, shipped       mean 37.4   min 21   fts_bench/optimised.json
        #     fts, pre-pruning   mean 41.4   min 28   fts_bench/baseline.json
        #
        # So the arms are graded on UNEQUAL numbers of document slots -- `dense`
        # on 11% fewer than the `fts` row published beside it -- inside a table
        # whose entire purpose is arm-vs-arm comparison. The bias runs against
        # whichever arm repeats documents most, which is `dense`, which is the
        # arm that wins; the published conclusion is therefore conservative
        # rather than inflated. It is still a mislabelled metric, and README.md
        # and docs/ablation.md carry the same footnote.
        #
        # nDCG@10 does not have this problem. Dedup happens here, BEFORE the
        # [:10] slice, and every arm's doc_ranking is comfortably longer than
        # 10, so that column really is 10 distinct documents for every arm.
        #
        # Not fixed here, deliberately. The fix is to retrieve deeper and
        # truncate to a fixed DOCUMENT count so every arm gets 50 slots, but
        # that changes what the metric measures and invalidates every number
        # published in this repo until a full re-run.
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
