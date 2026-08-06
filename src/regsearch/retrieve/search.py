"""Retrieval arms: lexical, dense, and reciprocal rank fusion.

Each arm returns a ranked list of the same `Hit` type so the eval harness can
treat them interchangeably and the ablation table is apples-to-apples.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from regsearch.config import get_settings
from regsearch.db import client as db

log = logging.getLogger(__name__)

Arm = Literal["fts", "dense", "hybrid", "hybrid_rerank"]


@dataclass
class Hit:
    passage_id: int
    doc_id: int
    text: str
    title: str
    score: float
    rank: int = 0
    # Per-arm provenance, so a fused hit can show where it came from.
    components: dict[str, float] = field(default_factory=dict)


@dataclass
class SearchResponse:
    query: str
    arm: Arm
    hits: list[Hit]
    latency_ms: float


# ------------------------------------------------------------------ lexical
def fts_search(query: str, k: int | None = None) -> list[Hit]:
    """Postgres full-text search ranked by ts_rank_cd.

    websearch_to_tsquery, not to_tsquery: it accepts free-form input ("chromatin
    accessibility -cancer") and never raises a syntax error on user text, which
    to_tsquery does readily.

    ts_rank_cd is a length-normalised TF-IDF variant -- cover density, not BM25.
    Named `fts` throughout so the eval table does not overclaim.
    """
    s = get_settings()
    k = k or s.bm25_topk

    sql = """
        SELECT p.passage_id, p.doc_id, p.text, d.title,
               ts_rank_cd(p.tsv, q, 32) AS score
        FROM passages p
        JOIN documents d USING (doc_id),
             websearch_to_tsquery('english', %(q)s) q
        WHERE p.tsv @@ q
        ORDER BY score DESC, p.passage_id
        LIMIT %(k)s
    """
    with db.connection() as conn:
        rows = conn.execute(sql, {"q": query, "k": k}).fetchall()

    return [
        Hit(
            passage_id=r["passage_id"],
            doc_id=r["doc_id"],
            text=r["text"],
            title=r["title"],
            score=float(r["score"]),
            rank=i + 1,
            components={"fts": float(r["score"])},
        )
        for i, r in enumerate(rows)
    ]


# -------------------------------------------------------------------- dense
def dense_search(query: str, k: int | None = None, ef_search: int = 100) -> list[Hit]:
    """Vector search over the HNSW index (cosine).

    Similarity is 1 - cosine_distance so that, like every other arm, higher is
    better and fusion never has to special-case a sign.
    """
    from regsearch.retrieve.embed import embed_query

    s = get_settings()
    k = k or s.dense_topk
    qvec = embed_query(query)

    sql = """
        SELECT p.passage_id, p.doc_id, p.text, d.title,
               1 - (p.embedding <=> %(v)s::vector) AS score
        FROM passages p
        JOIN documents d USING (doc_id)
        WHERE p.embedding IS NOT NULL
        ORDER BY p.embedding <=> %(v)s::vector
        LIMIT %(k)s
    """
    with db.connection() as conn:
        # ef_search trades recall against latency at query time. Set per
        # session so it cannot leak into other connections in the pool.
        conn.execute("SET LOCAL hnsw.ef_search = %s", (ef_search,))
        rows = conn.execute(sql, {"v": qvec, "k": k}).fetchall()

    return [
        Hit(
            passage_id=r["passage_id"],
            doc_id=r["doc_id"],
            text=r["text"],
            title=r["title"],
            score=float(r["score"]),
            rank=i + 1,
            components={"dense": float(r["score"])},
        )
        for i, r in enumerate(rows)
    ]


# ------------------------------------------------------------------- fusion
def rrf_fuse(runs: dict[str, list[Hit]], k: int, rrf_k: int | None = None) -> list[Hit]:
    """Reciprocal rank fusion: score = sum over arms of 1/(rrf_k + rank).

    Rank-based rather than score-based on purpose. ts_rank_cd and cosine
    similarity are on unrelated scales, so any weighted sum of raw scores would
    be dominated by whichever arm happens to have the larger range. Ranks are
    directly comparable and need no per-arm normalisation.
    """
    s = get_settings()
    rrf_k = rrf_k or s.rrf_k

    fused: dict[int, Hit] = {}
    for arm, hits in runs.items():
        for h in hits:
            contribution = 1.0 / (rrf_k + h.rank)
            if h.passage_id in fused:
                cur = fused[h.passage_id]
                cur.score += contribution
                cur.components[arm] = h.score
            else:
                fused[h.passage_id] = Hit(
                    passage_id=h.passage_id,
                    doc_id=h.doc_id,
                    text=h.text,
                    title=h.title,
                    score=contribution,
                    components={arm: h.score},
                )

    out = sorted(fused.values(), key=lambda h: (-h.score, h.passage_id))[:k]
    for i, h in enumerate(out):
        h.rank = i + 1
    return out


# ------------------------------------------------------------------ top-level
def search(
    query: str,
    arm: Arm = "hybrid",
    k: int = 10,
    candidate_k: int | None = None,
) -> SearchResponse:
    """Run one retrieval arm and return the top-k with wall-clock latency."""
    s = get_settings()
    candidate_k = candidate_k or max(s.bm25_topk, s.dense_topk)
    t0 = time.perf_counter()

    if arm == "fts":
        hits = fts_search(query, k=candidate_k)[:k]
    elif arm == "dense":
        hits = dense_search(query, k=candidate_k)[:k]
    elif arm in ("hybrid", "hybrid_rerank"):
        runs = {
            "fts": fts_search(query, k=candidate_k),
            "dense": dense_search(query, k=candidate_k),
        }
        fused = rrf_fuse(runs, k=s.rerank_topk)
        if arm == "hybrid_rerank":
            from regsearch.retrieve.rerank import rerank

            fused = rerank(query, fused)
        hits = fused[:k]
    else:
        raise ValueError(f"unknown arm: {arm}")

    for i, h in enumerate(hits):
        h.rank = i + 1

    return SearchResponse(
        query=query,
        arm=arm,
        hits=hits,
        latency_ms=(time.perf_counter() - t0) * 1000.0,
    )


def to_dict(resp: SearchResponse) -> dict[str, Any]:
    return {
        "query": resp.query,
        "arm": resp.arm,
        "latency_ms": round(resp.latency_ms, 2),
        "hits": [
            {
                "rank": h.rank,
                "passage_id": h.passage_id,
                "doc_id": h.doc_id,
                "title": h.title,
                "text": h.text,
                "score": round(h.score, 6),
                "components": {kk: round(vv, 6) for kk, vv in h.components.items()},
            }
            for h in resp.hits
        ],
    }
