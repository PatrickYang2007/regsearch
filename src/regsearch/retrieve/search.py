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
# websearch_to_tsquery ANDs its terms. That is correct for short keyword
# queries and catastrophic for long ones: a 10-word title requires a passage
# containing all 10 stems, which on this corpus matched 1 passage out of 99,567
# and pinned the arm's Recall@50 at 0.0013. Retrieval wants "rank by how many
# terms match", not "require every term".
#
# There is no or_to_tsquery in Postgres, so the parse is rewritten: split the
# tsquery on its top-level AND, OR the positive operands back together, and
# re-AND the negated ones. Negation must stay conjunctive -- folding !x into
# the OR would make the query match every passage that merely lacks x.
# Phrase operands ('a' <-> 'b', from quoted input) survive intact because the
# split is on ' & ' only.
_FTS_OR_TSQUERY = """
    WITH parsed AS (
        SELECT websearch_to_tsquery('english', %(q)s) AS tq
    ),
    parts AS (
        SELECT unnest(string_to_array(tq::text, ' & ')) AS part FROM parsed
    ),
    agg AS (
        SELECT string_agg(part, ' | ') FILTER (WHERE part NOT LIKE '!%%') AS pos,
               string_agg(part, ' & ') FILTER (WHERE part LIKE '!%%')     AS neg
        FROM parts
    )
    SELECT concat_ws(' & ', '(' || pos || ')', neg)::tsquery AS q
    FROM agg WHERE pos IS NOT NULL
"""

_FTS_AND_TSQUERY = "SELECT websearch_to_tsquery('english', %(q)s) AS q"


def fts_search(query: str, k: int | None = None) -> list[Hit]:
    """Postgres full-text search ranked by ts_rank_cd.

    websearch_to_tsquery, not to_tsquery: it accepts free-form input ("chromatin
    accessibility -cancer") and never raises a syntax error on user text, which
    to_tsquery does readily.

    ts_rank_cd is a length-normalised TF-IDF variant -- cover density, not BM25.
    Named `fts` throughout so the eval table does not overclaim. Note that
    ts_rank_cd already scores partial matches by cover density, so ORing the
    terms changes which passages are *candidates*, not how they are ranked.
    """
    s = get_settings()
    k = k or s.bm25_topk

    qsql = _FTS_OR_TSQUERY if s.fts_or_semantics else _FTS_AND_TSQUERY
    sql = f"""
        WITH tsq AS ({qsql})
        SELECT p.passage_id, p.doc_id, p.text, d.title,
               ts_rank_cd(p.tsv, tsq.q, 32) AS score
        FROM passages p
        JOIN documents d USING (doc_id),
             tsq
        WHERE p.tsv @@ tsq.q
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
        # ef_search trades recall against latency at query time. Scoped to the
        # transaction so it cannot leak into other connections in the pool.
        #
        # set_config(..., is_local=true), not `SET LOCAL`: SET is a utility
        # statement and takes no bind parameters, so the placeholder arrives at
        # the server as a literal "$1" and errors. set_config is an ordinary
        # function call with identical semantics. Its value argument is text.
        conn.execute(
            "SELECT set_config('hnsw.ef_search', %s, true)", (str(ef_search),)
        )
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
def rrf_fuse(
    runs: dict[str, list[Hit]],
    k: int,
    rrf_k: int | None = None,
    weights: dict[str, float] | None = None,
) -> list[Hit]:
    """Weighted reciprocal rank fusion: score = sum of w_arm/(rrf_k + rank).

    Rank-based rather than score-based on purpose. ts_rank_cd and cosine
    similarity are on unrelated scales, so any weighted sum of raw scores would
    be dominated by whichever arm happens to have the larger range. Ranks are
    directly comparable and need no per-arm normalisation.

    The weights exist because unweighted RRF assumes its inputs are equally
    trustworthy, and here they are not. Measured on the 171-query test split,
    plain fusion scored 0.0992 Recall@50 against 0.1236 for `dense` alone --
    fusing the much weaker lexical arm actively destroyed a good ranking,
    because one bad arm got an equal vote. Down-weighting it lets fusion
    contribute the recall lexical search genuinely adds without letting it
    outvote the stronger arm.

    An arm with no entry in `weights` defaults to 1.0, so passing None
    reproduces classic unweighted RRF exactly.
    """
    s = get_settings()
    rrf_k = rrf_k or s.rrf_k
    if weights is None:
        weights = s.rrf_weights if s.rrf_weighted else {}

    fused: dict[int, Hit] = {}
    for arm, hits in runs.items():
        w = weights.get(arm, 1.0)
        if w == 0.0:
            # A zero weight means "do not fuse this arm at all". Skipping is not
            # the same as contributing 0.0: a hit found only by a zero-weighted
            # arm must not enter the candidate pool with score 0 and displace a
            # genuine hit at the tail.
            continue
        for h in hits:
            contribution = w / (rrf_k + h.rank)
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
