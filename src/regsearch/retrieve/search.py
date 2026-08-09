"""Retrieval arms: lexical, dense, and reciprocal rank fusion.

Each arm returns a ranked list of the same `Hit` type so the eval harness can
treat them interchangeably and the ablation table is apples-to-apples.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
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
# The tsquery the lexical arm actually runs is built in Python, from the parse
# Postgres produces. Stemming has to happen server-side (it is the dictionary's
# job), but everything after it -- OR rewriting and common-term pruning -- is
# string manipulation on the parse, and doing it here rather than in SQL buys
# two things: it is unit-testable without a database, and the finished tsquery
# reaches the planner as a value it can estimate selectivity for, instead of
# being hidden behind a CTE it has to guess at.
#
# Cost: one extra round trip per query (sub-millisecond on a local socket,
# against an arm whose problem is measured in seconds).
_PARSE_SQL = "SELECT websearch_to_tsquery('english', %(q)s)::text AS tq"

# A plain single-lexeme operand, e.g. 'chromatin'. Anything else -- a phrase
# ('enhanc' <-> 'promot'), a parenthesised group, a weighted or prefix operand
# -- deliberately does not match; see _bare_lexeme.
_BARE_LEXEME_RE = re.compile(r"^'((?:[^']|'')*)'$")


def _bare_lexeme(part: str) -> str | None:
    """The single lexeme a tsquery operand names, or None if it names no single one.

    Only a plain quoted operand has a document frequency that can be looked up
    in `lexeme_df`. Phrase operands are left alone on purpose: a phrase is
    already far more selective than either of its words, so pruning it on one
    word's df would throw away the most discriminating thing in the query.
    """
    m = _BARE_LEXEME_RE.match(part.strip())
    if m is None:
        return None
    return m.group(1).replace("''", "'")


def split_tsquery(tq_text: str) -> tuple[list[str], list[str]]:
    """Split a top-level-AND tsquery into its (positive, negated) operands.

    websearch_to_tsquery only ever emits a conjunction at the top level, so
    splitting the rendered text on ' & ' is a complete parse of that level.
    Phrase operands survive intact because ' <-> ' is a different separator.

    An empty tsquery (whitespace input, or input that is all stop words) yields
    two empty lists, which callers must treat as "matches nothing" -- Postgres
    cannot even cast '()' to tsquery.
    """
    if not tq_text.strip():
        return [], []
    pos: list[str] = []
    neg: list[str] = []
    for part in tq_text.split(" & "):
        (neg if part.startswith("!") else pos).append(part)
    return pos, neg


def prune_common_terms(
    positives: list[str],
    common: Mapping[str, int],
    min_terms: int,
) -> tuple[list[str], list[str]]:
    """Split positive operands into (kept, dropped) on corpus document frequency.

    `common` is {lexeme: ndoc} for lexemes ALREADY known to exceed the
    threshold, so membership is the entire test and an absent lexeme is kept.

    The backstop matters more than the rule. A query whose every term is
    corpus-common ("gene expression analysis") would prune to nothing and return
    zero hits, which is strictly worse than being slow. When fewer than
    `min_terms` operands survive, the RAREST of the dropped ones are restored
    until they do -- so such a query still runs on the most selective evidence
    it has, even when that evidence is weak.

    Order is preserved and ties break on original position, so the same query
    always produces the same tsquery and two measurements are comparable.
    """
    keep_idx = [i for i, p in enumerate(positives) if _bare_lexeme(p) not in common]
    drop_idx = [i for i, p in enumerate(positives) if _bare_lexeme(p) in common]

    if len(keep_idx) < min_terms and drop_idx:
        # Stable sort over an already-ascending index list, so equal df keeps
        # the earlier term.
        drop_idx.sort(key=lambda i: common.get(_bare_lexeme(positives[i]) or "", 0))
        n_restore = min(min_terms - len(keep_idx), len(drop_idx))
        keep_idx.extend(drop_idx[:n_restore])
        drop_idx = drop_idx[n_restore:]

    return (
        [positives[i] for i in sorted(keep_idx)],
        [positives[i] for i in sorted(drop_idx)],
    )


def assemble_tsquery(positives: list[str], negatives: list[str]) -> str:
    """Rebuild a tsquery with the positives OR'd and the negations still AND'd.

    Negation must stay conjunctive. Folding !x into the OR would match every
    passage that merely lacks x, which is nearly the whole corpus -- an
    exclusion would turn into the broadest possible inclusion.

    Returns '' when there is nothing positive to match, including the
    negation-only case ("-cancer"): a query that only says what it does not want
    has no candidate set, and Postgres rejects an empty parenthesised tsquery
    outright.
    """
    if not positives:
        return ""
    q = "(" + " | ".join(positives) + ")"
    if negatives:
        q += " & " + " & ".join(negatives)
    return q


def build_fts_tsquery(
    parsed: str,
    or_semantics: bool = True,
    common: Mapping[str, int] | None = None,
    min_terms: int = 3,
) -> str:
    """Turn websearch_to_tsquery's output into the tsquery the arm will run.

    or_semantics=False returns the parse untouched -- that is the original
    all-terms-required behaviour, kept reproducible on purpose.

    Pure and database-free so it can be tested directly; the caller supplies
    `common` from the materialised lexeme_df table.
    """
    if not or_semantics:
        return parsed
    positives, negatives = split_tsquery(parsed)
    if common:
        positives, _dropped = prune_common_terms(positives, common, min_terms)
    return assemble_tsquery(positives, negatives)


# Scoring every OR match is the arm's whole cost: ts_rank_cd has to decode
# tsvector positions per row, and a title query OR-matches ~52k of 99,567
# passages. The LIMIT is applied to that ranking, so nothing is truncated before
# it is scored -- it is a top-N heapsort over the full match set.
#
# `documents` is joined AFTER the LIMIT. The previous shape joined first, which
# ran ~52,000 primary-key lookups (158k buffer hits) to attach a title to 100
# surviving rows. Result-identical: passages.doc_id is NOT NULL with a foreign
# key to documents, so the inner join cannot drop a passage.
_FTS_SEARCH_JOIN_LAST = """
    WITH cand AS (
        SELECT p.passage_id, p.doc_id, p.text,
               ts_rank_cd(p.tsv, %(tq)s::tsquery, 32) AS score
        FROM passages p
        WHERE p.tsv @@ %(tq)s::tsquery
        ORDER BY score DESC, p.passage_id
        LIMIT %(k)s
    )
    SELECT c.passage_id, c.doc_id, c.text, d.title, c.score
    FROM cand c
    JOIN documents d USING (doc_id)
    ORDER BY c.score DESC, c.passage_id
"""

# Pre-fix shape, kept so fts_join_after_limit=False reproduces the old plan.
_FTS_SEARCH_JOIN_FIRST = """
    SELECT p.passage_id, p.doc_id, p.text, d.title,
           ts_rank_cd(p.tsv, %(tq)s::tsquery, 32) AS score
    FROM passages p
    JOIN documents d USING (doc_id)
    WHERE p.tsv @@ %(tq)s::tsquery
    ORDER BY score DESC, p.passage_id
    LIMIT %(k)s
"""


@lru_cache(maxsize=8)
def _common_lexemes(min_frac: float) -> Mapping[str, int]:
    """Corpus-common lexemes, loaded once per process and per threshold.

    Cached because it is a corpus statistic that changes only when lexeme_df is
    rebuilt, and a per-query round trip for ~134 rows would eat the saving this
    whole mechanism exists to produce. Restart the process (or clear this cache)
    after `rebuild_lexeme_df`.
    """
    return db.load_common_lexemes(min_frac)


def fts_search(query: str, k: int | None = None) -> list[Hit]:
    """Postgres full-text search ranked by ts_rank_cd.

    websearch_to_tsquery, not to_tsquery: it accepts free-form input ("chromatin
    accessibility -cancer") and never raises a syntax error on user text, which
    to_tsquery does readily.

    ts_rank_cd is a length-normalised TF-IDF variant -- cover density, not BM25.
    Named `fts` throughout so the eval table does not overclaim. In particular
    it has no idf term, so dropping a common query term is not "down-weighting
    it to near zero" the way it would be under BM25: the term stops contributing
    to any cover at all. That is a real change to the ranking and is measured.
    """
    s = get_settings()
    k = k or s.bm25_topk

    common: Mapping[str, int] = {}
    if s.fts_or_semantics and s.fts_prune_common_terms:
        # An empty lexeme_df (never rebuilt) yields an empty map, which makes
        # pruning inert rather than an error. Degrade to slow, never to wrong.
        common = _common_lexemes(s.fts_df_max_frac)

    sql = _FTS_SEARCH_JOIN_LAST if s.fts_join_after_limit else _FTS_SEARCH_JOIN_FIRST

    with db.connection() as conn:
        parsed = conn.execute(_PARSE_SQL, {"q": query}).fetchone()["tq"]
        tq = build_fts_tsquery(
            parsed,
            or_semantics=s.fts_or_semantics,
            common=common,
            min_terms=s.fts_min_terms,
        )
        if not tq.strip():
            # No positive operand: an all-stop-word or negation-only query.
            # Postgres cannot cast an empty tsquery here, and there is nothing
            # to match anyway.
            return []
        rows = conn.execute(sql, {"tq": tq, "k": k}).fetchall()

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
