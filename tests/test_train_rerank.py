"""Pair construction and negative sampling for the reranker fine-tune.

Everything here is pure: no GPU, no model download, no database. That is the
point of the split in `train_rerank.py` -- the judgement lives in functions over
plain dataclasses, and `mine_training_pairs` is a thin loop around them.

The tests that matter are the four poisoning cases, because each one silently
produces a *contradictory* training label rather than an error:
  * a preprint twin of a cited paper mined as a hard negative;
  * the query's own paper mined as a hard negative;
  * one document contributing several near-identical negatives;
  * a document judged IRRELEVANT by hand mined as a positive.
"""

from __future__ import annotations

import types

import pytest

import regsearch.retrieve.train_rerank as tr
from regsearch.retrieve.train_rerank import (
    Candidate,
    MiningStats,
    build_pairs,
    positive_clusters,
    select_hard_negatives,
    to_canonical,
)


def cand(doc_id: int, rank: int, text: str | None = None) -> Candidate:
    """A candidate whose passage_id is derived from doc_id, for readability."""
    return Candidate(
        passage_id=doc_id * 10 + rank,
        doc_id=doc_id,
        text=text or f"passage from doc {doc_id}",
        rank=rank,
    )


# --------------------------------------------------------------- canonical
def test_missing_key_means_identity():
    """db.load_canonical_map omits self-mappings; identity must be implied.

    Treating a missing key as "unknown" instead would make every non-duplicate
    document unmappable and break every comparison downstream.
    """
    assert to_canonical(7, {}) == 7
    assert to_canonical(7, {7: 3}) == 3


def test_positive_clusters_collapses_twins():
    # 5 is a preprint of 3; both are cited, so they are one positive, not two.
    assert positive_clusters([3, 5, 9], {5: 3}) == {3, 9}


# ---------------------------------------------------- hard negative sampling
def test_negatives_exclude_qrel_documents():
    cands = [cand(1, 1), cand(2, 2), cand(3, 3)]
    got = select_hard_negatives(cands, {2}, {}, max_negatives=10)
    assert [c.doc_id for c in got] == [1, 3]


def test_canonical_twin_of_a_positive_is_not_a_negative():
    """The 7%-of-corpus poisoning case.

    doc 5 is the preprint of cited doc 3. Its text is near identical to a
    positive, and its doc_id is genuinely absent from the qrels, so the naive
    "not in qrels" rule labels it 0 -- a contradictory label, not a hard one.
    """
    cands = [cand(5, 1), cand(9, 2)]
    got = select_hard_negatives(cands, positive_clusters([3], {5: 3}), {5: 3}, 10)
    assert [c.doc_id for c in got] == [9]


def test_twin_block_is_counted_separately_from_a_plain_qrel_hit():
    stats = MiningStats()
    select_hard_negatives(
        [cand(5, 1), cand(3, 2)], positive_clusters([3], {5: 3}), {5: 3}, 10, stats=stats
    )
    # doc 5 was blocked as a twin; doc 3 was a qrel outright and is not a twin.
    assert stats.twins_blocked == 1


def test_twin_block_is_counted_when_the_qrel_named_the_preprint():
    """The direction the canonical set cannot report on its own.

    `rebuild_canonical_docs` prefers the published record as representative, so
    when the *preprint* (doc 5) is the cited one, the positive cluster is doc 3.
    A candidate for doc 3 is then blocked as a twin -- correctly, it is the same
    paper -- but its raw doc_id is a member of the canonical positive set, so
    comparing `cand.doc_id` against that set reads it as the judged document
    itself and counts nothing. Only the raw qrel ids can tell the two apart.
    """
    stats = MiningStats()
    cmap = {5: 3}
    got = select_hard_negatives(
        [cand(3, 1), cand(9, 2)],
        positive_clusters([5], cmap),
        cmap,
        10,
        stats=stats,
        qrel_doc_ids=[5],
    )
    assert [c.doc_id for c in got] == [9]  # the exclusion itself was never wrong
    assert stats.twins_blocked == 1


def test_build_pairs_hands_the_twin_counter_the_raw_qrel_ids():
    """Pins the wiring: unfixed there, the fix above is dead code in production.

    `build_pairs` is the only caller that has both the raw qrels and the
    clusters, so it is the only place the counter can be supplied from.
    """
    stats = MiningStats()
    build_pairs(
        query_id=1,
        query_text="q",
        qrels={5: 1},  # the preprint is the cited record; doc 3 is its cluster
        candidates=[cand(3, 1), cand(9, 2)],
        canonical_map={5: 3},
        fallback_passages={},
        max_negatives=8,
        stats=stats,
    )
    assert stats.twins_blocked == 1


def test_citing_document_is_excluded_from_negatives():
    """The self-match trap.

    Queries are paper titles, so the query's own paper comes back at rank ~1,
    and it is never in its own qrels because papers do not cite themselves.
    Labelling a near-exact title match irrelevant is the worst available lesson.
    """
    cands = [cand(42, 1), cand(7, 2)]
    got = select_hard_negatives(cands, {99}, {}, 10, exclude_docs=[42])
    assert [c.doc_id for c in got] == [7]


def test_citing_document_is_excluded_through_its_canonical_twin():
    # The preprint of the citing paper is just as much a self-match.
    cmap = {42: 8}
    got = select_hard_negatives([cand(42, 1), cand(7, 2)], {99}, cmap, 10, exclude_docs=[8])
    assert [c.doc_id for c in got] == [7]


def test_negatives_are_deduped_by_cluster_not_passage():
    """Three chunks of one abstract are one negative document, not three."""
    cands = [cand(1, 1), cand(1, 2), cand(5, 3), cand(9, 4)]
    got = select_hard_negatives(cands, set(), {5: 1}, max_negatives=10)
    # doc 1 appears twice and doc 5 is its twin: all three collapse to one.
    assert [c.doc_id for c in got] == [1, 9]


def test_cap_keeps_the_hardest_negatives():
    """Rank order is the hardness order; the cap must not take an arbitrary set."""
    cands = [cand(i, i) for i in range(1, 11)]
    got = select_hard_negatives(cands, set(), {}, max_negatives=3)
    assert [c.rank for c in got] == [1, 2, 3]


def test_skip_top_n_drops_the_head_of_the_pool():
    """False negatives concentrate at the top; skipping is the denoising knob."""
    cands = [cand(i, i) for i in range(1, 6)]
    got = select_hard_negatives(cands, set(), {}, max_negatives=2, skip_top_n=2)
    assert [c.rank for c in got] == [3, 4]


def test_no_candidates_yields_no_negatives():
    assert select_hard_negatives([], {1}, {}, max_negatives=8) == []


# ------------------------------------------------------------ pair building
def test_retrieved_positive_beats_the_fallback_passage():
    """If the right paper is in the pool, train on the passage that was in it.

    That is the decision the reranker actually has to make at inference: the
    positive is present but ranked too low.
    """
    retrieved = cand(3, 2, text="the retrieved chunk")
    fallback = {3: Candidate(passage_id=999, doc_id=3, text="the lead chunk")}
    pairs = build_pairs(
        query_id=1,
        query_text="q",
        qrels={3: 1},
        candidates=[cand(1, 1), retrieved],
        canonical_map={},
        fallback_passages=fallback,
        max_negatives=8,
    )
    pos = [p for p in pairs if p.label == 1.0]
    assert len(pos) == 1
    assert pos[0].kind == "positive_retrieved"
    assert pos[0].passage_text == "the retrieved chunk"


def test_unretrieved_positive_falls_back_to_the_lead_passage():
    fallback = {3: Candidate(passage_id=999, doc_id=3, text="the lead chunk")}
    pairs = build_pairs(
        query_id=1,
        query_text="q",
        qrels={3: 1},
        candidates=[cand(1, 1), cand(2, 2)],
        canonical_map={},
        fallback_passages=fallback,
        max_negatives=8,
    )
    pos = [p for p in pairs if p.label == 1.0]
    assert [(p.kind, p.passage_id) for p in pos] == [("positive_fallback", 999)]


def test_query_with_no_negatives_contributes_nothing():
    """Positives alone give BCE no contrast -- it just learns to predict 1."""
    pairs = build_pairs(
        query_id=1,
        query_text="q",
        qrels={3: 1},
        candidates=[cand(3, 1)],  # the only candidate is the positive
        canonical_map={},
        fallback_passages={},
        max_negatives=8,
    )
    assert pairs == []


def test_query_with_no_usable_positive_contributes_nothing():
    # Positive doc has no passages and was not retrieved: nothing to label 1.
    pairs = build_pairs(
        query_id=1,
        query_text="q",
        qrels={3: 1},
        candidates=[cand(1, 1)],
        canonical_map={},
        fallback_passages={},
        max_negatives=8,
    )
    assert pairs == []


def test_positive_cap_bounds_the_119_qrel_outlier():
    """One train query has 119 qrels; uncapped it would swamp a batch."""
    fallback = {d: Candidate(passage_id=d, doc_id=d, text=f"lead {d}") for d in range(100, 220)}
    pairs = build_pairs(
        query_id=1,
        query_text="q",
        qrels={d: 1 for d in range(100, 220)},
        candidates=[cand(1, 1), cand(2, 2)],
        canonical_map={},
        fallback_passages=fallback,
        max_negatives=2,
        max_positives=4,
    )
    assert sum(1 for p in pairs if p.label == 1.0) == 4
    assert sum(1 for p in pairs if p.label == 0.0) == 2


def test_twin_qrels_produce_one_positive_not_two():
    """Both records of a duplicate cluster cited => one paper, one positive."""
    fallback = {
        3: Candidate(passage_id=30, doc_id=3, text="published"),
        5: Candidate(passage_id=50, doc_id=5, text="preprint"),
    }
    pairs = build_pairs(
        query_id=1,
        query_text="q",
        qrels={3: 1, 5: 1},
        candidates=[cand(9, 1)],
        canonical_map={5: 3},
        fallback_passages=fallback,
        max_negatives=8,
    )
    assert sum(1 for p in pairs if p.label == 1.0) == 1


def test_labels_and_stats_are_consistent():
    stats = MiningStats()
    pairs = build_pairs(
        query_id=7,
        query_text="a query",
        qrels={3: 1},
        candidates=[cand(3, 1), cand(4, 2), cand(6, 3)],
        canonical_map={},
        fallback_passages={},
        max_negatives=8,
        stats=stats,
    )
    assert {p.query_id for p in pairs} == {7}
    assert sorted(p.label for p in pairs) == [0.0, 0.0, 1.0]
    assert (stats.positives, stats.negatives) == (1, 2)


# ------------------------------------------------- graded relevance in qrels
@pytest.mark.parametrize("rel", [1, 2, 3])
def test_only_relevance_above_zero_becomes_a_positive(rel):
    """A hand-judged non-match must never arrive as a label-1.0 positive.

    The scale is graded 0-3 with 0 meaning IRRELEVANT (`schema.sql:144`) and
    `evaluate_arm` reads `rel > 0`; taking every qrel key instead treats
    "somebody looked at this and said no" as "somebody cited this". Citation
    qrels are all relevance 1, so this only bites once manual judgements land --
    which is exactly why it has to be pinned before then.
    """
    fallback = {
        3: Candidate(passage_id=30, doc_id=3, text="the relevant paper"),
        4: Candidate(passage_id=40, doc_id=4, text="judged irrelevant by hand"),
    }
    pairs = build_pairs(
        query_id=1,
        query_text="q",
        qrels={3: rel, 4: 0},
        candidates=[cand(8, 1)],
        canonical_map={},
        fallback_passages=fallback,
        max_negatives=8,
    )
    assert [(p.doc_id, p.kind) for p in pairs if p.label == 1.0] == [
        (3, "positive_fallback")
    ]


def test_a_relevance_zero_qrel_is_still_available_as_a_hard_negative():
    """Judged-and-rejected is the best hard negative there is, not an exclusion.

    Treating the qrel keys as positives cost this twice over: doc 4 became a
    positive *and* was blocked out of the negative pool, leaving the query with
    no negatives and dropping it from training entirely.
    """
    pairs = build_pairs(
        query_id=1,
        query_text="q",
        qrels={3: 1, 4: 0},
        candidates=[cand(4, 1)],
        canonical_map={},
        fallback_passages={3: Candidate(passage_id=30, doc_id=3, text="cited")},
        max_negatives=8,
    )
    assert [(p.doc_id, p.label) for p in pairs] == [(3, 1.0), (4, 0.0)]


# ------------------------------------------------------ drop-out attribution
def test_the_two_drop_out_reasons_are_counted_separately():
    """`build_pairs` returns [] for two opposite reasons; [] cannot say which.

    No negatives means the candidate pool collapsed onto the qrels; no positive
    means the cited paper is not in this corpus. Booking both as
    "without negatives" makes the second invisible in `training_meta.json`.
    """
    stats = MiningStats()

    # The only candidate IS the positive, so nothing is left to contrast it.
    assert build_pairs(
        query_id=1,
        query_text="q",
        qrels={3: 1},
        candidates=[cand(3, 1)],
        canonical_map={},
        fallback_passages={},
        max_negatives=8,
        stats=stats,
    ) == []
    assert (stats.queries_without_negatives, stats.queries_without_positives) == (1, 0)

    # Negatives exist, but the cited doc was never retrieved and has no lead
    # passage, so there is nothing to label 1.
    assert build_pairs(
        query_id=2,
        query_text="q",
        qrels={3: 1},
        candidates=[cand(1, 1)],
        canonical_map={},
        fallback_passages={},
        max_negatives=8,
        stats=stats,
    ) == []
    assert (stats.queries_without_negatives, stats.queries_without_positives) == (1, 1)

    # A query that yields pairs must bump neither counter.
    assert build_pairs(
        query_id=3,
        query_text="q",
        qrels={3: 1},
        candidates=[cand(3, 1), cand(1, 2)],
        canonical_map={},
        fallback_passages={},
        max_negatives=8,
        stats=stats,
    )
    assert (stats.queries_without_negatives, stats.queries_without_positives) == (1, 1)


def test_mining_books_each_drop_out_exactly_once(monkeypatch):
    """The loop must not re-count what `build_pairs` already counted.

    The database and `search` are stubbed rather than reached: this is about
    the bookkeeping in `mine_training_pairs`, which is the only place the two
    counters are ever read from (they land in `training_meta.json`).
    """
    eval_set = [
        {"query_id": 1, "query_text": "healthy", "qrels": {3: 1}},
        {"query_id": 2, "query_text": "no negatives", "qrels": {3: 1}},
        {"query_id": 3, "query_text": "no positive", "qrels": {77: 1}},
    ]
    pools = {
        "healthy": [cand(3, 1), cand(4, 2)],
        "no negatives": [cand(3, 1)],
        "no positive": [cand(4, 1)],
    }
    monkeypatch.setattr(tr, "load_eval_set", lambda split, origin: eval_set)
    monkeypatch.setattr(tr.db, "load_canonical_map", lambda: {})
    monkeypatch.setattr(tr, "load_citing_docs", lambda texts: {})
    monkeypatch.setattr(tr, "load_lead_passages", lambda doc_ids: {})
    monkeypatch.setattr(
        tr, "search", lambda q, arm, k: types.SimpleNamespace(hits=pools[q])
    )

    pairs, stats = tr.mine_training_pairs(progress_every=0)

    assert stats.queries_seen == 3
    assert stats.queries_used == 1
    assert stats.queries_without_negatives == 1
    assert stats.queries_without_positives == 1
    # Every seen query is accounted for exactly once.
    assert (
        stats.queries_used
        + stats.queries_without_negatives
        + stats.queries_without_positives
        == stats.queries_seen
    )
    assert {p.query_id for p in pairs} == {1}


@pytest.mark.parametrize("n", [0, 1, 5])
def test_negative_cap_is_respected(n):
    cands = [cand(i, i) for i in range(1, 21)]
    got = select_hard_negatives(cands, set(), {}, max_negatives=n)
    assert len(got) == n
