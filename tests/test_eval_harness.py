"""What the harness's "Recall@50" column actually measures.

`evaluate_arm` asks `search` for `k_recall` PASSAGES and collapses them to
documents afterwards, so the recall column is not recall over 50 documents and
the arms do not get the same number of document slots. That is documented at
length in `harness.evaluate_arm`; these tests pin it, so a future change to the
retrieval depth cannot quietly alter what the published number means without
also breaking a test that points at the docstring.

Nothing here touches Postgres: `search` is stubbed, and the canonical map is
passed in.
"""

from __future__ import annotations

import math

import pytest

from regsearch.eval import harness
from regsearch.retrieve.search import Hit, SearchResponse


def _response(doc_ids: list[int]) -> SearchResponse:
    """One passage per entry, in rank order; repeat a doc_id to repeat a paper."""
    return SearchResponse(
        query="q",
        arm="fts",
        hits=[
            Hit(
                passage_id=1000 + i,
                doc_id=d,
                text="",
                title="",
                score=1.0 / (i + 1),
                rank=i + 1,
            )
            for i, d in enumerate(doc_ids)
        ],
        latency_ms=0.0,
    )


@pytest.fixture
def stub_search(monkeypatch):
    """Make `search` return a fixed passage list, truncated to k like the real one."""

    def install(doc_ids: list[int]):
        def fake_search(query, arm, k):
            return _response(doc_ids[:k])

        monkeypatch.setattr(harness, "search", fake_search)

    return install


def _eval_set(relevant_doc: int):
    return [{"query_id": 1, "query_text": "q", "qrels": {relevant_doc: 1}}]


def test_arms_get_unequal_document_slots(stub_search):
    """The central caveat: 50 passages buy a different number of documents per arm.

    Both arms retrieve exactly 50 passages and both rank the relevant document
    at the same DOCUMENT depth. The arm that repeats papers runs out of passages
    before it gets there and is charged a miss; a genuine recall-over-50-
    documents metric would have scored them the same.
    """
    diverse = list(range(49)) + [900]  # 50 passages, 50 distinct documents
    chunky = [d for d in range(10) for _ in range(5)] + [900]  # 50 passages, 10 docs

    stub_search(diverse)
    r_diverse = harness.evaluate_arm("dense", _eval_set(900), canonical_map={})

    stub_search(chunky)
    r_chunky = harness.evaluate_arm("fts", _eval_set(900), canonical_map={})

    assert r_diverse.recall_at_50 == 1.0
    # doc 900 is the chunky arm's 11th distinct document but its 51st passage,
    # and only 50 passages were retrieved.
    assert r_chunky.recall_at_50 == 0.0


def test_k_recall_is_a_passage_depth_so_document_ranks_are_shallower(stub_search):
    """Document ranks are positions in a list shorter than `k_recall`.

    The relevant paper is the 50th passage but only the 5th distinct document,
    so its reciprocal rank is 1/5. If `k_recall` were a document depth this
    would be 1/50.
    """
    doc_ids = [d for d in range(4) for _ in range(12)] + [900]  # 49 passages
    stub_search(doc_ids)

    res = harness.evaluate_arm("fts", _eval_set(900), canonical_map={})

    assert res.mrr == pytest.approx(1 / 5)


def test_ndcg_at_10_does_get_ten_distinct_documents(stub_search):
    """nDCG@10 is not affected: dedup happens before the [:10] slice.

    30 passages of one paper collapse to a single rank slot, so the relevant
    paper is at document rank 2 and lands inside the nDCG window.
    """
    doc_ids = [1] * 30 + [900] + [2] * 19
    stub_search(doc_ids)

    res = harness.evaluate_arm("fts", _eval_set(900), canonical_map={})

    # Gain 1 at document rank 2; ideal is that same gain at rank 1.
    assert res.ndcg_at_10 == pytest.approx(1 / math.log2(3))


def test_canonical_twins_share_one_document_slot(stub_search):
    """A preprint and its published twin must not consume two slots.

    Otherwise an arm that returns both is quietly given an extra chance at the
    tail of the ranking on top of already being credited for one paper.
    """
    stub_search([1, 2, 2, 3])
    cmap = {2: 1}  # doc 2 is the preprint of doc 1

    res = harness.evaluate_arm("fts", _eval_set(3), canonical_map=cmap)

    # docs collapse to [1, 3]: doc 3 is at document rank 2, not rank 4.
    assert res.mrr == pytest.approx(1 / 2)
