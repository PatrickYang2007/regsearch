"""HTTP layer.

No database anywhere in here. The retrieval layer is monkeypatched at the
boundary the API imports it through, so these tests assert what the API is
actually responsible for -- validating input, translating Hit objects into the
declared schema, and mapping failures onto status codes -- and not whether
Postgres is up. The arms themselves are covered by the eval harness against
real data; duplicating that here would only test the mock.

The mock is installed on `regsearch.api.app.search` rather than on
`regsearch.retrieve.search.search`, because the handler resolved the name at
import time and would not see a patch applied to the source module.
"""

from __future__ import annotations

from typing import get_args

import psycopg
import pytest
from fastapi.testclient import TestClient

from regsearch.api import app as api
from regsearch.api.models import Arm
from regsearch.retrieve.search import Arm as ArmLiteral
from regsearch.retrieve.search import Hit, SearchResponse


@pytest.fixture
def client() -> TestClient:
    return TestClient(api.app)


@pytest.fixture
def calls(monkeypatch) -> list[dict]:
    """Replace retrieval with a recorder that returns two plausible hits."""
    recorded: list[dict] = []

    def fake_search(query, arm="hybrid", k=10, candidate_k=None):
        recorded.append({"query": query, "arm": arm, "k": k})
        hits = [
            Hit(
                passage_id=11,
                doc_id=7,
                text="CTCF anchors chromatin loops at enhancer-promoter contacts.",
                title="Loop anchors",
                score=0.91,
                rank=1,
                components={"fts": 0.42, "dense": 0.88},
            ),
            Hit(
                passage_id=12,
                doc_id=8,
                text="Cohesin extrudes loops until it meets a bound CTCF site.",
                title="Extrusion",
                score=0.77,
                rank=2,
                components={"dense": 0.77},
            ),
        ]
        return SearchResponse(query=query, arm=arm, hits=hits[:k], latency_ms=12.5)

    monkeypatch.setattr(api, "search", fake_search)
    return recorded


# ------------------------------------------------------------------- health
def test_health_reports_counts_when_db_answers(client, monkeypatch):
    monkeypatch.setattr(
        api.db, "stats", lambda: {"documents": 19791, "passages": 99567}
    )
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["database"] is True
    assert body["counts"]["passages"] == 99567


def test_health_stays_200_when_db_is_down(client, monkeypatch):
    """Liveness must not depend on Postgres, which dies with the allocation."""

    def boom():
        raise psycopg.OperationalError("connection refused")

    monkeypatch.setattr(api.db, "stats", boom)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "degraded"
    assert body["database"] is False
    assert body["counts"] is None
    assert "connection refused" in body["detail"]


# ------------------------------------------------------------------- search
def test_search_defaults_to_hybrid(client, calls):
    r = client.get("/search", params={"q": "chromatin looping"})
    assert r.status_code == 200
    assert calls[0]["arm"] == "hybrid"
    assert r.json()["arm"] == "hybrid"


@pytest.mark.parametrize("arm", [a.value for a in Arm])
def test_every_arm_reaches_retrieval_as_a_plain_string(client, calls, arm):
    r = client.get("/search", params={"q": "enhancer", "arm": arm})
    assert r.status_code == 200
    assert calls[0]["arm"] == arm
    # search() branches on `arm == "fts"` etc; an enum member leaking through
    # would compare equal today but end up in the response payload as a type
    # the schema does not declare.
    assert type(calls[0]["arm"]) is str


def test_hit_payload_carries_component_scores(client, calls):
    hits = client.get("/search", params={"q": "ctcf"}).json()["hits"]
    assert [h["rank"] for h in hits] == [1, 2]
    assert hits[0]["components"] == {"fts": 0.42, "dense": 0.88}
    assert hits[1]["components"] == {"dense": 0.77}
    assert hits[0]["doc_id"] == 7
    assert hits[0]["title"] == "Loop anchors"
    assert hits[0]["text"].startswith("CTCF anchors")


def test_k_is_forwarded(client, calls):
    r = client.get("/search", params={"q": "ctcf", "k": 1})
    assert r.status_code == 200
    assert calls[0]["k"] == 1
    assert len(r.json()["hits"]) == 1


def test_unknown_arm_is_422_and_never_reaches_retrieval(client, calls):
    r = client.get("/search", params={"q": "ctcf", "arm": "bm25"})
    assert r.status_code == 422
    assert calls == []


@pytest.mark.parametrize("params", [{"k": 0}, {"k": 101}, {"k": "many"}])
def test_out_of_range_k_is_422(client, calls, params):
    r = client.get("/search", params={"q": "ctcf", **params})
    assert r.status_code == 422
    assert calls == []


@pytest.mark.parametrize("params", [{}, {"q": ""}])
def test_missing_or_empty_query_is_422(client, calls, params):
    assert client.get("/search", params=params).status_code == 422
    assert calls == []


def test_database_failure_during_search_is_503(client, monkeypatch):
    def boom(*a, **kw):
        raise psycopg.OperationalError("server closed the connection")

    monkeypatch.setattr(api, "search", boom)
    r = client.get("/search", params={"q": "ctcf", "arm": "dense"})
    assert r.status_code == 503


# ---------------------------------------------------------------- documents
def test_document_returns_metadata_and_passages(client, monkeypatch):
    monkeypatch.setattr(
        api,
        "get_document",
        lambda doc_id: {
            "doc_id": doc_id,
            "source": "MED",
            "ext_id": "12345",
            "pmid": "12345",
            "pmcid": None,
            "doi": "10.1000/x",
            "title": "Loop anchors",
            "abstract": "We map CTCF sites.",
            "journal": "Genome Biology",
            "pub_year": 2024,
            "cited_by": 3,
            "is_open_access": True,
            "canonical_doc_id": None,
            "passages": [
                {"passage_id": 11, "ordinal": 0, "section": "abstract", "text": "A."},
                {"passage_id": 12, "ordinal": 1, "section": "abstract", "text": "B."},
            ],
        },
    )
    body = client.get("/doc/7").json()
    assert body["doc_id"] == 7
    assert body["pub_year"] == 2024
    assert [p["ordinal"] for p in body["passages"]] == [0, 1]


def test_unknown_document_is_404(client, monkeypatch):
    monkeypatch.setattr(api, "get_document", lambda doc_id: None)
    r = client.get("/doc/999999")
    assert r.status_code == 404
    assert "999999" in r.json()["detail"]


def test_non_numeric_doc_id_is_422(client):
    assert client.get("/doc/abc").status_code == 422


def test_database_failure_during_document_lookup_is_503(client, monkeypatch):
    def boom(doc_id):
        raise psycopg.OperationalError("connection refused")

    monkeypatch.setattr(api, "get_document", boom)
    assert client.get("/doc/7").status_code == 503


# ------------------------------------------------------------------ schema
def test_arm_enum_matches_the_retrieval_layer():
    """Guards the one duplication the API cannot avoid.

    The allowed arms are spelled twice -- a Literal in retrieve.search and an
    Enum here, because FastAPI needs an Enum to emit a schema. If someone adds
    an arm to the retrieval layer, this fails rather than the API silently
    refusing the new arm with a 422.
    """
    assert {a.value for a in Arm} == set(get_args(ArmLiteral))


def test_openapi_declares_the_arms_and_hit_fields(client):
    spec = client.get("/openapi.json").json()
    hit = spec["components"]["schemas"]["HitModel"]["properties"]
    assert {"rank", "doc_id", "title", "text", "score", "components"} <= set(hit)
    assert set(spec["components"]["schemas"]["Arm"]["enum"]) == {
        a.value for a in Arm
    }


def test_openapi_never_calls_the_lexical_arm_bm25(client):
    """The lexical arm is ts_rank_cd. Calling it BM25 in a published schema
    would be a false claim about the system, so it is asserted, not remembered.
    """
    spec = client.get("/openapi.json").text.lower()
    assert "ts_rank_cd" in spec
    # The only permitted mention is the explicit denial in the API description.
    # Asserting the denial is *present* first, because count == count passes
    # vacuously at 0 == 0 -- which is exactly how the equality would rot if the
    # description were ever rewritten without it.
    assert "not bm25" in spec
    assert spec.count("bm25") == spec.count("not bm25")


def test_handlers_are_sync_so_starlette_offloads_them(client):
    """Pins the threadpool decision, which is invisible until it hurts.

    Retrieval is blocking end to end -- psycopg, and torch for the dense and
    rerank arms. Starlette runs a plain `def` handler in its worker threadpool
    and an `async def` one directly on the event loop, so making these async
    would let a single 6-second hybrid_rerank call stall every other request in
    the process. Nothing about the code would look wrong afterwards, hence a
    test rather than a comment.
    """
    import inspect

    paths = {"/health", "/search", "/doc/{doc_id}"}
    checked = set()
    for route in client.app.routes:
        if getattr(route, "path", None) in paths:
            assert not inspect.iscoroutinefunction(route.endpoint), route.path
            checked.add(route.path)
    # Or a renamed route would make the loop above pass by checking nothing.
    assert checked == paths


# ------------------------------------------------- get_document (no database)
class _FakeCursorResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows


class _FakeConn:
    """Records every statement so the SQL shape can be asserted without a DB."""

    def __init__(self, results):
        self._results = list(results)
        self.executed: list[tuple[str, dict]] = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        return _FakeCursorResult(self._results.pop(0))


@pytest.fixture
def fake_db(monkeypatch):
    """Swap db.connection() for a recorder. Returns a factory + a call log."""
    from contextlib import contextmanager

    from regsearch.api import queries

    checkouts: list[_FakeConn] = []

    def install(*result_sets):
        conn = _FakeConn(result_sets)

        @contextmanager
        def fake_connection():
            checkouts.append(conn)
            yield conn

        monkeypatch.setattr(queries.db, "connection", fake_connection)
        return conn

    install.checkouts = checkouts  # type: ignore[attr-defined]
    return install


def test_get_document_merges_passages_into_the_metadata_row(fake_db):
    from regsearch.api.queries import get_document

    conn = fake_db(
        [{"doc_id": 7, "title": "Loop anchors"}],
        [
            {"passage_id": 11, "ordinal": 0, "section": "title", "text": "A."},
            {"passage_id": 12, "ordinal": 1, "section": "abstract", "text": "B."},
        ],
    )
    doc = get_document(7)

    assert doc is not None
    assert doc["title"] == "Loop anchors"
    assert [p["ordinal"] for p in doc["passages"]] == [0, 1]
    # Both statements are parameterised on the same doc_id, and neither
    # interpolates it into the SQL text.
    assert [params for _, params in conn.executed] == [{"doc_id": 7}, {"doc_id": 7}]
    assert all("7" not in sql for sql, _ in conn.executed)


def test_get_document_returns_none_without_querying_passages(fake_db):
    """A miss must not pay for the passage scan -- 404 is the common bad path."""
    from regsearch.api.queries import get_document

    conn = fake_db([])  # no metadata row
    assert get_document(999999) is None
    assert len(conn.executed) == 1


def test_get_document_uses_one_pool_checkout(fake_db):
    """The pool tops out at 8 connections; taking two per request would halve
    the concurrency the service can sustain for no gain."""
    from regsearch.api.queries import get_document

    fake_db([{"doc_id": 7}], [])
    get_document(7)
    assert len(fake_db.checkouts) == 1  # type: ignore[attr-defined]
