"""FastAPI surface over the existing retrieval arms.

Every ranked response comes from regsearch.retrieve.search.search(). Nothing in
this package re-implements or re-tunes retrieval, and that is the point: the
ablation table in docs/ablation.md only describes *this service* if the service
runs the same code the eval measured. Any convenience reimplementation here
would silently invalidate the published numbers.

Handlers are plain `def`, not `async def`, so Starlette runs them in its worker
threadpool. The retrieval path is blocking end to end (psycopg, and torch for
the dense and rerank arms); an async handler would pin the event loop for the
whole query and stall every other request.
"""

from __future__ import annotations

import logging

import psycopg
from fastapi import FastAPI, HTTPException, Path, Query

from regsearch import __version__
from regsearch.api.models import (
    Arm,
    DocumentModel,
    HealthResponse,
    SearchResult,
)
from regsearch.api.queries import get_document
from regsearch.db import client as db
from regsearch.retrieve.search import search

log = logging.getLogger(__name__)

DESCRIPTION = """
Hybrid retrieval over regulatory-genomics literature.

Four retrieval arms, the same four the offline evaluation measures:

| arm | what it does |
|---|---|
| `fts` | Postgres full-text search ranked by `ts_rank_cd` (cover density, a length-normalised TF-IDF variant -- not BM25) |
| `dense` | vector search over an HNSW index of passage embeddings (cosine) |
| `hybrid` | reciprocal rank fusion of the two above |
| `hybrid_rerank` | the fused list re-scored by a cross-encoder |

Relevance labels behind the published comparison are citation-derived weak
supervision, not human judgements, and the cross-encoder is an off-the-shelf
checkpoint rather than a fine-tuned one.
"""

app = FastAPI(
    title="regsearch",
    version=__version__,
    description=DESCRIPTION,
    # The DB pool opens lazily on first use rather than at startup, so the
    # process comes up (and /health can report the outage) even when Postgres
    # is down -- which on this cluster is the normal state between allocations.
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness, database reachability, and corpus counts.

    Always 200 while the process is alive: this is a liveness signal, and a
    probe that raises tells a caller nothing about whether the server is up.
    Read `database` and `counts` to decide whether it is ready to serve.

    A first call against a dead database blocks until the connection pool gives
    up, so this is not a low-latency probe when Postgres is missing.
    """
    try:
        counts = db.stats()
    except Exception as exc:  # noqa: BLE001 -- see docstring: must not raise
        log.warning("health check could not reach the database: %s", exc)
        return HealthResponse(
            status="degraded",
            version=__version__,
            database=False,
            detail=str(exc),
        )
    return HealthResponse(
        status="ok", version=__version__, database=True, counts=counts
    )


@app.get("/search", response_model=SearchResult, tags=["retrieval"])
def search_endpoint(
    q: str = Query(..., min_length=1, max_length=1000, description="Query text."),
    arm: Arm = Query(Arm.hybrid, description="Which retrieval arm to run."),
    k: int = Query(10, ge=1, le=100, description="Number of hits to return."),
) -> SearchResult:
    """Run one retrieval arm and return the ranked passages.

    Cost, measured on the 1-CPU dev node: `dense` is the fast arm (~40 ms p50),
    `fts` runs ~1.4 s because it ORs its query terms and therefore ranks tens of
    thousands of candidate passages, `hybrid` pays for both.

    **`hybrid_rerank` takes roughly 5-10 SECONDS per query on CPU** -- it runs a
    cross-encoder forward pass over 100 fused candidates, and the first call
    additionally loads the model. It is not the default for that reason; ask for
    it explicitly and expect to wait. On this deployment it also monopolises the
    single CPU, so concurrent requests queue behind it.

    `arm` is validated against the four known arms before it reaches the
    retrieval layer; anything else is a 422.
    """
    try:
        # arm.value, not arm: search() takes the Literal string, and passing the
        # enum member would put a non-str type into SearchResponse.arm.
        resp = search(q, arm=arm.value, k=k)  # type: ignore[arg-type]
    except psycopg.Error as exc:
        # Covers pool timeouts too (PoolTimeout subclasses OperationalError).
        # The corpus DB is a dependency, not a caller error, hence 503.
        log.warning("search failed against the database: %s", exc)
        raise HTTPException(
            status_code=503, detail="corpus database unavailable"
        ) from exc

    return SearchResult.from_response(resp, k=k)


@app.get("/doc/{doc_id}", response_model=DocumentModel, tags=["corpus"])
def document_endpoint(
    doc_id: int = Path(..., ge=1, description="Internal document id."),
) -> DocumentModel:
    """Document metadata and its passages, in chunk order.

    `doc_id` is this corpus's internal id -- the one search hits carry -- not a
    PMID or a DOI.
    """
    try:
        doc = get_document(doc_id)
    except psycopg.Error as exc:
        log.warning("document lookup failed against the database: %s", exc)
        raise HTTPException(
            status_code=503, detail="corpus database unavailable"
        ) from exc

    if doc is None:
        raise HTTPException(status_code=404, detail=f"no document with doc_id {doc_id}")
    return DocumentModel(**doc)


def run(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Start the server. Entry point for a CLI command; not wired up here.

    Binds to loopback by default: on a shared cluster node an unauthenticated
    service on 0.0.0.0 is reachable by every other user on the box.
    """
    import uvicorn

    uvicorn.run("regsearch.api.app:app", host=host, port=port, reload=reload)
