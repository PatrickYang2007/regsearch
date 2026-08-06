"""Ingestion orchestration: Europe PMC -> documents + passages."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from regsearch.db import client as db
from regsearch.ingest.chunk import chunk_document
from regsearch.ingest.europepmc import EuropePMCClient, normalise_record

log = logging.getLogger(__name__)

# Seed queries spanning the regulatory-genomics space. Broad enough that the
# corpus has real topical structure (so retrieval is non-trivial), narrow
# enough to stay on-domain.
DEFAULT_QUERIES: list[str] = [
    'TITLE_ABS:"transcription factor binding site"',
    'TITLE_ABS:"chromatin accessibility"',
    'TITLE_ABS:"ATAC-seq"',
    'TITLE_ABS:"ChIP-seq" AND TITLE_ABS:"motif"',
    'TITLE_ABS:"enhancer" AND TITLE_ABS:"gene expression"',
    'TITLE_ABS:"cis-regulatory element"',
    'TITLE_ABS:"promoter" AND TITLE_ABS:"regulation"',
    'TITLE_ABS:"CTCF" AND TITLE_ABS:"chromatin"',
    'TITLE_ABS:"regulatory variant"',
    'TITLE_ABS:"expression quantitative trait loci"',
    'TITLE_ABS:"single-cell" AND TITLE_ABS:"chromatin"',
    'TITLE_ABS:"DNase hypersensitivity"',
    'TITLE_ABS:"histone modification" AND TITLE_ABS:"enhancer"',
    'TITLE_ABS:"deep learning" AND TITLE_ABS:"regulatory genomics"',
    'TITLE_ABS:"sequence model" AND TITLE_ABS:"gene regulation"',
]


async def ingest_query(
    epmc: EuropePMCClient,
    query: str,
    max_results: int,
    skip_seen: bool = True,
) -> dict[str, int]:
    """Fetch one query, write documents and their passage chunks."""
    if skip_seen:
        with db.connection() as conn:
            seen = conn.execute(
                "SELECT n_docs FROM ingest_log WHERE query = %s", (query,)
            ).fetchone()
        if seen:
            log.info("skip (already ingested): %s", query)
            return {"documents": 0, "passages": 0, "skipped": 1}

    records = await epmc.search_all(query, max_results=max_results)

    rows: list[dict[str, Any]] = []
    for rec in records:
        norm = normalise_record(rec)
        # Drop abstract-less records: a title-only passage is too thin to be a
        # useful retrieval unit and it dilutes the corpus.
        if norm and norm["ext_id"] and norm["abstract"]:
            rows.append(norm)

    if not rows:
        log.warning("no usable records for %s", query)
        return {"documents": 0, "passages": 0, "skipped": 0}

    id_map = db.upsert_documents(rows)

    passage_rows: list[dict[str, Any]] = []
    for r in rows:
        doc_id = id_map.get((r["source"], r["ext_id"]))
        if doc_id is None:
            continue
        for ordinal, p in enumerate(chunk_document(r["title"], r["abstract"])):
            passage_rows.append(
                {
                    "doc_id": doc_id,
                    "ordinal": ordinal,
                    "section": p["section"],
                    "text": p["text"],
                }
            )

    n_pass = db.upsert_passages(passage_rows)

    with db.connection() as conn:
        conn.execute(
            """
            INSERT INTO ingest_log (query, n_docs) VALUES (%s, %s)
            ON CONFLICT (query) DO UPDATE
              SET n_docs = EXCLUDED.n_docs, fetched_at = now()
            """,
            (query, len(rows)),
        )
        conn.commit()

    log.info("%s -> %d docs, %d passages", query, len(rows), n_pass)
    return {"documents": len(rows), "passages": n_pass, "skipped": 0}


async def ingest(
    queries: list[str] | None = None,
    max_results: int = 2000,
    skip_seen: bool = True,
) -> dict[str, int]:
    queries = queries or DEFAULT_QUERIES
    totals = {"documents": 0, "passages": 0, "skipped": 0}

    async with EuropePMCClient() as epmc:
        # Sequential across queries on purpose: concurrency lives inside the
        # client (semaphore + token bucket), and fanning out here would just
        # queue behind the same rate limiter while making failures harder to
        # attribute to a query.
        for q in queries:
            try:
                res = await ingest_query(epmc, q, max_results, skip_seen=skip_seen)
                for k, v in res.items():
                    totals[k] += v
            except Exception:
                log.exception("ingest failed for query: %s", q)

    return totals


def run_ingest(
    queries: list[str] | None = None,
    max_results: int = 2000,
    skip_seen: bool = True,
) -> dict[str, int]:
    return asyncio.run(ingest(queries, max_results, skip_seen))
