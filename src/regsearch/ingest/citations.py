"""Harvest weak relevance labels from the citation graph.

Hand-labelling enough pairs to train a cross-encoder is not feasible for a side
project, so labels come from citations: if paper A cites paper B, then something
about A is a plausible query for which B is a relevant answer.

Two tiers, because the strong version needs data we only have for part of the
corpus:

  Tier 1 (all docs)  -- the citing paper's TITLE is the pseudo-query, each cited
                        paper present in our corpus is a positive. Cheap, one
                        API call per document, and noisy: a title is a topic,
                        not a question.

  Tier 2 (OA only)   -- the actual sentence containing the citation marker,
                        pulled from full text. Much closer to a real query, but
                        Europe PMC only serves fullTextXML for the open-access
                        subset, so it does not cover the whole corpus.

Both are recorded in citation_contexts. Tier is inferable from the text, and the
README states which fed which experiment -- these labels train the reranker and
must never be reported as evaluation accuracy.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from regsearch.db import client as db
from regsearch.ingest.chunk import split_sentences
from regsearch.ingest.europepmc import EuropePMCClient

log = logging.getLogger(__name__)


def _corpus_id_map() -> dict[str, int]:
    """External identifiers -> doc_id, for resolving references to our corpus.

    Keyed by every identifier a reference might carry (pmid / pmcid / doi),
    because Europe PMC reference entries are inconsistent about which they
    include.
    """
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT doc_id, pmid, pmcid, doi FROM documents"
        ).fetchall()

    out: dict[str, int] = {}
    for r in rows:
        for key in (r["pmid"], r["pmcid"], r["doi"]):
            if key:
                out[str(key).strip().lower()] = r["doc_id"]
    return out


def _resolve(ref: dict[str, Any], id_map: dict[str, int]) -> int | None:
    for field in ("id", "pmid", "pmcid", "doi"):
        val = ref.get(field)
        if val and str(val).strip().lower() in id_map:
            return id_map[str(val).strip().lower()]
    return None


async def harvest_tier1(
    limit: int = 2000, min_refs: int = 3
) -> dict[str, int]:
    """Title-as-query labels for documents with resolvable references."""
    id_map = _corpus_id_map()

    with db.connection() as conn:
        docs = conn.execute(
            """
            SELECT d.doc_id, d.source, d.ext_id, d.title
            FROM documents d
            WHERE NOT EXISTS (
                SELECT 1 FROM citation_contexts c WHERE c.citing_doc_id = d.doc_id
            )
            ORDER BY d.cited_by DESC   -- well-cited papers have richer reference lists
            LIMIT %s
            """,
            (limit,),
        ).fetchall()

    n_pairs = 0
    n_docs = 0

    async with EuropePMCClient() as epmc:
        for doc in docs:
            try:
                refs = await epmc.references(doc["source"], doc["ext_id"])
            except Exception:
                log.debug("references failed for %s:%s", doc["source"], doc["ext_id"])
                continue

            cited_ids = {
                cid
                for ref in refs
                if (cid := _resolve(ref, id_map)) is not None
                and cid != doc["doc_id"]  # self-citation adds no signal
            }
            if len(cited_ids) < min_refs:
                continue

            rows = [
                {
                    "citing": doc["doc_id"],
                    "cited": cid,
                    "ctx": doc["title"],
                }
                for cid in cited_ids
            ]
            with db.connection() as conn:
                with conn.cursor() as cur:
                    cur.executemany(
                        """
                        INSERT INTO citation_contexts
                            (citing_doc_id, cited_doc_id, context_text)
                        VALUES (%(citing)s, %(cited)s, %(ctx)s)
                        ON CONFLICT DO NOTHING
                        """,
                        rows,
                    )
                conn.commit()

            n_pairs += len(rows)
            n_docs += 1
            if n_docs % 50 == 0:
                log.info("tier1: %d docs -> %d pairs", n_docs, n_pairs)

    return {"documents": n_docs, "pairs": n_pairs}


# Matches inline citation markers: [12], [3,4], (Smith et al., 2020)
_CITE_MARKER = re.compile(r"\[\d+(?:\s*[,-]\s*\d+)*\]|\([A-Z][A-Za-z]+ et al\.?,? \d{4}\)")


def extract_citing_sentences(full_text: str) -> list[str]:
    """Sentences that carry an inline citation marker.

    The marker itself is stripped: we want the claim, and leaving '[12]' in
    would let the reranker key on punctuation that no real query contains.
    """
    out: list[str] = []
    for sent in split_sentences(full_text):
        if _CITE_MARKER.search(sent):
            cleaned = _CITE_MARKER.sub("", sent).strip()
            cleaned = re.sub(r"\s{2,}", " ", cleaned)
            # Too-short fragments are usually table or figure debris.
            if 60 <= len(cleaned) <= 400:
                out.append(cleaned)
    return out


def run_harvest(limit: int = 2000, min_refs: int = 3) -> dict[str, int]:
    return asyncio.run(harvest_tier1(limit=limit, min_refs=min_refs))
