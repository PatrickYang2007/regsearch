"""Read-only lookups the retrieval arms do not already provide.

Only the document view lives here. Anything that ranks belongs in
regsearch.retrieve.search and is called from there, never re-implemented.
"""

from __future__ import annotations

from typing import Any

from regsearch.db import client as db

_DOC_SQL = """
    SELECT doc_id, source, ext_id, pmid, pmcid, doi, title, abstract,
           journal, pub_year, cited_by, is_open_access, canonical_doc_id
    FROM documents
    WHERE doc_id = %(doc_id)s
"""

_PASSAGES_SQL = """
    SELECT passage_id, ordinal, section, text
    FROM passages
    WHERE doc_id = %(doc_id)s
    ORDER BY ordinal
"""


def get_document(doc_id: int) -> dict[str, Any] | None:
    """Document metadata plus its passages, or None if the id is unknown.

    Both statements run on one checkout of the shared pool: the pool tops out at
    8 connections, so a handler that took two would halve the concurrency the
    service can sustain for no gain.
    """
    with db.connection() as conn:
        doc = conn.execute(_DOC_SQL, {"doc_id": doc_id}).fetchone()
        if doc is None:
            return None
        passages = conn.execute(_PASSAGES_SQL, {"doc_id": doc_id}).fetchall()

    return {**doc, "passages": [dict(p) for p in passages]}
