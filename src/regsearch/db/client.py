"""Postgres access layer.

Everything goes through a single connection pool. pgvector types are registered
per-connection so `vector` columns round-trip as numpy arrays / lists.
"""

from __future__ import annotations

import atexit
import logging
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from regsearch.config import get_settings

log = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

_pool: ConnectionPool | None = None


def _configure(conn: psycopg.Connection) -> None:
    # Must run on every new connection: pgvector's adapters are per-connection.
    register_vector(conn)


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        s = get_settings()
        # Close the pool at exit rather than leaving it to __del__. The pool's
        # destructor joins its worker threads, which raises
        # PythonFinalizationError if it runs during interpreter shutdown.
        # atexit fires early enough that the join still succeeds.
        atexit.register(close_pool)
        _pool = ConnectionPool(
            s.dsn,
            min_size=1,
            max_size=8,
            configure=_configure,
            kwargs={"row_factory": dict_row},
            open=True,
        )
    return _pool


@contextmanager
def connection():
    with get_pool().connection() as conn:
        yield conn


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


# --------------------------------------------------------------------- setup
def apply_schema() -> None:
    """Create tables/indexes if absent. Idempotent."""
    sql = SCHEMA_PATH.read_text()
    with connection() as conn:
        conn.execute(sql)
        conn.commit()
    log.info("schema applied")


def build_vector_index(m: int = 16, ef_construction: int = 64) -> None:
    """Build the HNSW index.

    Deliberately called AFTER bulk load: inserting into an existing HNSW index
    is far slower than one bulk build. Safe to re-run -- IF NOT EXISTS.
    """
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SET maintenance_work_mem = '2GB'")
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS passages_embedding_idx
                ON passages USING hnsw (embedding vector_cosine_ops)
                WITH (m = {int(m)}, ef_construction = {int(ef_construction)})
                """
            )
        conn.commit()
    log.info("hnsw index built (m=%d, ef_construction=%d)", m, ef_construction)


def analyze() -> None:
    with connection() as conn:
        conn.execute("ANALYZE documents")
        conn.execute("ANALYZE passages")
        conn.commit()


# -------------------------------------------------------------------- upsert
def upsert_documents(rows: Sequence[dict[str, Any]]) -> dict[tuple[str, str], int]:
    """Insert documents, returning {(source, ext_id): doc_id} for all rows.

    ON CONFLICT DO UPDATE (rather than DO NOTHING) so re-ingest refreshes
    mutable fields like cited_by, and so RETURNING yields a row for every input
    -- DO NOTHING would silently skip existing rows and leave gaps in the map.
    """
    if not rows:
        return {}

    sql = """
        INSERT INTO documents
            (source, ext_id, pmid, pmcid, doi, title, abstract,
             journal, pub_year, cited_by, is_open_access)
        VALUES
            (%(source)s, %(ext_id)s, %(pmid)s, %(pmcid)s, %(doi)s, %(title)s,
             %(abstract)s, %(journal)s, %(pub_year)s, %(cited_by)s, %(is_open_access)s)
        ON CONFLICT (source, ext_id) DO UPDATE SET
            cited_by  = EXCLUDED.cited_by,
            abstract  = COALESCE(EXCLUDED.abstract, documents.abstract),
            title     = EXCLUDED.title
        RETURNING doc_id, source, ext_id
    """
    out: dict[tuple[str, str], int] = {}
    with connection() as conn:
        with conn.cursor() as cur:
            for r in rows:
                cur.execute(sql, r)
                got = cur.fetchone()
                if got:
                    out[(got["source"], got["ext_id"])] = got["doc_id"]
        conn.commit()
    return out


def upsert_passages(rows: Sequence[dict[str, Any]]) -> int:
    """Insert passage chunks. Existing (doc_id, ordinal) pairs are left alone.

    DO NOTHING here (unlike documents): re-running ingest must not clobber
    embeddings that a later GPU pass already wrote.
    """
    if not rows:
        return 0
    sql = """
        INSERT INTO passages (doc_id, ordinal, section, text)
        VALUES (%(doc_id)s, %(ordinal)s, %(section)s, %(text)s)
        ON CONFLICT (doc_id, ordinal) DO NOTHING
    """
    with connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, rows)
            n = cur.rowcount
        conn.commit()
    return max(n, 0)


def update_embeddings(pairs: Iterable[tuple[int, Any]]) -> int:
    """Write embeddings for (passage_id, vector) pairs."""
    pairs = list(pairs)
    if not pairs:
        return 0
    with connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE passages SET embedding = %s WHERE passage_id = %s",
                [(vec, pid) for pid, vec in pairs],
            )
        conn.commit()
    return len(pairs)


def iter_unembedded(batch_size: int = 1000):
    """Yield batches of passages still missing an embedding."""
    while True:
        with connection() as conn:
            rows = conn.execute(
                """
                SELECT passage_id, text FROM passages
                WHERE embedding IS NULL
                ORDER BY passage_id
                LIMIT %s
                """,
                (batch_size,),
            ).fetchall()
        if not rows:
            return
        yield rows


# --------------------------------------------------------------------- stats
def stats() -> dict[str, int]:
    with connection() as conn:
        row = conn.execute(
            """
            SELECT
              (SELECT count(*) FROM documents)                        AS documents,
              (SELECT count(*) FROM passages)                         AS passages,
              (SELECT count(*) FROM passages WHERE embedding IS NULL) AS unembedded,
              (SELECT count(*) FROM citation_contexts)                AS citation_contexts,
              (SELECT count(*) FROM eval_queries)                     AS eval_queries,
              (SELECT count(*) FROM eval_qrels)                       AS eval_qrels
            """
        ).fetchone()
    return dict(row) if row else {}
