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


# ------------------------------------------------------------ lexeme statistics
def rebuild_lexeme_df() -> dict[str, int]:
    """Materialise per-lexeme document frequency from ts_stat. Idempotent.

    ts_stat walks every tsvector in the table, so this is a full scan (~1 s on
    99,567 passages). Run it after ingest, next to analyze() -- the lexical
    query builder reads the result on every query and must not pay for the scan.

    Full replace inside one transaction rather than an upsert: a lexeme that has
    disappeared from the corpus must lose its row, and a half-rebuilt table
    would prune on a mixture of two corpora.
    """
    with connection() as conn:
        n_passages = conn.execute("SELECT count(*) AS n FROM passages").fetchone()["n"]
        if not n_passages:
            log.warning("no passages; lexeme_df left empty")
            conn.execute("TRUNCATE lexeme_df")
            conn.commit()
            return {"lexemes": 0, "n_passages": 0}

        conn.execute("TRUNCATE lexeme_df")
        conn.execute(
            """
            INSERT INTO lexeme_df (lexeme, ndoc, df_frac)
            SELECT word, ndoc, ndoc::real / %(n)s
            FROM ts_stat('SELECT tsv FROM passages')
            """,
            {"n": n_passages},
        )
        row = conn.execute("SELECT count(*) AS n FROM lexeme_df").fetchone()
        conn.commit()

    log.info(
        "lexeme_df: %s distinct lexemes over %s passages", row["n"], n_passages
    )
    return {"lexemes": row["n"], "n_passages": n_passages}


def load_common_lexemes(min_frac: float) -> dict[str, int]:
    """{lexeme: ndoc} for lexemes occurring in more than `min_frac` of passages.

    Only the common tail is returned, and that is the point: the query builder's
    test is membership, so anything absent is by definition rare enough to keep.
    On this corpus the result is tiny -- 134 lexemes at 5%, 47 at 10% -- which is
    what makes it cheap to hold in process memory instead of joining per query.
    """
    with connection() as conn:
        rows = conn.execute(
            "SELECT lexeme, ndoc FROM lexeme_df WHERE df_frac > %s",
            (float(min_frac),),
        ).fetchall()
    return {r["lexeme"]: r["ndoc"] for r in rows}


# ------------------------------------------------------- duplicate clustering
# Title after stripping every non-alphanumeric. Punctuation and case are the
# only things that reliably differ between a preprint and its published record
# ("Decoding cnidarian cell type gene regulation" vs the same with a trailing
# period), so normalising them away is what makes the two collide.
#
# DOI cannot be the key: a preprint keeps its 10.1101/... DOI after publication
# and the journal assigns a different one, so the twins never share a DOI.
_NORM_TITLE = "lower(regexp_replace(title, '[^a-z0-9]', '', 'gi'))"

# Below this length a normalised title is too generic to be evidence of
# identity on its own. Every cluster observed under it on this corpus was in
# fact a genuine duplicate (patent families), so the guard costs nothing here
# and protects against a future corpus where it would not.
MIN_TITLE_CHARS = 10


def rebuild_canonical_docs() -> dict[str, int]:
    """Point every document at its cluster representative. Idempotent.

    The representative prefers a published record over a preprint, then the
    lowest doc_id for determinism -- two runs must agree, or eval numbers from
    different runs are not comparable.
    """
    sql = f"""
        WITH norm AS (
            SELECT doc_id, source, {_NORM_TITLE} AS nt
            FROM documents
            WHERE title IS NOT NULL
              AND length({_NORM_TITLE}) >= %(min_chars)s
        ),
        ranked AS (
            SELECT doc_id,
                   first_value(doc_id) OVER (
                       PARTITION BY nt
                       -- PPR (preprint) sorts last, so the published record
                       -- wins the cluster whenever one exists.
                       ORDER BY (source = 'PPR'), doc_id
                   ) AS canon
            FROM norm
        )
        UPDATE documents d
           SET canonical_doc_id = r.canon
          FROM ranked r
         WHERE d.doc_id = r.doc_id
           AND d.canonical_doc_id IS DISTINCT FROM r.canon
    """
    with connection() as conn:
        # Documents no longer meeting the rule (title edited by a re-ingest)
        # must lose a stale pointer, or a cluster can outlive its evidence.
        conn.execute("UPDATE documents SET canonical_doc_id = NULL")
        conn.execute(sql, {"min_chars": MIN_TITLE_CHARS})
        row = conn.execute(
            """
            SELECT count(*) FILTER (WHERE canonical_doc_id <> doc_id) AS merged,
                   count(DISTINCT canonical_doc_id)
                     FILTER (WHERE canonical_doc_id <> doc_id) AS clusters
            FROM documents
            """
        ).fetchone()
        conn.commit()

    log.info(
        "canonical docs: %s documents merged into %s clusters",
        row["merged"], row["clusters"],
    )
    return {"merged": row["merged"], "clusters": row["clusters"]}


def load_canonical_map() -> dict[int, int]:
    """{doc_id: canonical_doc_id} for merged documents only.

    Self-mappings are omitted -- callers should treat a missing key as identity,
    which keeps this dict small (thousands, not the whole corpus).
    """
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT doc_id, canonical_doc_id
            FROM documents
            WHERE canonical_doc_id IS NOT NULL AND canonical_doc_id <> doc_id
            """
        ).fetchall()
    return {r["doc_id"]: r["canonical_doc_id"] for r in rows}


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
