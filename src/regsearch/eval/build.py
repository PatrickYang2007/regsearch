"""Construct evaluation sets.

Two paths:

  build_citation_evalset()  -- automatic, from harvested citation labels. Split
                               by CITING DOCUMENT, so a paper's labels land
                               wholly in train or wholly in test. Splitting by
                               pair would put near-duplicate contexts from one
                               paper on both sides and inflate test scores.

  export_pool_for_judging() -- TREC-style pooling: take the top-n from every
                               arm, union them, and emit a CSV to hand-judge.
                               Pooling matters because judging only one arm's
                               output biases the qrels toward that arm --
                               anything the others uniquely found would be
                               scored as irrelevant purely because nobody looked.
"""

from __future__ import annotations

import csv
import hashlib
import logging
from pathlib import Path
from typing import Any

from regsearch.db import client as db
from regsearch.retrieve.search import Arm, search

log = logging.getLogger(__name__)


def _split_for(doc_id: int, test_frac: float = 0.2) -> str:
    """Deterministic hash split -- stable across runs and machines.

    Hashing the id beats random.shuffle here: re-running must not reshuffle the
    split, or numbers from two runs are not comparable.
    """
    h = int(hashlib.sha256(str(doc_id).encode()).hexdigest()[:8], 16)
    return "test" if (h % 100) < int(test_frac * 100) else "train"


def build_citation_evalset(test_frac: float = 0.2, min_positives: int = 2) -> dict[str, int]:
    """Turn harvested citation contexts into queries + qrels."""
    with db.connection() as conn:
        rows = conn.execute(
            """
            SELECT citing_doc_id, context_text,
                   array_agg(cited_doc_id) AS cited
            FROM citation_contexts
            GROUP BY citing_doc_id, context_text
            """
        ).fetchall()

    n_q = 0
    n_r = 0
    for r in rows:
        cited = [c for c in r["cited"] if c is not None]
        if len(cited) < min_positives:
            continue

        split = _split_for(r["citing_doc_id"], test_frac)
        with db.connection() as conn:
            got = conn.execute(
                """
                INSERT INTO eval_queries (query_text, origin, split)
                VALUES (%s, 'citation', %s)
                ON CONFLICT (query_text) DO UPDATE SET split = EXCLUDED.split
                RETURNING query_id
                """,
                (r["context_text"], split),
            ).fetchone()
            qid = got["query_id"]

            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO eval_qrels (query_id, doc_id, relevance)
                    VALUES (%s, %s, 1)
                    ON CONFLICT (query_id, doc_id) DO NOTHING
                    """,
                    [(qid, c) for c in cited],
                )
            conn.commit()
        n_q += 1
        n_r += len(cited)

    log.info("citation evalset: %d queries, %d qrels", n_q, n_r)
    return {"queries": n_q, "qrels": n_r}


def export_pool_for_judging(
    queries: list[str],
    out_path: Path,
    arms: list[Arm] | None = None,
    depth: int = 10,
) -> int:
    """Emit a pooled candidate CSV for manual relevance judging.

    Fill the `relevance` column with 0-3 and load it back with
    `regsearch import-qrels`.
    """
    arms = arms or ["fts", "dense", "hybrid"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["query", "doc_id", "title", "passage", "found_by", "relevance"])

        for q in queries:
            pooled: dict[int, dict[str, Any]] = {}
            for arm in arms:
                for h in search(q, arm=arm, k=depth).hits:
                    if h.doc_id not in pooled:
                        pooled[h.doc_id] = {
                            "title": h.title,
                            "text": h.text[:300],
                            "arms": set(),
                        }
                    pooled[h.doc_id]["arms"].add(arm)

            for doc_id, info in pooled.items():
                w.writerow(
                    [
                        q,
                        doc_id,
                        info["title"],
                        info["text"],
                        "|".join(sorted(info["arms"])),
                        "",  # judge fills this in
                    ]
                )
                n += 1

    log.info("wrote %d pooled candidates to %s", n, out_path)
    return n


def import_qrels(csv_path: Path, split: str = "test") -> dict[str, int]:
    """Load hand-filled judgements back as origin='manual'."""
    n_q = 0
    n_r = 0
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            rel_raw = (row.get("relevance") or "").strip()
            if not rel_raw:
                continue  # unjudged -- skip rather than assume irrelevant
            try:
                rel = int(rel_raw)
            except ValueError:
                continue

            with db.connection() as conn:
                got = conn.execute(
                    """
                    INSERT INTO eval_queries (query_text, origin, split)
                    VALUES (%s, 'manual', %s)
                    ON CONFLICT (query_text) DO UPDATE
                      SET origin = 'manual', split = EXCLUDED.split
                    RETURNING query_id
                    """,
                    (row["query"], split),
                ).fetchone()
                conn.execute(
                    """
                    INSERT INTO eval_qrels (query_id, doc_id, relevance)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (query_id, doc_id) DO UPDATE
                      SET relevance = EXCLUDED.relevance
                    """,
                    (got["query_id"], int(row["doc_id"]), rel),
                )
                conn.commit()
            n_r += 1
            n_q = max(n_q, 1)

    return {"qrels": n_r}
