#!/usr/bin/env python
"""Measure the lexical arm: latency AND quality, in one pass, per configuration.

Exists because the two numbers must move together or not at all. `fts` was made
~35x more accurate by ORing its query terms and ~250x slower by the same change;
any further tuning has to be reported as a pair, so this script refuses to print
one without the other.

Why not `regsearch eval`: the full harness runs all four arms, and the
`hybrid_rerank` arm costs 5-10 s/query on CPU. Nothing here needs it. This
retrieves the `fts` arm only, caches its ranked passage lists to JSON, and
scores them offline -- so a scoring bug costs a re-score, not a re-retrieval.

Scoring mirrors `eval.harness.evaluate_arm` exactly, including the two things
that are easy to get subtly wrong:
  * `search()` retrieves `candidate_k` passages and then slices to k=50, so
    Recall@50 is over documents drawn from the top-50 PASSAGES, not top-50 docs;
  * duplicate preprint/published records are collapsed via the canonical map
    before scoring, on both the ranking and the qrels.

Usage:
    .venv/bin/python scripts/bench_fts.py --label baseline
    .venv/bin/python scripts/bench_fts.py --label pruned-0.05
    .venv/bin/python scripts/bench_fts.py --compare baseline pruned-0.05
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from regsearch.config import get_settings  # noqa: E402
from regsearch.db import client as db  # noqa: E402
from regsearch.eval.harness import load_eval_set  # noqa: E402
from regsearch.eval.metrics import (  # noqa: E402
    ndcg_at_k,
    percentile,
    recall_at_k,
    reciprocal_rank,
)
from regsearch.retrieve.search import fts_search  # noqa: E402

log = logging.getLogger("bench_fts")

RUNS_DIR = Path(__file__).resolve().parents[1] / "data" / "fts_bench"

# Latency is meaningless on a cold shared_buffers, and the first query of a run
# also pays psycopg's connection setup. A fixed warmup makes every configuration
# pay the same startup cost, so the comparison is between query plans rather
# than between cache states.
WARMUP = 3


def run_config(label: str, n: int | None, k_recall: int = 50) -> dict:
    s = get_settings()
    candidate_k = max(s.bm25_topk, s.dense_topk)

    eval_set = load_eval_set(split="test", origin="citation")
    if n:
        eval_set = eval_set[:n]
    log.info("label=%s  queries=%d  candidate_k=%d", label, len(eval_set), candidate_k)

    for item in eval_set[:WARMUP]:
        fts_search(item["query_text"], k=candidate_k)

    runs = []
    t0 = time.perf_counter()
    for i, item in enumerate(eval_set, 1):
        q = item["query_text"]
        t = time.perf_counter()
        hits = fts_search(q, k=candidate_k)
        latency = (time.perf_counter() - t) * 1000.0
        runs.append(
            {
                "query_id": item["query_id"],
                "qrels": item["qrels"],
                "latency_ms": latency,
                "n_hits": len(hits),
                "hits": [[h.passage_id, h.doc_id] for h in hits],
            }
        )
        if i % 25 == 0:
            log.info("  %d/%d (%.0fs)", i, len(eval_set), time.perf_counter() - t0)

    return {
        "label": label,
        "candidate_k": candidate_k,
        "k_recall": k_recall,
        "settings": {
            "fts_or_semantics": s.fts_or_semantics,
            "fts_prune_common_terms": getattr(s, "fts_prune_common_terms", None),
            "fts_df_max_frac": getattr(s, "fts_df_max_frac", None),
            "fts_min_terms": getattr(s, "fts_min_terms", None),
            "fts_join_after_limit": getattr(s, "fts_join_after_limit", None),
        },
        "runs": runs,
    }


def score(payload: dict, cmap: dict[int, int]) -> dict[str, float]:
    """Doc-level Recall@50 / nDCG@10 / MRR over a cached run, plus latency."""
    k_recall = payload.get("k_recall", 50)
    recalls, ndcgs, rrs, lats, nhits = [], [], [], [], []

    for run in payload["runs"]:
        seen: set[int] = set()
        doc_ranking: list[int] = []
        # search() slices the candidate list to k before the harness sees it.
        for _pid, did in run["hits"][:k_recall]:
            d = cmap.get(did, did)
            if d not in seen:
                seen.add(d)
                doc_ranking.append(d)

        qrels: dict[int, int] = {}
        for d, rel in run["qrels"].items():
            c = cmap.get(int(d), int(d))
            qrels[c] = max(qrels.get(c, 0), int(rel))
        relevant = {d for d, rel in qrels.items() if rel > 0}

        recalls.append(recall_at_k(doc_ranking, relevant, k_recall))
        ndcgs.append(ndcg_at_k(doc_ranking, qrels, 10))
        rrs.append(reciprocal_rank(doc_ranking, relevant))
        lats.append(run["latency_ms"])
        nhits.append(run["n_hits"])

    n = max(len(payload["runs"]), 1)
    return {
        "n": len(payload["runs"]),
        "recall@50": sum(recalls) / n,
        "ndcg@10": sum(ndcgs) / n,
        "mrr": sum(rrs) / n,
        "p50_ms": percentile(lats, 50),
        "p95_ms": percentile(lats, 95),
        "mean_ms": sum(lats) / n,
        "mean_hits": sum(nhits) / n,
    }


HEADER = (
    f"{'label':<22} | {'n':>3} | {'Recall@50':>9} | {'nDCG@10':>8} | {'MRR':>7} | "
    f"{'p50 ms':>8} | {'p95 ms':>9}"
)


def fmt(label: str, m: dict[str, float]) -> str:
    return (
        f"{label:<22} | {m['n']:>3} | {m['recall@50']:>9.4f} | {m['ndcg@10']:>8.4f} | "
        f"{m['mrr']:>7.4f} | {m['p50_ms']:>8.1f} | {m['p95_ms']:>9.1f}"
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", help="run this configuration and cache it under LABEL")
    ap.add_argument("--n", type=int, default=None, help="limit to first N queries")
    ap.add_argument("--compare", nargs="*", help="score cached labels side by side")
    args = ap.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    cmap = db.load_canonical_map()

    if args.label:
        payload = run_config(args.label, args.n)
        path = RUNS_DIR / f"{args.label}.json"
        path.write_text(json.dumps(payload))
        log.info("cached -> %s", path)
        print("\n" + HEADER)
        print("-" * len(HEADER))
        print(fmt(args.label, score(payload, cmap)))

    labels = args.compare
    if labels is not None:
        if not labels:
            labels = sorted(p.stem for p in RUNS_DIR.glob("*.json"))
        print("\n" + HEADER)
        print("-" * len(HEADER))
        rows = []
        for lb in labels:
            p = RUNS_DIR / f"{lb}.json"
            if not p.exists():
                log.warning("no cached run for %r", lb)
                continue
            payload = json.loads(p.read_text())
            m = score(payload, cmap)
            rows.append((lb, m, payload))
            print(fmt(lb, m))

        if len(rows) >= 2:
            base_lb, base, _ = rows[0]
            print(f"\nrelative to {base_lb!r}:")
            for lb, m, _ in rows[1:]:
                print(
                    f"  {lb:<20} Recall@50 {100 * (m['recall@50'] / base['recall@50'] - 1):+6.1f}%  "
                    f"nDCG@10 {100 * (m['ndcg@10'] / base['ndcg@10'] - 1):+6.1f}%  "
                    f"MRR {100 * (m['mrr'] / base['mrr'] - 1):+6.1f}%  "
                    f"p50 {base['p50_ms'] / max(m['p50_ms'], 1e-9):.1f}x faster  "
                    f"p95 {base['p95_ms'] / max(m['p95_ms'], 1e-9):.1f}x faster"
                )
        for lb, _m, payload in rows:
            print(f"  [{lb}] settings: {payload.get('settings')}")

    if not args.label and args.compare is None:
        ap.error("pass --label to measure, or --compare to score cached runs")


if __name__ == "__main__":
    main()
