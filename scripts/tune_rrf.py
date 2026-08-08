#!/usr/bin/env python
"""Pick the RRF arm weights empirically instead of guessing them.

Plain unweighted fusion scored *worse* than the dense arm alone (0.0992 vs
0.1236 Recall@50), because RRF gives every input an equal vote and the lexical
arm is much weaker. The fix is to weight the arms -- but the weight has to come
from a measurement, not from taste.

The expensive part of a sweep is retrieval, not fusion: the lexical arm costs
~1.4 s per query, so re-running it once per candidate weight would take hours.
Retrieval does not depend on the weights at all, so each arm runs ONCE, its
ranked list is cached, and every weight is then evaluated by re-fusing the
cached lists in memory. That turns an hours-long sweep into one retrieval pass
plus arithmetic.

Usage:
    .venv/bin/python scripts/tune_rrf.py            # uses cache if present
    .venv/bin/python scripts/tune_rrf.py --refresh  # force re-retrieval
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from regsearch.config import get_settings  # noqa: E402
from regsearch.db import client as db  # noqa: E402
from regsearch.eval.harness import load_eval_set  # noqa: E402
from regsearch.eval.metrics import ndcg_at_k, recall_at_k, reciprocal_rank  # noqa: E402
from regsearch.retrieve.search import dense_search, fts_search  # noqa: E402

log = logging.getLogger("tune_rrf")

CACHE = Path(__file__).resolve().parents[1] / "data" / "rrf_tuning_runs.json"

# Weight on the lexical arm; dense is pinned at 1.0 since only the ratio
# matters -- scaling both weights scales every fused score identically and
# cannot reorder anything.
FTS_WEIGHTS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]


def build_cache(candidate_k: int) -> dict:
    """Run each arm once per query and store the ranked passage lists."""
    eval_set = load_eval_set(split="test", origin="citation")
    log.info("retrieving for %d queries (this is the slow part)", len(eval_set))

    runs = []
    t0 = time.perf_counter()
    for i, item in enumerate(eval_set, 1):
        q = item["query_text"]
        runs.append(
            {
                "qrels": item["qrels"],
                # (passage_id, doc_id) in rank order. Rank is the list index,
                # so it does not need storing.
                "fts": [[h.passage_id, h.doc_id] for h in fts_search(q, k=candidate_k)],
                "dense": [[h.passage_id, h.doc_id] for h in dense_search(q, k=candidate_k)],
            }
        )
        if i % 25 == 0:
            log.info("  %d/%d (%.0fs elapsed)", i, len(eval_set), time.perf_counter() - t0)

    return {"candidate_k": candidate_k, "runs": runs}


def fuse_and_score(
    cache: dict,
    w_fts: float,
    rrf_k: int,
    fuse_depth: int,
    k_recall: int,
    k_ndcg: int,
    cmap: dict[int, int],
) -> dict[str, float]:
    """Re-fuse cached runs at one weight setting and score the result.

    Mirrors rrf_fuse + evaluate_arm exactly: fuse at passage level, cut to
    fuse_depth, take the first k_recall, then collapse to documents in
    first-appearance order. Any divergence here would make the sweep optimise
    something the real system does not do.
    """
    recalls, ndcgs, rrs = [], [], []

    for run in cache["runs"]:
        scored: dict[int, float] = {}
        passage_doc: dict[int, int] = {}
        for arm, w in (("fts", w_fts), ("dense", 1.0)):
            if w == 0.0:
                continue
            for rank0, (pid, did) in enumerate(run[arm]):
                scored[pid] = scored.get(pid, 0.0) + w / (rrf_k + rank0 + 1)
                passage_doc[pid] = did

        top = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))[:fuse_depth][:k_recall]

        seen: set[int] = set()
        doc_ranking: list[int] = []
        for pid, _ in top:
            d = cmap.get(passage_doc[pid], passage_doc[pid])
            if d not in seen:
                seen.add(d)
                doc_ranking.append(d)

        qrels: dict[int, int] = {}
        for d, rel in run["qrels"].items():
            c = cmap.get(int(d), int(d))
            qrels[c] = max(qrels.get(c, 0), int(rel))
        relevant = {d for d, rel in qrels.items() if rel > 0}

        recalls.append(recall_at_k(doc_ranking, relevant, k_recall))
        ndcgs.append(ndcg_at_k(doc_ranking, qrels, k_ndcg))
        rrs.append(reciprocal_rank(doc_ranking, relevant))

    n = max(len(cache["runs"]), 1)
    return {
        "recall@50": sum(recalls) / n,
        "ndcg@10": sum(ndcgs) / n,
        "mrr": sum(rrs) / n,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-run retrieval")
    args = ap.parse_args()

    s = get_settings()
    candidate_k = max(s.bm25_topk, s.dense_topk)

    if CACHE.exists() and not args.refresh:
        log.info("using cached runs at %s", CACHE)
        cache = json.loads(CACHE.read_text())
    else:
        cache = build_cache(candidate_k)
        CACHE.write_text(json.dumps(cache))
        log.info("cached runs to %s", CACHE)

    cmap = db.load_canonical_map()
    log.info("canonical map: %d merged documents", len(cmap))

    print(f"\n{'w_fts':>7} | {'Recall@50':>10} | {'nDCG@10':>9} | {'MRR':>8}")
    print("-" * 44)
    rows = []
    for w in FTS_WEIGHTS:
        m = fuse_and_score(
            cache, w, s.rrf_k, s.rerank_topk, 50, 10, cmap
        )
        rows.append((w, m))
        print(f"{w:>7.2f} | {m['recall@50']:>10.4f} | {m['ndcg@10']:>9.4f} | {m['mrr']:>8.4f}")

    # w_fts = 0.0 is dense-alone-through-the-fusion-path, so it is the number
    # weighting has to beat to be worth having at all.
    baseline = next(m for w, m in rows if w == 0.0)
    best = max(rows, key=lambda r: r[1]["ndcg@10"])
    print(f"\ndense-only (w_fts=0): nDCG@10 {baseline['ndcg@10']:.4f}")
    print(f"best w_fts = {best[0]:.2f}: nDCG@10 {best[1]['ndcg@10']:.4f}")
    if best[1]["ndcg@10"] <= baseline["ndcg@10"]:
        print("\n=> fusion does NOT beat dense alone at any weight tested.")
        print("   Report that honestly rather than shipping a weight that loses.")


if __name__ == "__main__":
    main()
