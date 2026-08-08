> ⚠️ **The `hybrid` and `hybrid_rerank` rows are STALE — do not quote them as
> current.** They were measured with *unweighted* reciprocal rank fusion. The
> committed default is now weighted (`rrf_weights = {"fts": 0.5, "dense": 1.0}`,
> see `src/regsearch/config.py`), so both fused arms would score differently
> today. **No replacement numbers have been measured yet** and none are guessed
> at here. Re-run after the reranker fine-tune completes:
> `regsearch eval --split test --origin citation --out docs/ablation.md`.
>
> The `fts` and `dense` rows are **unaffected** — neither arm reads
> `rrf_weights`.

| arm | Recall@50 | nDCG@10 | MRR | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| `fts` | 0.0462 | 0.0146 | 0.0430 | 1434.4 | 3265.0 |
| `dense` | 0.1236 | 0.0536 | 0.1100 | 38.3 | 50.3 |
| `hybrid` ⚠️ stale | 0.0992 | 0.0339 | 0.0792 | 1409.1 | 3369.9 |
| `hybrid_rerank` ⚠️ stale | 0.1212 | 0.0513 | 0.0971 | 9630.2 | 15192.2 |

_n=171 queries, split=test, origin=citation, canonicalize=True._

_⚠️ Fused arms measured under unweighted RRF; the committed default is now
`w_fts=0.5`. Pending re-measurement._

**Weak supervision.** Labels are citation-derived, not human relevance judgements: a query is a paper's title and its positives are the papers it cites. The same signal trains the reranker, so `hybrid_rerank` rows are scored against the family of labels they learn from. Not a human-judged result.

**Off-the-shelf reranker.** `hybrid_rerank` uses the public `ms-marco-MiniLM-L-6-v2` checkpoint. No fine-tuned model exists — the training script is written but has never been run.
