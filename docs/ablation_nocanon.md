> ⚠️ **The `hybrid` and `hybrid_rerank` rows are STALE — do not quote them as
> current.** They were measured with *unweighted* reciprocal rank fusion. The
> committed default is now weighted (`rrf_weights = {"fts": 0.5, "dense": 1.0}`,
> see `src/regsearch/config.py`), so both fused arms would score differently
> today. **No replacement numbers have been measured yet** and none are guessed
> at here.
>
> The `fts` and `dense` rows are **unaffected** — neither arm reads
> `rrf_weights`.
>
> This file is the `--no-canonicalize` companion to `docs/ablation.md`, kept so
> the duplicate-clustering fix can be attributed separately from the lexical fix.

| arm | Recall@50 | nDCG@10 | MRR | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| `fts` | 0.0462 | 0.0128 | 0.0356 | 1546.5 | 3732.1 |
| `dense` | 0.1221 | 0.0483 | 0.1033 | 40.1 | 53.2 |
| `hybrid` ⚠️ stale | 0.0984 | 0.0287 | 0.0742 | 1493.2 | 3623.6 |
| `hybrid_rerank` ⚠️ stale | 0.1207 | 0.0488 | 0.0915 | 9267.0 | 14437.8 |

_n=171 queries, split=test, origin=citation, canonicalize=False._

_⚠️ Fused arms measured under unweighted RRF; the committed default is now
`w_fts=0.5`. Pending re-measurement._

**Weak supervision.** Labels are citation-derived, not human relevance judgements: a query is a paper's title and its positives are the papers it cites. The same signal trains the reranker, so `hybrid_rerank` rows are scored against the family of labels they learn from. Not a human-judged result.

**Off-the-shelf reranker.** `hybrid_rerank` uses the public `ms-marco-MiniLM-L-6-v2` checkpoint. No fine-tuned model exists — the training script is written but has never been run.
