| arm | Recall@50 | nDCG@10 | MRR | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| `fts` | 0.0501 | 0.0172 | 0.0448 | 152.6 | 489.4 |
| `dense` | 0.1236 | 0.0536 | 0.1100 | 12.3 | 18.2 |
| `hybrid` | 0.1284 | 0.0500 | 0.1010 | 160.6 | 509.2 |
| `hybrid_rerank` | 0.1388 | 0.0523 | 0.1294 | 254.4 | 600.7 |

_n=171 queries, split=test, origin=citation, canonicalize=True._

**Weak supervision.** Labels are citation-derived, not human relevance judgements: a query is a paper's title and its positives are the papers it cites. The same signal trains the reranker, so `hybrid_rerank` rows are scored against the family of labels they learn from. Not a human-judged result.

**Recall@50 is over 50 passages, not 50 documents.** Each arm is asked for 50 passages and those are collapsed to their parent documents afterwards, so the column is recall among the distinct documents that fit inside a top-50 passage list. The arms consequently get unequal numbers of document slots: over these 171 queries `dense` averages 33.3 distinct documents (min 18) and `fts` 37.4 (min 21), so `dense` is graded on ~11% fewer slots than the arm printed beside it. The bias runs against whichever arm repeats documents most, which is `dense` — the arm that wins nDCG@10 — so the comparison understates it rather than flattering it. nDCG@10 is unaffected: the collapse happens before the top 10 is taken, so that column is 10 genuine documents for every arm.

**Latency is not comparable to earlier runs of this table.** These p50/p95 figures were measured on a GPU node with 2 CPUs; the previous published run used 8 CPUs and no GPU, and the one before that a single CPU. `hybrid_rerank` p50 moved 1185.2 → 254.4 ms purely because the cross-encoder now runs on the GPU. The quality columns are device-independent and remain comparable across all three runs.

**The reranker was retrained for this run and quality barely moved.** The previous checkpoint's hard negatives were mined before the lexical arm was pruned, which turned over ~80% of what that arm returns, so the checkpoint was stale by its own recorded fingerprint. Retraining on the current candidate pool changed Recall@50 not at all (0.1388), nDCG@10 by +0.0001 and MRR by +0.0007. The staleness was real and worth fixing for reproducibility; it was not, on this evidence, costing measurable accuracy.
