| arm | Recall@50 | nDCG@10 | MRR | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| `hybrid_rerank` | 0.1254 | 0.0467 | 0.0912 | 1173.2 | 1980.4 |

_n=171 queries, split=test, origin=citation, canonicalize=True._

**Weak supervision.** Labels are citation-derived, not human relevance judgements: a query is a paper's title and its positives are the papers it cites. The same signal trains the reranker, so `hybrid_rerank` rows are scored against the family of labels they learn from. Not a human-judged result.
