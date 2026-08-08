| arm | Recall@50 | nDCG@10 | MRR | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| `fts` | 0.0462 | 0.0128 | 0.0356 | 1546.5 | 3732.1 |
| `dense` | 0.1221 | 0.0483 | 0.1033 | 40.1 | 53.2 |
| `hybrid` | 0.0984 | 0.0287 | 0.0742 | 1493.2 | 3623.6 |
| `hybrid_rerank` | 0.1207 | 0.0488 | 0.0915 | 9267.0 | 14437.8 |

_n=171 queries, split=test, origin=citation, canonicalize=False._

**Weak supervision.** Labels are citation-derived, not human relevance judgements: a query is a paper's title and its positives are the papers it cites. The same signal trains the reranker, so `hybrid_rerank` rows are scored against the family of labels they learn from. Not a human-judged result.
