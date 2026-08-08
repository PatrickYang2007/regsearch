| arm | Recall@50 | nDCG@10 | MRR | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| `fts` | 0.0462 | 0.0146 | 0.0430 | 1434.4 | 3265.0 |
| `dense` | 0.1236 | 0.0536 | 0.1100 | 38.3 | 50.3 |
| `hybrid` | 0.0992 | 0.0339 | 0.0792 | 1409.1 | 3369.9 |
| `hybrid_rerank` | 0.1212 | 0.0513 | 0.0971 | 9630.2 | 15192.2 |

_n=171 queries, split=test, origin=citation, canonicalize=True._

**Weak supervision.** Labels are citation-derived, not human relevance judgements: a query is a paper's title and its positives are the papers it cites. The same signal trains the reranker, so `hybrid_rerank` rows are scored against the family of labels they learn from. Not a human-judged result.
