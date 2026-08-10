| arm | Recall@50 | nDCG@10 | MRR | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| `fts` | 0.0501 | 0.0172 | 0.0448 | 174.0 | 609.4 |
| `dense` | 0.1236 | 0.0536 | 0.1100 | 20.5 | 29.7 |
| `hybrid` | 0.1284 | 0.0500 | 0.1010 | 187.6 | 564.5 |
| `hybrid_rerank` | 0.1388 | 0.0522 | 0.1287 | 1185.2 | 1907.0 |

_n=171 queries, split=test, origin=citation, canonicalize=True._

**Weak supervision.** Labels are citation-derived, not human relevance judgements: a query is a paper's title and its positives are the papers it cites. The same signal trains the reranker, so `hybrid_rerank` rows are scored against the family of labels they learn from. Not a human-judged result.

**Recall@50 is over 50 passages, not 50 documents.** Each arm is asked for 50 passages and those are collapsed to their parent documents afterwards, so the column is recall among the distinct documents that fit inside a top-50 passage list. The arms consequently get unequal numbers of document slots: over these 171 queries `dense` averages 33.3 distinct documents (min 18) and `fts` 37.4 (min 21), so `dense` is graded on ~11% fewer slots than the arm printed beside it. The bias runs against whichever arm repeats documents most, which is `dense` — the arm that wins nDCG@10 — so the comparison understates it rather than flattering it. nDCG@10 is unaffected: the collapse happens before the top 10 is taken, so that column is 10 genuine documents for every arm.
