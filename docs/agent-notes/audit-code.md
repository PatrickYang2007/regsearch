# Correctness audit — 2026-08-08

Line-by-line review of the recently committed work, looking for latent bugs a
passing benchmark would not expose. Code state audited: `479d1cf` /
`f0f8c64` (later commits are docs/slurm only and do not touch these files).

Scope: `retrieve/search.py`, `retrieve/train_rerank.py`, `db/client.py`,
`eval/harness.py`, `eval/metrics.py`.

Repros live in the session scratchpad; every claim below was reproduced or
queried against the live database. Nothing in `src/` was modified.

---

## CONFIRMED

### 1. HIGH — `_retrieval_fingerprint()` is blind to the settings that
### determine the candidate pool, and the shipped checkpoint is already stale

`src/regsearch/retrieve/train_rerank.py:535-545`

The fingerprint exists for one stated reason (`train_rerank.py:523-526`):

> The negatives are whatever `hybrid` returned at mining time, so the
> checkpoint is only valid for this retrieval configuration. Recorded so a
> stale checkpoint is identifiable rather than silently wrong.

It records `arm, rrf_weighted, rrf_weights, rrf_k, fts_or_semantics,
bm25_topk, dense_topk`. It does **not** record `fts_prune_common_terms`,
`fts_df_max_frac`, or `fts_min_terms` — the three settings commit `479d1cf`
added, which change the lexical arm's ranked list. `479d1cf` did not touch
`train_rerank.py`.

Evidence that the checkpoint on disk is stale by exactly this criterion:

| | timestamp |
|---|---|
| `data/models/reranker/model.safetensors` | 2026-08-07 23:56 |
| commit `479d1cf` (pruning) | 2026-08-08 22:21 |
| `lexeme_df.built_at` | 2026-08-08 22:21:32 |

And the pool really did move. Comparing the two cached bench runs
(`data/fts_bench/{baseline,optimised}.json`, same 171 queries):

```
fts top-50 PASSAGE overlap pre- vs post-pruning: mean 0.265  median 0.180  min 0.000
fts top-50 DOC     overlap pre- vs post-pruning: mean 0.264  median 0.182  min 0.000
```

~80% of the lexical arm's top-50 turned over. The 4,952 hard negatives in
`training_meta.json` were mined from a fused pool that no longer exists, and
`retrieval_config` in that same file reports nothing about it.

Fix: add the three fields to `_retrieval_fingerprint()`.

### 2. MEDIUM — "Recall@50" is recall over ~35 documents, and the arms do not
### get the same number of document slots

`src/regsearch/eval/harness.py:114-131`

`search(q, arm, k=k_recall)` returns **50 passages**; the passage→document
collapse happens afterwards. So the metric is "recall among the distinct
documents that fit inside the top-50 passages". Measured on the cached runs:

| arm | mean distinct docs in top-50 passages | min |
|---|---:|---:|
| `dense` | 33.3 | 18 |
| `fts` (pre-pruning) | 41.4 | 28 |
| `fts` (pruned) | 37.4 | 21 |

`dense` is graded on ~20% fewer document slots than `fts`, in a table whose
entire purpose is arm-vs-arm comparison. It is also inconsistent with nDCG@10
in the same table, which *does* get a genuine 10 distinct documents because
dedup happens before the `[:10]` slice.

The bias runs against the arm that repeats documents most (`dense`), so the
published "dense wins" conclusion is conservative rather than inflated — but
the number is mislabelled. Only `scripts/bench_fts.py:16-17` states this;
`harness.py`, `README.md` and `docs/ablation.md` all just say "Recall@50".

### 3. MEDIUM-LOW — `split_tsquery`'s premise is false, and the eval set
### falsifies it

`src/regsearch/retrieve/search.py:78-95`

> websearch_to_tsquery only ever emits a conjunction at the top level

It does not. The English word "or" anywhere in the input becomes a tsquery
`|`. Real test query:

```
Enhancer-promoter interactions and transcription are largely maintained upon
acute loss of CTCF, cohesin, WAPL or YY1.
  -> ... & 'cohesin' & 'wapl' | 'yy1'
```

6 of 790 eval queries (1 test, 5 train) parse with a top-level `|`. Benign
for the OR rewrite itself, but the OR branch survives as one opaque operand
that `_bare_lexeme` cannot read, so no lexeme inside it can ever be pruned —
the latency fix silently does not apply there. The real consequence is #4.

### 4. MEDIUM-LOW — `assemble_tsquery` hoists a negation out of the OR branch
### it belonged to

`src/regsearch/retrieve/search.py:135-152`

```
input   : chromatin or cancer -gene
parse   : 'chromatin' | 'cancer' & !'gene'      == chromatin | (cancer & !gene)
split   : pos=["'chromatin' | 'cancer'"]  neg=["!'gene'"]
assemble: ('chromatin' | 'cancer') & !'gene'    == (chromatin | cancer) & !gene
```

`&` binds tighter than `|`, so the parse scopes `!gene` to the `cancer`
branch only. After the rewrite it applies to both, and passages containing
"chromatin" *and* "gene" are silently dropped.

Not reachable through the eval set (0 of 790 queries produce a `!`), so no
published number is affected. It **is** reachable through `regsearch search`
and the `/search` HTTP endpoint, both of which take free-form user text.

### 5. LOW — `fts_min_terms` disables pruning for short queries, far more
### often than the prose suggests

`src/regsearch/retrieve/search.py:98-132`, `config.py:134-137`

Both describe the backstop as protecting "a query made entirely of common
words". It actually fires whenever fewer than `min_terms` operands survive,
including when two perfectly good rare terms did:

```
chromatin accessibility cancer  ->  ('chromatin' | 'access' | 'cancer')
```

Nothing is pruned. On the 171-query test split the backstop fires on 14
queries (8%), and those 14 keep a corpus-common lexeme in the final tsquery.
Behaviour is defensible; the documentation undersells the trigger.

### 6. LOW (latent) — a relevance-0 qrel would be trained as a positive

`src/regsearch/retrieve/train_rerank.py:217`

`build_pairs` takes `positive_clusters(qrel_doc_ids)` from
`list(item["qrels"])`, i.e. all qrel keys. `evaluate_arm:111` filters
`rel > 0`; `build_pairs` does not. Not currently triggerable — all 8,075 rows
in `eval_qrels` have `relevance = 1` and training runs `origin='citation'` —
but `import_qrels` explicitly supports 0 ("0 irrelevant", `schema.sql:144`),
which is roadmap step 5. When manual qrels land, a document judged
*irrelevant by hand* becomes a label-1.0 training example.

### 7. LOW — twin counter compares a raw doc_id against a set of cluster ids

`src/regsearch/retrieve/train_rerank.py:175`

`if stats is not None and cand.doc_id not in positives` — `positives` holds
canonical ids. When the qrel names the preprint and the candidate is the
published twin, the block happens but is not counted. Stats-only; the
exclusion itself is correct.

### 8. LOW — `queries_without_negatives` also counts queries with no positive

`src/regsearch/retrieve/train_rerank.py:420`, `build_pairs:229` and `:267`

`build_pairs` returns `[]` on two different paths; the caller attributes both
to "no negatives".

### 9. LOW — dangling evidence reference

`config.py:130` cites `docs/agent-notes/fts-latency.md` for the sweep that
chose `fts_df_max_frac=0.05` / `fts_min_terms=3`. The file is neither on disk
nor in git.

### 10. LOW — the shipped `w_fts=0.5` was chosen on a lexical arm that no
### longer exists

`config.py:83-88`, `README.md:124`

`data/rrf_tuning_runs.json` is dated 2026-08-07 22:57; pruning landed
2026-08-08 22:21 and turned over ~80% of the fts top-50 (see #1). The sweep
table that justifies shipping 0.5 describes a fusion input that has since
changed.

### 11. INFO — the newest and most intricate code has no unit tests

`build_fts_tsquery`, `split_tsquery`, `prune_common_terms`,
`assemble_tsquery`, `rrf_fuse` and `evaluate_arm` appear nowhere in `tests/`.
95 tests pass and none touches this path — despite `search.py:50-52`
justifying the Python-side design partly on "it is unit-testable without a
database". Finding #4 is a one-line unit test.

---

## CHECKED AND CORRECT

* **Join-after-limit is genuinely result-identical.** `passages_doc_id_fkey
  FOREIGN KEY (doc_id) REFERENCES documents(doc_id)` exists in the live
  database, `passages.doc_id` is `attnotnull = t`, `documents.doc_id` is the
  primary key (so the inner join can neither drop nor duplicate a row), and
  there are 0 orphan passages. Ran `_FTS_SEARCH_JOIN_LAST` and
  `_FTS_SEARCH_JOIN_FIRST` over 8 real test queries at k=100: identical
  passage_id lists, identical scores to 12 dp, identical titles. Both use the
  same deterministic `ORDER BY score DESC, passage_id`.
* **`prune_common_terms` restores the RAREST dropped terms, as claimed.**
  `drop_idx.sort(key=ndoc)` is ascending, so `drop_idx[:n_restore]` is the
  lowest-df subset. Verified against real corpus frequencies: restores
  `chromatin` (24,797) and `cell` (36,560) before `express` / `gene`.
* **Degenerate tsquery inputs are all handled.** Empty, whitespace-only,
  all-stop-word, negation-only (`-cancer`) and negated-phrase-only
  (`-"gene expression"`) each yield `''` and return `[]` without touching the
  database. Single-term, phrase (`<->`), hyphenated-compound and distance
  (`<2>`) operands rebuild to valid tsqueries with correct precedence — `<->`
  and `<N>` bind tighter than `|`, so phrase groups survive the OR rewrite
  intact. 0 cast failures across all 171 test queries plus a 16-input
  adversarial battery.
* **No SQL injection.** The tsquery reaches Postgres as a bound parameter
  (`%(tq)s::tsquery`); `'; DROP TABLE passages; --` parses to
  `('drop' | 'tabl' | 'passag')` and returns rows. The only f-string SQL is
  `_NORM_TITLE` (a module constant) and `int(m)` / `int(ef_construction)` in
  `build_vector_index`.
* **`rrf_fuse`'s zero-weight skip is consistent on all three paths.**
  `weights=None` with `rrf_weighted=False` gives `{}`, so every arm defaults
  to 1.0 and nothing is skipped; an explicit `weights={}` behaves identically;
  only an explicit `0.0` skips. `scripts/tune_rrf.py:93-97` mirrors it, which
  is why the sweep's `w_fts=0.00` row (0.1236) equals the standalone `dense`
  row to 4 dp.
* **Negative mining compares clusters on both sides.** Positives via
  `positive_clusters()`, candidates via `to_canonical(cand.doc_id)`, and
  `exclude_docs` via `to_canonical` into `blocked` — all three at cluster
  level (`train_rerank.py:166-184`). A positive cannot re-enter as a negative
  through a second passage: the `seen` dedup is also keyed on cluster.
* **`load_citing_docs`'s docstring claim holds.** All 619 train and all 171
  test citation queries resolve to a citing document; 0 unresolved.
* **No train/test leakage from `build_citation_evalset`.** `eval_queries`
  has `UNIQUE (query_text)` and the insert does `ON CONFLICT DO UPDATE SET
  split`, so one `context_text` emitted by two citing documents would merge
  their qrels and take whichever split was written last. There are 0 such
  texts in `citation_contexts`.
* **`rebuild_canonical_docs` is correct.** The `i` flag on
  `regexp_replace(title,'[^a-z0-9]','','gi')` makes the negated class
  case-insensitive, so uppercase letters survive to `lower()`. The blanket
  `SET canonical_doc_id = NULL` before re-clustering prevents stale pointers.
  `ORDER BY (source='PPR'), doc_id` is deterministic and prefers the published
  record. `load_canonical_map` omitting self-mappings matches both consumers'
  `.get(d, d)` convention.
* **`rebuild_lexeme_df` is correct.** `ts_stat`'s `ndoc` is per-tsvector
  document frequency, and `n_passages` is read inside the same transaction as
  the `TRUNCATE` + `INSERT`, so the denominator matches the corpus that was
  scanned. 102,165 rows; 134 above 5%.
* **`metrics.py` is correct on every convention it claims.** Recall
  denominator is `|relevant|` not `k`; RR is 1-based; DCG uses `log2(i+1)`
  with `i` 1-based so rank 1 is undiscounted; nDCG's ideal is built from the
  *collapsed* qrels and capped at `k`; `percentile` is nearest-rank with no
  index underflow at p=0. The 23 tests in `tests/test_metrics.py` pin each of
  these and none is vacuous.
* **Qrels collapsing in `evaluate_arm:107-111` takes `max` per cluster**, and
  the same collapsed dict feeds nDCG's ideal — so merging twins cannot inflate
  the idcg denominator.
