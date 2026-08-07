# Dev log

Running notes: what was built, what broke, and why decisions were made the way
they were. Newest session first.

---

## Session 2 — 2026-08-07

Theme of this session: **the eval harness earned its keep.** Running it end to
end for the first time exposed two bugs that had been invisible while the arms
were unexercised, one of which had a retrieval arm scoring a hard zero. Neither
was findable by reading the code — both needed real numbers on real data.

### State at end of session

| Thing | Status |
|---|---|
| Postgres 17 + pgvector under Apptainer | working (restart per allocation) |
| Corpus | 19,791 docs / 99,567 passages |
| Embedding pass | **complete** — 99,567/99,567, 0 unembedded |
| HNSW index | **built** (m=16, ef_construction=64) |
| Lexical (`fts`) arm | working — **was broken, see below** |
| Dense arm | working — **was broken, see below** |
| Hybrid (RRF) arm | working, first time exercised |
| `hybrid_rerank` arm | working, but **off-the-shelf** cross-encoder |
| Citation harvest | **done** — 8,075 pairs from 790 docs |
| Eval set | **built** — 790 queries / 8,075 qrels (171 test / 619 train) |
| Ablation table | **generated** — `docs/ablation.md` |
| Duplicate-record clustering | **done** — 1,382 docs → 1,335 clusters |
| Unit tests | **47 passing** (chunker + metrics) |
| Reranker fine-tune | not started — 619 train queries sitting unused |
| FastAPI service | not started |
| Manual judged eval set | not started — every number so far is weak supervision |

### Where to pick up — start here next session

Postgres does not survive the allocation that started it. Every session begins:

```bash
scripts/pg_start.sh     # idempotent; re-records the new hostname
regsearch stats         # expect 19,791 / 99,567 / 0 unembedded / 8,075 contexts
```

Then, in the order I would do them:

**1. Fine-tune the reranker.** This is the biggest gap. `hybrid_rerank` is
currently a public off-the-shelf checkpoint, so the ablation has no trained
component in it at all — and 619 training queries are sitting unused from the
split. Train a cross-encoder on `origin='citation' AND split='train'`, write the
checkpoint to `data/models/reranker/`, and `rerank.py` picks it up automatically
(it prefers that path and only falls back to the public model when it is
absent). **Run this on an A100 via `sbatch`, modelled on `slurm/embed.sbatch`** —
the dev node has 1 CPU and the eval's rerank arm alone took ~13 minutes there.
Needs negatives: sample hard negatives from the fused top-k that are not in the
qrels, not random documents. Then re-run the ablation and compare against the
off-the-shelf row already in `docs/ablation.md`.

**2. FastAPI service.** Not started, and it is the largest gap against the job
postings this project targets — right now there is no serving layer at all,
just a CLI. `serve` extras are already declared in `pyproject.toml`
(fastapi/uvicorn/anthropic). Wants `/search?arm=`, `/doc/{id}`, and a RAG
answer endpoint; reuse `retrieve.search.search()` directly so the arms stay
identical between eval and serving.

**3. Manual judged eval set.** `export_pool_for_judging` is written and has
never been run. TREC-style pooling over all four arms, then hand-judge —
**this needs a human, do not generate the labels.** Until it exists every
number in the repo is weak supervision and `load_eval_set` correctly refuses to
default to it.

Lower priority, tracked: real BM25 as a genuine fifth arm; `COPY` + single
`UPDATE ... FROM` for embedding writes (the ~55min pass is one UPDATE per
passage over TCP); an explicit include/exclude decision on patents and theses.

### Two bugs the eval found

**1. The dense arm never worked.** `SET LOCAL hnsw.ef_search = %s`. `SET` is a
utility statement and takes no bind parameters, so the placeholder reached the
server as a literal `$1` and every dense *and* hybrid query raised
`SyntaxError`. It survived session 1 only because there were no embeddings to
query against, so the arm was never run. Fixed with
`set_config('hnsw.ef_search', v, true)` — an ordinary function call, same
transaction-scoped semantics, so it still cannot leak across pooled connections.

**2. The lexical arm was returning nothing.** First ablation scored `fts` at
**0.0013 Recall@50**. That is not "lexical is weaker", that is broken.
`websearch_to_tsquery` ANDs its terms, and the eval queries are paper titles
averaging ~10 content words — so a passage had to contain all ten stems. On
this corpus:

```
AND matches:      1 passage   (of 99,567)
OR  matches: 32,117 passages
```

Postgres has no `or_to_tsquery`, so the fix rewrites the parse: split the
tsquery on its top-level `&`, OR the positive operands, re-AND the negated
ones. **Negation has to stay conjunctive** — folding `!x` into the OR would
match every passage that merely lacks `x`, which is nearly all of them.

This is the single most valuable thing in the repo for interview purposes: a
metric that looked like a boring "lexical loses to dense" result was actually
a bug, and only the ablation surfaced it.

### Numbers

The only table that finished and was read back is the **first, pre-fix run** —
kept here because it is the evidence for bug 2, not because it is a result:

| arm | Recall@50 | nDCG@10 | MRR | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| `fts` | 0.0013 | 0.0013 | 0.0019 | 5.7 | 20.5 |
| `dense` | 0.1221 | 0.0483 | 0.1033 | 59.3 | 146.0 |
| `hybrid` | 0.1224 | 0.0470 | 0.0943 | 45.5 | 67.9 |
| `hybrid_rerank` | 0.1262 | 0.0432 | 0.0849 | 5676.7 | 7776.8 |

_n=171, split=test, origin=citation. **Both the `fts` bug and the duplicate
inflation are present in these numbers — do not quote them.**_

> **The corrected tables were still running when this session ended.** Two
> variants were launched (`--no-canonicalize` then `--canonicalize`) to
> attribute the lexical fix and the dedup fix separately. If
> `docs/ablation.md` / `docs/ablation_nocanon.md` are missing or predate
> 2026-08-07 17:00, they did not survive; regenerate with:
>
> ```bash
> regsearch eval --split test --origin citation --no-canonicalize --out docs/ablation_nocanon.md
> regsearch eval --split test --origin citation --canonicalize    --out docs/ablation.md
> ```
>
> Budget ~15 min each on a 1-CPU node — the rerank arm dominates. On a GPU
> node it is minutes. `docs/` is deliberately **not committed yet**: the only
> file in it is the stale pre-fix table above, and a broken table sitting in
> the repo looking like a result is worse than no table.

What is already known about the corrected numbers without re-running: the
`fts` p50 will rise sharply (from 5.7 ms to roughly 700 ms) because ORing the
terms takes it from ranking 1 matched passage to ranking ~32k. That is a real
recall/latency trade and the table should show it, not hide it.

### The duplicate-record problem

A paper can enter the corpus twice — a bioRxiv preprint (`PPR`) and its
published version (`MED`) are separate Europe PMC records. **1,335 clusters,
1,382 documents, 7% of the corpus.** Composition: 1,167 MED+PPR, 90 AGR+MED,
the rest patent families and within-source repeats.

Why it corrupts the eval rather than just wasting space: qrels name one record,
retrieval frequently surfaces the other, and that scored as a **miss for
finding the right paper**. Both twins also burned separate top-k slots. In the
test split, 52 of 171 queries (30%) have at least one positive with a twin,
covering 129 of 1,747 qrels. Crucially the distortion is *uneven across arms*
— dense is likeliest to return both twins adjacently — so it biased the
arm-vs-arm comparison, which is the one thing this project exists to measure.

Fix: documents carry a `canonical_doc_id`; evaluation scores at cluster level.
Nothing is deleted — the records are real and the citation graph's foreign keys
still point at them.

**Clustering is on the normalised title, not DOI.** A preprint keeps its
`10.1101/...` DOI after publication and the journal assigns a different one, so
the twins never share a DOI. Title after stripping non-alphanumerics is what
makes them collide. Published records win the cluster; ties break on lowest
`doc_id` so two runs agree.

Checked before trusting it: every short-title cluster on this corpus turned out
to be a genuine duplicate (patent families with identical abstracts), and
sampled MED+PPR pairs were real preprint/published twins. 1,038 of 1,335
clusters have differing abstracts, which is expected — preprints get revised —
so abstract equality would have been the wrong confirmation test.

### Corpus composition (noticed, not yet acted on)

```
MED 15,256 | PPR 4,226 | AGR 169 | PAT 49 | ETH 48 | PMC 34 | CBA 5 | CTX 4
```

There are **49 patent records and 48 theses** in what is meant to be a
literature corpus. Not obviously wrong — Europe PMC indexes them — but it is a
corpus-design choice that was never made deliberately. Worth an explicit
include/exclude decision before any headline number is quoted.

### Decisions worth remembering

**Both fixes are behind boolean toggles** (`fts_or_semantics`,
`--canonicalize/--no-canonicalize`) so the pre-fix numbers stay reproducible.
An ablation that cannot reproduce its own history is not an ablation.

**The weak-supervision caveat now lives in the markdown table itself**, not in
a footer. A footer gets separated from the table the moment someone copies it
into a slide.

**Tests pin conventions, not just behaviour.** The metrics tests assert linear
vs exponential nDCG gain, IDCG truncated at *k*, recall's denominator, and
nearest-rank percentile — because a reported table is only comparable to
another if both used the same definitions. One chunker test (`fold`) was
initially passing vacuously: the naive fixture left the tail over `min_chars`
so the branch never ran. It now asserts the unfolded case first.

### Honesty items — carried forward, still binding

1. **The lexical arm is still not BM25.** It is `ts_rank_cd` with OR'd terms —
   a cover-density TF-IDF variant, no k1/b saturation. Named `fts` for this
   reason. Real BM25 remains a follow-up, and would now be a genuinely
   interesting *fifth* arm rather than a relabelling.

2. **`hybrid_rerank` is the OFF-THE-SHELF `ms-marco-MiniLM-L-6-v2`.** There is
   no fine-tuned checkpoint — `data/models/reranker` does not exist and
   `rerank.py` logs a warning saying exactly this. Every rerank number so far
   is a public baseline, not a trained model.

3. **Every number in the repo is weak supervision.** Labels are citation-
   derived: query = a paper's title, positives = the papers it cites. The same
   signal will train the reranker. `load_eval_set` still defaults to
   `origin='manual'` so this cannot leak into a headline by accident, and
   `export_pool_for_judging` still has never been run.

### Bugs hit and fixed (this session)

| Symptom | Cause | Fix |
|---|---|---|
| `SyntaxError: syntax error at or near "$1"` on every dense/hybrid query | `SET LOCAL` is a utility statement, takes no bind parameters | `set_config(..., is_local => true)` |
| `fts` Recall@50 = 0.0013 | `websearch_to_tsquery` ANDs ~10 title terms; 1 passage of 99,567 matched | rewrite parse to OR positives, keep negations conjunctive |
| All arms quietly deflated, unevenly | preprint/published twins are distinct `doc_id`s; right paper scored as a miss | `canonical_doc_id` + cluster-level scoring |
| Chunker fold test passed vacuously | fixture tail exceeded `min_chars`, branch never ran | assert the unfolded case first |
| 175KB of per-batch Slurm logs tracked in git | `.gitignore` covered `data/` but not `slurm/logs/` | untracked and ignored |

---

## Session 1 — 2026-08-06

### State at end of session

| Thing | Status |
|---|---|
| Postgres 17 + pgvector 0.8.6 under Apptainer | working |
| Corpus ingested from Europe PMC | 19,791 docs / 99,567 passages |
| Lexical (`fts`) retrieval | working, returns sensible hits |
| Embedding pass on A100 | **interrupted at ~44%** — resumable, see below |
| HNSW index | not built yet (blocked on embedding) |
| Dense / hybrid / rerank arms | code written, not yet exercised |
| Eval harness + metrics | code written, no labels yet |
| Citation weak-labelling | code written, **never run** |
| Reranker fine-tune | not started |
| FastAPI service | not started |

10 commits, all pushed to `github.com/PatrickYang2007/regsearch` (public).
~1,900 lines of first-party code.

### Where to pick up

Postgres runs on whichever node started it and records that hostname in
`data/run/pg_host`. It does **not** survive the allocation that launched it. So:

```bash
scripts/pg_start.sh          # idempotent; re-records the new hostname
regsearch stats              # confirm 19,791 docs / 99,567 passages survived
sbatch slurm/embed.sbatch    # resumes: filters on `embedding IS NULL`
```

The embedding pass is resumable by construction — it selects passages where
`embedding IS NULL`, so a killed job loses only the in-flight batch. Roughly
55k passages remained at ~30/s, so budget ~30 min of A100 time.

After embedding completes the sbatch builds the HNSW index automatically. Then
the dense and hybrid arms work and the remaining sequence is:

```bash
regsearch harvest-citations --limit 1500   # NEVER RUN YET — reranker labels
regsearch build-evalset
regsearch eval --origin citation --out docs/ablation.md
```

### Decisions worth remembering

**Postgres over TCP, not a Unix socket.** The first design used a Unix socket
in a `0700` directory — no TCP listener, no port collisions with other users on
a shared node. It was wrong. A Unix socket is *local IPC*: a Slurm job on
another node sees the socket file on `/vast` and still cannot connect to it.
Cost two failed GPU jobs to diagnose. Now: TCP, `scram-sha-256`, 32-char
generated password (gitignored, `0600`), `pg_hba` restricted to RFC1918 ranges
so it is unreachable from off-cluster. Same-node clients still use the socket.

**Bypass the container's docker-entrypoint.** `pgvector/pgvector:pg17` expects
to run as root so it can chown PGDATA and su to `postgres`. Under Apptainer we
run as our own uid, which has no entry in the container's `/etc/passwd`, so the
entrypoint aborts. Calling `initdb`/`postgres` directly works — Postgres only
requires that PGDATA be owned by the running uid.

**Rank-based fusion, not score-based.** `ts_rank_cd` and cosine similarity live
on unrelated scales; any weighted sum of raw scores is dominated by whichever
arm happens to have the wider range. RRF uses ranks, which need no per-arm
normalisation.

**Document-level metrics.** Retrieval returns passages, but scoring collapses
to first-appearance document order. Passage-level scoring would let a chunkier
arm win by returning three passages from the same paper.

**Eval split by citing document, not by pair.** Splitting by pair puts
near-duplicate contexts from one paper on both sides of the split and inflates
test scores. The split is a hash of the doc id, so it is stable across runs and
machines — two runs are comparable.

### Honesty items — do not let these slide

1. **The lexical arm is not BM25.** It is Postgres `ts_rank_cd`, a
   length-normalised TF-IDF/cover-density variant with no k1/b saturation
   terms. Named `fts` everywhere for this reason. Do not describe it as BM25
   on a résumé or in the README; a reviewer who knows IR will catch it.
   Real BM25 remains a tracked follow-up.

2. **Citation labels train the reranker; they must not also be the headline
   eval.** `load_eval_set` defaults to `origin='manual'` precisely so this
   cannot happen by accident. A table computed on `origin='citation'` is
   legitimate but must be *labelled as such* — it is weak supervision scored
   against itself, not human relevance judgement. The pooled-judging export
   (`export_pool_for_judging`) exists to produce a real manual set; that work
   has not been done.

3. **No numbers exist yet.** The README's ablation table is empty on purpose.
   Nothing should be quoted anywhere until `regsearch eval` has actually run.

### Bugs hit and fixed

| Symptom | Cause | Fix |
|---|---|---|
| `PythonFinalizationError` on exit | `psycopg_pool.__del__` joins threads during interpreter shutdown | close the pool via `atexit` |
| GPU job would have died on import | `uv` resolved Python 3.14; torch ships no 3.14 wheels | pin `requires-python = ">=3.11,<3.13"` |
| `PoolTimeout` from Python, psql fine | server on default port 5432, config said 5433; over a socket the port selects the *filename* | thread one port value through both |
| `syntax error at or near "("` | expression `md5(...)` inside a table-level `UNIQUE` constraint | unique *index* instead |
| `syntax error at or near ":"` | `psql -c` does not expand `:'var'` — only script/stdin input does | feed the `ALTER ROLE` on stdin |
| GPU job: "cannot reach Postgres" (twice) | (a) Unix socket is not cross-host; (b) the preflight helper was still hardcoded to the socket | TCP + transport auto-selection by hostname |

### Open questions for next session

- Run the citation harvest? It is the gate on both reranker training data and
  any ablation numbers. ~1,500 papers' reference lists at 6 req/s.
- Embedding writes are one `UPDATE` per passage over TCP — that is the real
  reason the pass takes ~55 min rather than a few. `COPY` into a temp table
  plus a single `UPDATE ... FROM` would cut it substantially. Worth doing if
  the corpus grows; not worth restarting the current run for.
- No tests yet. `tests/` exists and is empty. The chunker and the metrics
  module are the two things most worth unit-testing — both are pure functions
  with fiddly edge cases (abbreviation handling, nDCG normalisation).
