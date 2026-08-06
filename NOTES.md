# Dev log

Running notes: what was built, what broke, and why decisions were made the way
they were. Newest session first.

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
