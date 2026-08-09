# Documentation audit — 2026-08-08

Scope: `README.md`, `START_HERE.md`, `NOTES.md`, `JUDGING.md`, docstrings/comments
in `src/regsearch/api/`. Cross-checked against the live DB, `data/models/reranker/`,
`slurm/logs/*`, `data/fts_bench/*`, `data/tune_rrf.log`, and the code.

Read-only audit. Nothing was edited, committed, or pushed.

## Headline

Two events landed on 2026-08-08 and no narrative doc was updated for either:

1. **The cross-encoder fine-tune completed.** Slurm job 8740006, `COMPLETED`,
   39:48, exit `0:0`. Checkpoint in `data/models/reranker/`. `rerank.model_path()`
   resolves to it right now (verified by execution). Every doc that says "no
   trained model exists" is false, including the **published OpenAPI description**.
2. **The ablation was re-run on all four arms** (job 8867761) and the result is
   **committed at HEAD** in `docs/ablation.md` — while `README.md` and
   `START_HERE.md` still print the superseded table. The public repo contradicts
   itself.

Consequence: `hybrid_rerank` now beats `dense` on Recall@50 and MRR, so the
repo's headline finding ("plain dense wins every metric") is no longer true.

## Timeline reconstructed from logs

| time (2026-08-08) | event |
|---|---|
| 22:22 | `data/judging_pool.csv` generated (704 rows / 40 queries) |
| 22:23–22:28 | job 8867761: 4-arm ablation with **fine-tuned** reranker → `docs/ablation.md` |
| ~22:29 | checkpoint moved `reranker` → `reranker_trained` (A/B setup, START_HERE §3a) |
| 22:31–22:35 | job 8868037: `hybrid_rerank` only, **off-the-shelf** → `docs/ablation_offtheshelf.md` |
| 22:35 | checkpoint restored `reranker_trained` → `reranker` |

The restore completed correctly. `model_path()` returns the fine-tuned directory.
This is not a fault — noted only because the audit observed it mid-flight.

## Current vs. published numbers

`docs/ablation.md` (HEAD, job 8867761) vs README/START_HERE tables:

| arm | R@50 now | R@50 doc | nDCG@10 now | nDCG@10 doc | MRR now | MRR doc |
|---|---|---|---|---|---|---|
| fts | 0.0501 | 0.0462 | 0.0172 | 0.0146 | 0.0448 | 0.0430 |
| dense | 0.1236 | 0.1236 | 0.0536 | 0.0536 | 0.1100 | 0.1100 |
| hybrid | 0.1284 | 0.0992 | 0.0500 | 0.0339 | 0.1010 | 0.0792 |
| hybrid_rerank | 0.1388 | 0.1212 | 0.0522 | 0.0513 | 0.1287 | 0.0971 |

Fine-tune vs off-the-shelf, otherwise identical settings:

| reranker | R@50 | nDCG@10 | MRR |
|---|---|---|---|
| fine-tuned | 0.1388 | 0.0522 | 0.1287 |
| off-the-shelf | 0.1254 | 0.0467 | 0.0912 |

The fine-tune wins on all three. That answers the open test in START_HERE §3 and
supports keeping `w_fts=0.5`. Not yet stated in any narrative doc.

## Latency is not comparable across the two tables

`slurm/eval.sbatch` sets `--cpus-per-task=8` (log: `OMP_NUM_THREADS=8`). The old
table was measured on the 1-CPU dev node. Nothing in `docs/ablation.md` records
the CPU count, so the `dense` 38.3→20.5 ms and `hybrid_rerank` 9630→1185 ms
improvements conflate code changes with an 8x thread increase.

The `fts` 1434→174 ms improvement is independently corroborated by
`data/fts_bench/*` (p50 1531.2 → 175.3, **8.73x**, same harness, same node), so
that one is a real code win.

## Verified correct (recomputed, not trusted)

- Corpus: documents 19,791 / passages 99,567 / unembedded 0 / citation contexts
  8,075 / eval queries 790 / eval qrels 8,075. `regsearch stats` matches
  START_HERE's quoted output exactly.
- Splits: 171 test / 619 train, all `origin='citation'`; zero `manual` rows.
- Judging pool: 704 candidates, 40 queries, `fts` uniquely contributed 259,
  mean 17.6 candidates/query, all `relevance` blank. Pooling is
  fts+dense+hybrid top-10 deduped by doc_id (`eval/build.py:92`).
- Dedup: 18,409 self-referential / 1,382 point elsewhere / 0 NULL; 1,335 clusters
  of size>1 (= 6.98% of corpus). Matches `api/models.py` and NOTES.
- RRF sweep: all nine rows in `data/tune_rrf.log` match `config.py`, README and
  START_HERE exactly. No weight beats dense alone on nDCG@10 or MRR.
- Lexeme df: gene 47.8%, express 40.1%, chromatin 24.9% — exact.
- Corpus composition: PAT 49, ETH 48 — exact.
- Tests: 95 collected (chunk 24 + metrics 23 = 47, api 27, train_rerank 21).
- qrels/query: mean 10.2, min 3, max 119 — exact.
- Postgres: `listen_addresses='*'`, scram-sha-256 on RFC1918 only, loopback TCP
  rejected, hostname recorded in `data/run/pg_host`.
- All sbatch TCP-vs-socket comments are now correct (the old `embed.sbatch` drift
  is genuinely fixed).
- `ts_rank_cd` is never called BM25 in any doc or in the OpenAPI text.
- Every ablation table carries the weak-supervision disclaimer.

## Falsified

See the ranked report. Principal items:

- Fine-tune "never run" — README:14-18, 103-104, 139-141; START_HERE:53-54, 60-62,
  131-133; NOTES:364-366; `api/app.py:48-49` (published); `docs/ablation_nocanon.md`.
- "No replacement numbers measured" — README:87-92, START_HERE:75-80.
- "Plain dense wins every metric" — README:109, START_HERE:82.
- "`0700` directory" justifying `--auth-local=trust` — README:172,
  `scripts/pg_start.sh:71`. Actual mode of `data/run` is **2755**. Protection comes
  from `/vast/palmer/pi/garg` being 2770 (group `garg`), so any garg-group member
  on that node can connect password-free.
- "no `regsearch serve` yet" — START_HERE:52, 265-266, 328-333; `api/app.py:147`;
  NOTES:160-164. The command exists at `cli.py:334`.
- `source: MED | PMC | PPR` — `api/models.py:104` (published). Eight sources exist.
- `docs/agent-notes/fts-latency.md` referenced by `config.py:130` does not exist.
- OR-match size: README says ~32k, `config.py:114`/`schema.sql:105` say ~52k.
  Measured over 12 test queries: unpruned median 39,846; the shipped pruned path
  matches median 8,560.
