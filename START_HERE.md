# Start here

Plain-language orientation. Read this first if you've been away from the
project and want to know where it stands and what to do next.

- **This file** = what's going on, what to do.
- [NOTES.md](NOTES.md) = the detailed dev log (dense; written for future-you mid-task).
- [README.md](README.md) = the public-facing description for people who find the repo.

Last updated: **2026-08-10**, end of session 5.

---

## 1. What this project is

A search engine over regulatory-genomics papers. You give it a query, it returns
ranked passages from the literature.

The point is **not** the search demo. The point is the **comparison table**: four
different retrieval strategies, all measured on the same query set, so you can
say which one actually works rather than asserting it. That table is the thing
worth showing an interviewer.

The four strategies ("arms"):

| arm | what it does |
|---|---|
| `fts` | keyword search (Postgres full-text) |
| `dense` | vector search (embeddings + HNSW index) |
| `hybrid` | fuses the two above with reciprocal rank fusion (now weighted) |
| `hybrid_rerank` | takes the fused results and re-scores them with a cross-encoder |

---

## 2. Where things stand

**The whole pipeline works end to end**, and as of session 3 there is also an
HTTP service in front of it. You can ingest, embed, search on all four arms,
serve them over HTTP, and produce the comparison table.

| | status |
|---|---|
| Corpus (19,791 papers / 99,567 passages) | loaded |
| Embeddings | complete, all 99,567 |
| Vector index (HNSW) | built |
| All four retrieval arms | working |
| Fusion weighting | **done** — swept and shipped at `w_fts=0.5` (§3) |
| Training labels (from citations) | 8,075 pairs harvested |
| Eval set | 790 queries (171 test / 619 train) |
| Ablation table | current → [docs/ablation.md](docs/ablation.md), all four arms |
| Unit tests | 133, passing |
| Web API (FastAPI) | **works**, wired to `regsearch serve` |
| **Trained reranker** | **done** — fine-tuned, +41% MRR over off-the-shelf |
| Lexical arm latency | **fixed** — 8.7× faster and better quality |
| GPU actually usable | **yes, as of session 5** — cu126 pin, `device: cuda` in training_meta |
| Reranker retrained on current pool | **done** — no longer stale by its own fingerprint |
| tsquery OR/negation bugs | **fixed** — real parser, 26 tests |
| Recall@50 mislabelling | **documented** — metric unchanged, caveat now stated everywhere |
| RRF sweep | **re-run** post-pruning; now agrees with the shipped `hybrid` row |
| Human-judged eval set | **not done — this is yours, see [JUDGING.md](JUDGING.md)** |

**Postgres may or may not be running** — it dies with whatever Slurm allocation
started it. `scripts/pg_start.sh` is idempotent; run it and move on (§4).

**The one thing to verify before trusting any rerank number:**

```bash
ls data/models/reranker/config.json    # must exist
ls -d data/models/reranker_trained     # must NOT exist
```

If `reranker_trained` is present, an A/B comparison was interrupted before
restoring the checkpoint, and `hybrid_rerank` is silently serving the
off-the-shelf model. Fix with `mv data/models/reranker_trained data/models/reranker`.

`data/models/reranker.backup-2026-08-10/` is the *previous* checkpoint, kept
when session 5 retrained. It is inert — nothing loads it — and it is safe to
delete once you trust the current one. It is deliberately not named
`reranker_trained`, to stay clear of the trap above.

---

## 3. The results, and what they mean

| arm | Recall@50 | nDCG@10 | MRR | p50 ms |
|---|---:|---:|---:|---:|
| `fts` | 0.0501 | 0.0172 | 0.0448 | 152.6 |
| `dense` | 0.1236 | **0.0536** | 0.1100 | **12.3** |
| `hybrid` | 0.1284 | 0.0500 | 0.1010 | 160.6 |
| `hybrid_rerank` | **0.1388** | 0.0523 | **0.1294** | 254.4 |

**Plain reading: the full pipeline wins on finding things, and plain `dense`
still wins on ordering the top handful — while being ~21x faster.** Reranking
pulls more correct papers into the top 50 and gets the first correct one higher,
but hasn't overtaken dense on the graded top-10 measure.

Read `Recall@50` with the caveat in [docs/ablation.md](docs/ablation.md): it is
recall among the distinct documents that fit inside 50 *passages*, and the arms
get unequal document slots (`dense` 33.3, `fts` 37.4). The bias runs against
`dense`, so its win understates rather than flatters.

In session 3 both fused arms *lost* to plain dense. Two changes flipped that:
the reranker got fine-tuned, and the keyword arm stopped being nearly useless.

### What the fine-tune actually bought

Three things changed at once, so that table can't tell you which one helped. So
the checkpoint was moved aside and `hybrid_rerank` re-run with everything else
identical — same queries, same fusion weights, same keyword config:

| `hybrid_rerank` | Recall@50 | nDCG@10 | MRR |
|---|---:|---:|---:|
| off-the-shelf model | 0.1254 | 0.0467 | 0.0912 |
| **your fine-tuned model** | **0.1388** | **0.0522** | **0.1287** |
| | +10.7% | +11.8% | **+41.1%** |

That is the number worth quoting, because it is the only one that isolates the
fine-tune from everything bundled with it.

**That limitation is now closed, and the answer was "it barely mattered".** Those
negatives had been mined *before* the keyword arm was optimised, a change that
turned over ~80% of what that arm returns, so the checkpoint was stale by its own
recorded fingerprint. Session 5 retrained on the current pool. Recall@50 did not
move at all (0.1388), nDCG@10 went +0.0001 and MRR +0.0007. Worth fixing for
reproducibility — the fingerprint now matches the retriever — but it was not
costing measurable accuracy, and saying so is more useful than implying a gain
nobody measured.

**A sharper limitation, found in session 5:** the fine-tuned model is a *pool
discriminator*, not a relevance model. Ask it to score an obviously off-topic
passage and it will happily rank a cake recipe above a passage restating the
query, because every negative it ever trained on was a hard negative mined from
the top-50 and was therefore already on-topic. This does not touch any number in
the table — `hybrid_rerank` only ever reranks the fused top-100, where the model
is in-distribution — but it means the scores must never be thresholded, which
directly constrains the proposed `/answer` endpoint. Full write-up and evidence
in [docs/agent-notes/reranker-ood.md](docs/agent-notes/reranker-ood.md).

### Two things you must not claim about these numbers

1. **These are not human relevance judgements.** The labels come from citations:
   a query is a paper's title, and the "correct answers" are the papers it
   cites. That is weak supervision, and the reranker *trains on the same
   signal*, so it is being graded against the family of labels it learned from.
   Say so.
2. **The keyword arm is not BM25.** It is Postgres `ts_rank_cd`. Related, but a
   reviewer who knows retrieval will catch it. It is named `fts` everywhere for
   this reason.

Also worth knowing: latency columns are **not** comparable across sessions —
this table ran on a GPU node with 2 CPUs, the previous one on 8 CPUs, the one
before that on 1. `hybrid_rerank` p50 fell 1185 → 254 ms purely because the
cross-encoder moved to the GPU. The `fts` speedup is the one measured on the
same node with the same harness, so it is the real one.

## 4. Getting it running

### Every single session, first thing

```bash
cd /vast/palmer/pi/garg/Patrick/regsearch
source .venv/bin/activate    
scripts/pg_start.sh
regsearch stats
```

**`regsearch` is not a system command.** It's installed inside this project's
virtualenv, so a fresh terminal has never heard of it. Either activate the venv
as above, or skip activation and spell out the path — `.venv/bin/regsearch
stats` — or use `uv run regsearch stats`. All three are equivalent; activating
once per terminal is the least typing.

**Why you have to do this every time:** Postgres here isn't a background service
like on a laptop. You have no root on the cluster, so it runs as an ordinary
process inside a container, owned by whatever Slurm job started it. When that job
ends, the process is killed with it. Your *data* is safe on `/vast` (~850 MB in
`data/pgdata`) — it's only the server that goes away. `pg_start.sh` restarts it
and records the new node name so clients can find it.

`regsearch stats` should print exactly:

```
documents         19,791
passages          99,567
unembedded             0     <- must be 0
citation contexts  8,075
eval queries         790
eval qrels         8,075
```

If any number is different, stop and find out why before running anything else.

### Model weights: nothing to do

`HF_HOME` is set automatically in `config.py`, so model weights land in
`data/hf` on `/vast` rather than eating your home quota. You do **not** need to
export anything. Both models are already cached there, so they load offline.

### Try a search

```bash
regsearch search "enhancer promoter interaction chromatin looping" --arm dense
```

### Try the web API

```bash
uv run --extra serve uvicorn regsearch.api.app:app --port 8000
curl 'localhost:8000/health'
curl 'localhost:8000/search?q=enhancer+promoter+looping&arm=dense&k=5'
```

Or just `regsearch serve` — it wraps the same thing and defaults to loopback.

It binds to loopback on purpose. On a shared node, a search service on
`0.0.0.0` is reachable by every other person logged into that machine. Use an
SSH port-forward if you want it from your laptop.

**Don't time the first request.** The embedding model loads lazily *inside* it,
so the first `dense` query reports ~12 seconds and every one after it reports
~50 ms. The reported `latency_ms` includes that load. Warm numbers only.

### Re-run the comparison table

```bash
regsearch eval --split test --origin citation --out docs/ablation.md
```

Takes ~15 min on a 1-CPU node (the cross-encoder arm dominates), minutes on a GPU
node.

---

## 5. What to do next — pick up here

**Everything mechanical on the old list is done.** Sessions 5 closed items 1, 2,
4 and 5 below. What is left is the one thing that needs a human.

### 1. Hand-judge the pool — *only you can do this, and it now gates everything*

`data/judging_pool.csv`, 704 candidates over 40 realistic queries, ~2 hours.
Instructions in **[JUDGING.md](JUDGING.md)**.

Until this exists, **every number in this repo is weak supervision** — the
reranker is graded against the same signal family it trained on. This is the
single thing standing between the project and a table that needs no asterisk,
and it is now the only roadmap item that changes what the project *is* rather
than how tidy it is.

```bash
regsearch import-qrels data/judging_pool.csv
regsearch eval --origin manual --out docs/ablation_manual.md
```

One thing to know before you start: `build_pairs` now filters `relevance > 0`,
so a document you judge **irrelevant** is correctly excluded from training
positives. Before session 5 it would have been trained as a label-1.0 positive
*and* blocked from the negative pool. Judge 0s freely; they are handled.

### 2. Decide about patents and theses

49 patents and 48 theses are in the corpus because Europe PMC indexes them.
Nobody decided that. It is a one-line call, and it is worth making before you
quote a headline number in an interview — a free-form `/search` for
"chromatin or cancer" currently returns a patent as its top hit.

### Lower priority

- **Real BM25** as a genuine fifth arm (`fts` is `ts_rank_cd`, and always will
  be — this would be a new arm, not a relabel).
- **An `/answer` endpoint**: retrieve, then have Claude write a cited answer.
  `anthropic` is already in the `serve` extra. **Read
  [docs/agent-notes/reranker-ood.md](docs/agent-notes/reranker-ood.md) first** —
  the obvious design ("only cite passages the reranker scores above X") is
  exactly what this reranker cannot support.
- **Quantify the reranker's out-of-distribution behaviour properly.** The
  session-5 evidence is 1 query and 8 pairs: enough to establish direction, not
  magnitude.
- **Consider renaming the `Recall@50` column** to something that says "passages"
  — the caveat is now documented everywhere, but a rename touching every
  published table should be one deliberate pass, ideally decided together with
  whether to switch to a fixed document count instead.

### Done in session 5, for the record

1. **GPU usable.** `pyproject` pins the cu126 torch build; verified in the
   project venv on a real A5000, and `training_meta.json` now records
   `"device": "cuda"`. The 6.2 GB probe venv is deleted, its logs preserved in
   `docs/agent-notes/evidence/`. `slurm/eval.sbatch` is a GPU job: the full
   ablation went from ~15 min to **2m29s**, the fine-tune from 17 min to
   **2m52s**.
2. **Reranker retrained** on the current candidate pool. Quality barely moved
   (§3) — the honest result, not the hoped-for one.
3. **tsquery parser** replaces the `' & '` string split, fixing pruning inside
   OR branches and negation scope. 6 of 790 queries change, exactly the 6 with a
   top-level `|`.
4. **Recall@50 documented** for what it measures; `config.py`'s dangling
   citation to a sweep that never happened is corrected.
5. **RRF sweep re-run** post-pruning. It now agrees with the shipped `hybrid`
   row, which the old sweep contradicted. `w_fts=0.5` still stands: fusion loses
   to `dense` on nDCG@10 at every weight, but `hybrid` exists to hand the
   reranker a deeper pool, and 0.5 maximises Recall@50 — which is the pool the
   reranker then reorders.
6. **Mining fixes** so hand-judged 0s cannot poison training (see item 1).

## 6. Things that will trip you up

**Postgres dies with your allocation.** Covered above. It's the single most
common "why is nothing working" cause. Run `scripts/pg_start.sh`.

**`regsearch: command not found`** means you haven't activated the virtualenv in
this terminal. `source .venv/bin/activate`. It is not installed system-wide.

**Start Postgres before submitting a GPU job**, not after. Both sbatch jobs check
the connection on startup and exit immediately if they can't reach the database.
(A GPU job talks to Postgres over **TCP**, never the Unix socket — a socket is
local to one machine and can't be used across nodes even on shared storage. That
misunderstanding cost two failed jobs in session 1. `slurm/embed.sbatch` used to
carry a comment claiming the opposite; it was corrected in session 3.)

**`uv sync` will remove pytest** *and* the serve extra, and it will drop you back
to whatever `uv.lock` says. Sync with everything you want to keep:
`uv sync --extra embed --extra dev --extra serve`. The 133 tests need `dev`.

**A checkpoint appearing in `data/models/reranker/` silently changes what
`hybrid_rerank` means.** `rerank.py` prefers that directory and falls back to the
public model only when it's absent — nothing announces the switch. So an A/B
comparison that moves the checkpoint aside must move it back, and a smoke-test
run must never write there. Check §2 before trusting any rerank number.

**Check `"device"` in `training_meta.json` anyway.** The cu126 pin fixed
`torch.cuda.is_available()`, and it is `True` as of session 5 — but the failure
mode it fixed was *silent*: torch swallowed the CUDA init error as a warning and
trained on CPU while holding an idle A5000. If a `uv sync` ever resolves torch
from PyPI's default index again, that returns without announcing itself. The
metadata field is the cheap check.

**The reranker's scores are not calibrated relevance.** It ranks well inside the
candidate pool and nonsensically outside it — see §3 and
[docs/agent-notes/reranker-ood.md](docs/agent-notes/reranker-ood.md). Never
threshold on them.

**The corpus contains 49 patents and 48 theses.** Europe PMC indexes them, so
they came along with everything else. Not necessarily wrong, but nobody ever
decided to include them. Worth an explicit call before quoting a headline number.

---

## 7. Current git state

Committed and pushed to `github.com/PatrickYang2007/regsearch` (public).
Session 5 added, newest first:

```
Refresh the ablation on GPU with the retrained reranker
Record that the fine-tuned reranker is a pool discriminator
Say what "Recall@50" actually counts
Stop a hand-judged irrelevant document training as a positive
Parse the tsquery instead of splitting it on ' & '
Pin the cu126 torch build so the GPU is actually used
```

The reranker commit is the one to re-read before an interview: it is the only
finding here that changes how the headline result should be *described*, and it
came out of a verification job whose assertion failed for a reason that had
nothing to do with what it was verifying.

Session 4, for context:

```
Bring README in line with what the code actually does
Record lexical pruning settings in the training fingerprint
Correct published OpenAPI text that no longer matches reality
Actually chmod the socket directory that trust auth depends on
Record the torch/CUDA diagnosis and the verified fix
Measure the fine-tuned reranker against the off-the-shelf one
Run the ablation as a batch job; fix scripts/start.sh
Add judging guide for the manual eval set
Wire serve, train-reranker, and the judging workflow into the CLI
Prune corpus-wide terms from the lexical query: 8.7x faster and better
```

Two of those are worth knowing about specifically: the socket-permission
commit is a **security fix** (trust auth was relying on a directory mode the
script never actually set), and the fingerprint commit records that the
**shipped checkpoint is stale by its own criterion** — see §5.2.

`4f38ce9` is the one worth re-reading — its commit message carries the full
weight sweep and the argument for shipping a setting that scores worse on two of
three metrics.

Session 2, for context:

```
c5fe153  Add first ablation results; dense beats both hybrid arms
697071b  Log session 2: two eval bugs, duplicate clustering, resume point
774bcd2  Score duplicate records as one document in eval
2ce82a3  Fix lexical arm: OR query terms instead of ANDing them
ad330ba  Unit-test the chunker and the ranking metrics
62b5e97  Fix dense arm: SET LOCAL takes no bind parameters
```

The two "Fix" commits are the interesting ones: both arms had been written in
session 1 but never actually run, so the bugs sat invisible until there were
embeddings to query against and an eval to expose them.
