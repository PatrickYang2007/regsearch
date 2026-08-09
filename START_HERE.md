# Start here

Plain-language orientation. Read this first if you've been away from the
project and want to know where it stands and what to do next.

- **This file** = what's going on, what to do.
- [NOTES.md](NOTES.md) = the detailed dev log (dense; written for future-you mid-task).
- [README.md](README.md) = the public-facing description for people who find the repo.

Last updated: **2026-08-08**, end of session 4.

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
| Unit tests | 95, passing |
| Web API (FastAPI) | **works**, wired to `regsearch serve` |
| **Trained reranker** | **done** — fine-tuned, +41% MRR over off-the-shelf |
| Lexical arm latency | **fixed** — 8.7× faster and better quality |
| GPU actually usable | **no** — wrong torch wheel; fix verified, not applied |
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

---

## 3. The results, and what they mean

| arm | Recall@50 | nDCG@10 | MRR | p50 ms |
|---|---:|---:|---:|---:|
| `fts` | 0.0501 | 0.0172 | 0.0448 | 174.0 |
| `dense` | 0.1236 | **0.0536** | 0.1100 | **20.5** |
| `hybrid` | 0.1284 | 0.0500 | 0.1010 | 187.6 |
| `hybrid_rerank` | **0.1388** | 0.0522 | **0.1287** | 1185.2 |

**Plain reading: the full pipeline now wins on finding things, and plain `dense`
still wins on ordering the top handful — while being ~58x faster.** Reranking
pulls more correct papers into the top 50 and gets the first correct one higher,
but hasn't overtaken dense on the graded top-10 measure.

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

**Known limitation, state it if asked:** those training negatives were mined
*before* the keyword arm was optimised, and that change turned over ~80% of what
the keyword arm returns. So the model is trained against a candidate pool it no
longer sees. The comparison above is still valid — both rows ran under today's
configuration — but retraining would likely do better.

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
this table ran on 8 cores, earlier ones on 1. The `fts` speedup is the one
measured on the same node with the same harness, so it is the real one.

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

Ordered. The first two are small and mechanical; #3 is yours and gates
everything else.

### 1. Make the GPU actually usable — *10 minutes, blocks all future training*

Last session's fine-tune allocated an A5000 and **trained on CPU**. The wheel is
wrong: `torch 2.13.0+cu130` needs a CUDA 13 driver, the cluster has 12.8. This
is not a cluster problem — the driver is current — it is a packaging default
(PyPI ships the cu130 build for linux, and `pyproject` pins only `torch>=2.3`).

The fix is verified on a real GPU node (`torch.cuda.is_available() -> True`,
same torch version, no other dependency moves). Append to `pyproject.toml`:

```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cu126" }]

[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true
```

Then `uv lock && uv sync --extra embed`. `explicit = true` is load-bearing —
without it, everything else would also resolve from the PyTorch index.

Full evidence in `docs/agent-notes/torch-cuda.md`. **Afterwards, delete the
6.3 GB probe venv:** `rm -rf .uv_cache/torchprobe`.

### 2. Retrain the reranker on the current candidate pool

The shipped checkpoint's negatives were mined *before* lexical term pruning, and
pruning turned over ~80% of what the keyword arm returns. The model is trained
against a pool it no longer sees. It still beat the off-the-shelf baseline by
41% MRR, so this is an improvement opportunity, not a defect.

With #1 done this is fast on a real GPU:

```bash
scripts/pg_start.sh                       # must be up BEFORE sbatch
sbatch slurm/finetune_rerank.sbatch
# then re-measure:
sbatch slurm/eval.sbatch
```

`_retrieval_fingerprint()` now records the pruning settings, so the new
checkpoint's `training_meta.json` will show whether it matches the retriever.

### 3. Hand-judge the pool — *only you can do this*

`data/judging_pool.csv`, 704 candidates over 40 realistic queries, ~2 hours.
Instructions in **[JUDGING.md](JUDGING.md)**.

Until this exists, **every number in this repo is weak supervision** — the
reranker is graded against the same signal family it trained on. This is the
single thing standing between the project and a table that needs no asterisk.

```bash
regsearch import-qrels data/judging_pool.csv
regsearch eval --origin manual --out docs/ablation_manual.md
```

### 4. Fix what the code audit found (`docs/agent-notes/audit-code.md`)

None of these invalidate a published number; all are real:

- **"Recall@50" is recall over ~35 documents, and arms get unequal slots.**
  `search(k=50)` returns 50 *passages*, collapsed to documents afterward —
  `dense` averages 33.3 distinct docs, `fts` 41.4. Document it in the harness
  and README, or retrieve deeper and truncate to a fixed document count.
- **`split_tsquery` assumes a top-level AND**, but the word "or" makes
  Postgres emit a `|`. 6 of 790 eval queries hit it; pruning silently skips
  the OR branch.
- **`assemble_tsquery` hoists negations out of OR branches** —
  `chromatin or cancer -gene` excludes `gene` from both sides. Not reachable
  from the eval set, but reachable from `/search` free-form input.
- **The tsquery builders and `rrf_fuse` have zero tests.** 95 tests pass and
  none touch that path — which is how the above survived.

### 5. Re-run the RRF sweep

`scripts/tune_rrf.py`'s table predates term pruning, so it describes fusing a
weaker keyword arm than the one that ships. Its `w_fts=0.5` row disagrees with
the current `hybrid` row for the same setting. Cheap — it caches retrieval and
re-fuses offline.

### Lower priority

- **Real BM25** as a genuine fifth arm (`fts` is `ts_rank_cd`, and always will
  be — this would be a new arm, not a relabel).
- **A `/answer` endpoint**: retrieve, then have Claude write a cited answer.
  This is the only genuinely LLM-shaped piece; `anthropic` is already in the
  `serve` extra. At demo volume it costs single-digit dollars.
- **Patents and theses** (49 + 48 documents) are in the corpus because Europe
  PMC indexes them. Nobody decided that. Worth an explicit call.

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

**`uv sync` will remove pytest.** It was installed ad hoc. Use
`uv sync --extra dev` to keep the 95 tests runnable.

**A checkpoint appearing in `data/models/reranker/` silently changes what
`hybrid_rerank` means.** `rerank.py` prefers that directory and falls back to the
public model only when it's absent — nothing announces the switch. So an A/B
comparison that moves the checkpoint aside must move it back, and a smoke-test
run must never write there. Check §2 before trusting any rerank number.

**Your GPU jobs may not be using the GPU.** `torch.cuda.is_available()` is
currently `False` — wrong wheel, see §5.1. Jobs still complete; they just run on
CPU and take far longer. `training_meta.json` records `"device"`, so check it.

**The corpus contains 49 patents and 48 theses.** Europe PMC indexes them, so
they came along with everything else. Not necessarily wrong, but nobody ever
decided to include them. Worth an explicit call before quoting a headline number.

---

## 7. Current git state

Committed and pushed to `github.com/PatrickYang2007/regsearch` (public).
Session 4 added, newest first:

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
