# Start here

Plain-language orientation. Read this first if you've been away from the
project and want to know where it stands and what to do next.

- **This file** = what's going on, what to do.
- [NOTES.md](NOTES.md) = the detailed dev log (dense; written for future-you mid-task).
- [README.md](README.md) = the public-facing description for people who find the repo.

Last updated: **2026-08-07**, end of session 3.

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
| Ablation table | generated → [docs/ablation.md](docs/ablation.md), **two rows now stale** (§3) |
| Unit tests | 95, passing |
| Web API (FastAPI) | **exists and works** — but no `regsearch serve` command yet |
| Reranker fine-tune script | **written, never run** — no trained model exists |
| Trained reranker | **not done** — needs a GPU job |
| Human-judged eval set | **not done** |

**Right now Postgres is not running.** That's normal, not a problem — see §4.

**The important thing to not get wrong:** writing the fine-tuning script is not
the same as having a fine-tuned model. `data/models/reranker/` does not exist,
so `hybrid_rerank` is still the public off-the-shelf checkpoint, exactly as it
was in session 2. Nothing in this repo has been trained.

---

## 3. The results, and what they mean

| arm | Recall@50 | nDCG@10 | MRR | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| `fts` | 0.0462 | 0.0146 | 0.0430 | 1434.4 | 3265.0 |
| `dense` | **0.1236** | **0.0536** | **0.1100** | **38.3** | **50.3** |
| `hybrid` ⚠️ stale | 0.0992 | 0.0339 | 0.0792 | 1409.1 | 3369.9 |
| `hybrid_rerank` ⚠️ stale | 0.1212 | 0.0513 | 0.0971 | 9630.2 | 15192.2 |

> ⚠️ **The bottom two rows no longer describe the code.** They were measured
> before fusion was weighted, and the shipped default is now `w_fts=0.5`. Nobody
> has re-measured them, and **no replacement numbers have been invented** — the
> table gets re-run in one pass once the GPU fine-tune finishes, so the new
> fusion and the trained reranker land together. The top two rows are still
> correct: neither arm touches the fusion weights.

**Plain reading: the simplest arm wins.** `dense` beats everything on every
quality metric *and* is about 40× faster. The two "sophisticated" arms were
worse than the simple one.

Why that happened, in one line each:

- **`hybrid` lost** because rank fusion gave both inputs an equal vote, and
  the keyword arm is much weaker — so a bad arm drags a good one down.
- **`hybrid_rerank` didn't rescue it** because it's re-ranking the already-worse
  fused list, using an off-the-shelf model that was never trained on your data.

**This is a good result, not a failure.** "I built a hybrid search system and
measured that the fancy parts made it worse" is a stronger interview story than
a tidy chart, because it shows the evaluation was real.

### The obvious fix was tried, and it didn't work

Session 3's first job was what session 2 called the cheapest win available: stop
giving the weak keyword arm an equal vote. Fusion now takes per-arm weights, and
nine settings between 0 and 1 were swept.

**The sweep said no.** There is no setting that makes fusion better than plain
`dense` on ranking quality. The trade is completely one-directional: the more
keyword evidence you mix in, the more relevant papers turn up *somewhere* in the
top 50, and the worse the ordering gets at the very top. No middle setting
escapes it.

| keyword weight | Recall@50 | nDCG@10 | MRR |
|---:|---:|---:|---:|
| 0.0 — dense only | 0.1236 | **0.0536** | **0.1100** |
| 0.5 — **what shipped** | **0.1267** | 0.0449 | 0.0916 |
| 1.0 — the old equal vote | 0.0992 | 0.0339 | 0.0792 |

**So why ship 0.5 at all, if 0 ranks better?** Because in this pipeline fusion
isn't the last step — it hands its results to the cross-encoder, whose entire job
is to fix the ordering. What fusion needs to give it is the *biggest pile of
correct papers to reorder*, and 0.5 gives the biggest pile. In other words the
`hybrid` row is a candidate-generator being graded as if it were a final answer,
which is why its nDCG looks bad.

**That reasoning has a test attached to it.** If the fine-tuned reranker turns
out not to beat the off-the-shelf one, the justification is gone and the weight
should go back to 0. Don't let it sit at 0.5 unexamined.

### Two things you must not claim about these numbers

1. **These are not human relevance judgements.** The labels come from citations:
   a query is a paper's title, and the "correct answers" are the papers it cites.
   That's weak supervision. Call it that.
2. **The reranker is off-the-shelf**, not fine-tuned. There is no trained model
   in this project yet — the training script exists, but it has never been run
   on real data.

Also, permanently: **the keyword arm is not BM25.** It's Postgres `ts_rank_cd`.
Related, but a reviewer who knows information retrieval will catch the
difference. It's named `fts` everywhere on purpose.

---

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
ends, the process is killed with it. Your *data* is safe on `/vast` (888 MB in
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

There is **no `regsearch serve` command yet** — that's the one loose end from
session 3. Until it's wired, launch uvicorn directly as above.

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

## 5. What to do next

In the order I'd do them. The first two go together and are the whole story of
the next session.

### 1. Actually run the reranker fine-tune — *the one big thing left*

Everything is written and nothing has been trained. The script mines hard
negatives from real fused search results, filters out the three kinds of
poisoned label (see NOTES.md §3), and saves to `data/models/reranker/` — where
the search code picks it up automatically and stops falling back to the public
model.

```bash
scripts/pg_start.sh                    # MUST be up first; the job preflights it
sbatch slurm/finetune_rerank.sbatch
```

Budget ~2 hours, but **most of that is not GPU work**. Mining the training
examples runs one real hybrid search per query, 619 times, at ~1.4 s each — call
it 15-25 minutes of database time before the first gradient step. The training
itself is minutes on an A100.

The job checks after training that the saved checkpoint is actually the one the
search code will load. That check exists because the alternative failure — a save
that produced no config file — is invisible: the eval just quietly reports the
old baseline numbers a second time and nothing looks wrong.

### 2. Re-run the comparison table and replace the two stale rows

```bash
regsearch eval --split test --origin citation --out docs/ablation.md
```

One run covers both changes at once — weighted fusion *and* the trained
reranker. Then update the same table in `README.md` and drop the ⚠️ markers.

**Then answer the question from §3:** did the trained reranker beat the
off-the-shelf one? If not, the reason for keeping `w_fts=0.5` is gone and it
should go back to 0.

### 3. Wire up `regsearch serve`

The API works; there's just no CLI command for it. One Typer command calling
`regsearch.api.app.run()`. **Import it inside the function body, not at the top
of `cli.py`** — `fastapi` lives in an optional extra, and a top-level import
would make every `regsearch ingest` on a login node require it.

### 4. Cap the keyword arm's candidate set

`fts` takes 1.4 seconds per query. The reason: it ORs the query terms, which is
what fixed its accuracy, but it means ranking ~32,000 matched passages instead
of 1. Limit the candidates *before* scoring them. The recall came from the OR —
nothing requires ranking every single match.

This got more valuable in session 3: it's also what makes the fine-tune's mining
phase take 25 minutes, so fixing it speeds up every future training run.

### 5. Build a human-judged eval set — *needs you, not a model*

`export_pool_for_judging` is written and has never been run. It pools the top
results from all four arms and emits a CSV for you to hand-score.

**You have to do the judging yourself.** Generating these labels with a model
would make the entire evaluation circular. Until this exists, every number in the
repo is weak supervision — which is why `load_eval_set` refuses to default to it.

---

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

**Don't confuse "the training script exists" with "a model was trained."** The
fine-tune has only ever been run as a 4-query smoke test, which wrote to a
scratch directory on purpose — dropping a 2-step model into
`data/models/reranker/` would silently swap out the baseline and quietly corrupt
the next comparison table.

**The corpus contains 49 patents and 48 theses.** Europe PMC indexes them, so
they came along with everything else. Not necessarily wrong, but nobody ever
decided to include them. Worth an explicit call before quoting a headline number.

---

## 7. Current git state

Committed to `github.com/PatrickYang2007/regsearch` (public). Session 3 added:

```
f0f8c64  Add cross-encoder fine-tune with hard-negative mining
8ce7746  Correct embed.sbatch's stale Unix-socket comment
ac1c434  Add FastAPI service over the existing retrieval arms
4f38ce9  Weight the RRF arms; fusion trades top-precision for depth-recall
62647dc  Set HF_HOME in config so weights stay off the home quota
43a1590  Add START_HERE.md orientation guide
```

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
