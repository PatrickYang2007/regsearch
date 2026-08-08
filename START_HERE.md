# Start here

Plain-language orientation. Read this first if you've been away from the
project and want to know where it stands and what to do next.

- **This file** = what's going on, what to do.
- [NOTES.md](NOTES.md) = the detailed dev log (dense; written for future-you mid-task).
- [README.md](README.md) = the public-facing description for people who find the repo.

Last updated: **2026-08-07**, end of session 2.

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
| `hybrid` | fuses the two above with reciprocal rank fusion |
| `hybrid_rerank` | takes the fused results and re-scores them with a cross-encoder |

---

## 2. Where things stand

**The whole pipeline works end to end.** As of session 2 you can ingest, embed,
search on all four arms, and produce the comparison table. That wasn't true
before — two of the four arms were broken and had never actually been run.

| | status |
|---|---|
| Corpus (19,791 papers / 99,567 passages) | loaded |
| Embeddings | complete, all 99,567 |
| Vector index (HNSW) | built |
| All four retrieval arms | working |
| Training labels (from citations) | 8,075 pairs harvested |
| Eval set | 790 queries (171 test / 619 train) |
| Ablation table | generated → [docs/ablation.md](docs/ablation.md) |
| Unit tests | 47, passing |
| Reranker fine-tune | **not done** |
| Web API | **not started** |
| Human-judged eval set | **not done** |

**Right now Postgres is not running.** That's normal, not a problem — see §4.

---

## 3. The results, and what they mean

| arm | Recall@50 | nDCG@10 | MRR | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| `fts` | 0.0462 | 0.0146 | 0.0430 | 1434.4 | 3265.0 |
| `dense` | **0.1236** | **0.0536** | **0.1100** | **38.3** | **50.3** |
| `hybrid` | 0.0992 | 0.0339 | 0.0792 | 1409.1 | 3369.9 |
| `hybrid_rerank` | 0.1212 | 0.0513 | 0.0971 | 9630.2 | 15192.2 |

**Plain reading: the simplest arm wins.** `dense` beats everything on every
quality metric *and* is about 40× faster. The two "sophisticated" arms are worse
than the simple one.

Why that happened, in one line each:

- **`hybrid` loses** because rank fusion gives both inputs an equal vote, and
  the keyword arm is much weaker — so a bad arm drags a good one down.
- **`hybrid_rerank` doesn't rescue it** because it's re-ranking the already-worse
  fused list, using an off-the-shelf model that was never trained on your data.

**This is a good result, not a failure.** "I built a hybrid search system and
measured that the fancy parts made it worse" is a stronger interview story than
a tidy chart, because it shows the evaluation was real. It also points directly
at the next two things to fix (§5).

### Two things you must not claim about these numbers

1. **These are not human relevance judgements.** The labels come from citations:
   a query is a paper's title, and the "correct answers" are the papers it cites.
   That's weak supervision. Call it that.
2. **The reranker is off-the-shelf**, not fine-tuned. There is no trained model
   in this project yet.

Also, permanently: **the keyword arm is not BM25.** It's Postgres `ts_rank_cd`.
Related, but a reviewer who knows information retrieval will catch the
difference. It's named `fts` everywhere on purpose.

---

## 4. Getting it running

### Every single session, first thing

```bash
cd /vast/palmer/pi/garg/Patrick/regsearch
scripts/pg_start.sh
regsearch stats
```

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

### Also set this before anything that loads a model

```bash
export HF_HOME="$PWD/data/hf"
```

Without it, model weights download into your home directory instead of the
project. Home is quota-limited (you're at ~30 GB of 125 GB). This is currently
only set inside `slurm/embed.sbatch`, not for interactive use — see §6.

### Try a search

```bash
regsearch search "enhancer promoter interaction chromatin looping" --arm dense
```

### Re-run the comparison table

```bash
regsearch eval --split test --origin citation --out docs/ablation.md
```

Takes ~15 min on a 1-CPU node (the cross-encoder arm dominates), minutes on a GPU
node.

---

## 5. What to do next

In the order I'd do them. The first two are cheap and directly address why the
table looks the way it does.

### 1. Weight the fusion arms — *cheapest win available*

`hybrid` currently loses to `dense` because reciprocal rank fusion treats both
inputs as equally trustworthy. Give the dense arm more weight, or drop the
keyword arm when its top score is low. Goal: make fusion stop being harmful.

### 2. Cap the keyword arm's candidate set

`fts` takes 1.4 seconds per query. The reason: it now ORs the query terms, which
is what fixed its accuracy, but it means ranking ~32,000 matched passages instead
of 1. Limit the candidates *before* scoring them. The recall came from the OR —
nothing requires ranking every single match.

### 3. Fine-tune the reranker — *biggest gap*

Right now `hybrid_rerank` uses a public checkpoint, so your ablation contains no
trained model at all. You have **619 training queries sitting unused** from the
train split.

- Train a cross-encoder on `origin='citation' AND split='train'`.
- Save it to `data/models/reranker/` — the code picks it up from there
  automatically and falls back to the public model only when it's absent.
- **Run this on a GPU via `sbatch`**, modelled on `slurm/embed.sbatch`. The
  cross-encoder alone took ~13 minutes on the 1-CPU dev node.
- Use *hard* negatives — papers that came back in the fused top-k but aren't
  cited — not random papers. Random negatives make the task trivial and the model
  learns nothing useful.

### 4. Build the web API

Not started. This is the biggest gap against the internship postings this project
was built for — they ask for API/serving experience and right now there's only a
command-line tool. `fastapi` and `uvicorn` are already declared in
`pyproject.toml` under the `serve` extra, just not installed yet
(`uv sync --extra serve`).

Reuse `retrieve.search.search()` directly so the arms behave identically in the
API and in the eval — otherwise you can't claim the table describes the service.

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

**`HF_HOME` isn't set outside the sbatch script.** So interactive runs download
model weights to your home quota, defeating the thing `embed.sbatch` was written
to avoid. Worth fixing in `config.py` so every entry point gets it.

**`slurm/embed.sbatch` has a comment that is wrong.** It claims GPU jobs reach
Postgres "over the shared-filesystem Unix socket." That's the bug that cost two
failed GPU jobs in session 1 — a Unix socket is local to one machine and can't be
used across nodes, even on shared storage. It was fixed to use TCP. The *code* is
fine; the *comment* describes the old broken design. Fix it before you copy this
file for the reranker job.

**Start Postgres before submitting a GPU job**, not after. The job checks the
connection on startup and exits immediately if it can't reach the database.

**`uv sync` will remove pytest.** It was installed ad hoc. Use
`uv sync --extra dev` to keep the 47 tests runnable.

**The corpus contains 49 patents and 48 theses.** Europe PMC indexes them, so
they came along with everything else. Not necessarily wrong, but nobody ever
decided to include them. Worth an explicit call before quoting a headline number.

---

## 7. Current git state

Everything is committed and pushed to `github.com/PatrickYang2007/regsearch`
(public). Session 2 added:

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
