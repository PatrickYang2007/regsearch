# Dev log

Running notes: what was built, what broke, and why decisions were made the way
they were. Newest session first.

---

## Session 4 — 2026-08-08

Theme: **the pipeline finally beats its own baseline, and two audits found that
the repo had stopped describing itself accurately.** The fine-tune completed,
the lexical arm got 8.7× faster *and* better, and then a pair of review agents
turned up a security bug, a stale-checkpoint trap, and a README that had
drifted into claiming the opposite of what the code did.

### 1. The fine-tune ran, and it worked

Job 8740006 completed in 39:48. 8,788 pairs, 275 steps, 1 epoch.

**It trained on CPU despite holding a GPU.** `torch 2.13.0+cu130` is built for
CUDA 13.0; the cluster's driver is 570.211.01 = CUDA 12.8. CUDA guarantees
compatibility only within a major family, so `cuda.is_available()` is False and
it silently fell back. Torch's own warning says "driver is too old", which sent
me looking at the cluster — the driver is current; the *wheel* is wrong. Root
cause is `pyproject` pinning `torch>=2.3` with no index, and PyPI's default
linux wheel for 2.13.0 being the cu130 build. Fix is verified but **not
applied** (see "Where to pick up").

Mining stats vindicate the self-match exclusion emphatically:

| | |
|---|---|
| queries used | 619 / 619 (none without negatives) |
| positives / negatives | 3,836 / 4,952 |
| canonical twins blocked | 29 |
| **self-matches blocked** | **1,895** |

**1,895 — roughly 38% of all negatives — were the papers' own abstracts.**
Queries are titles, so search returns the source paper at rank ~1, and it is
never in its own qrels. Unfiltered, a third of the training signal would have
taught the model that correct answers are wrong.

### 2. Results

| arm | Recall@50 | nDCG@10 | MRR | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| `fts` | 0.0501 | 0.0172 | 0.0448 | 174.0 | 609.4 |
| `dense` | 0.1236 | **0.0536** | 0.1100 | **20.5** | **29.7** |
| `hybrid` | 0.1284 | 0.0500 | 0.1010 | 187.6 | 564.5 |
| `hybrid_rerank` | **0.1388** | 0.0522 | **0.1287** | 1185.2 | 1907.0 |

Both fused arms now beat `dense` on Recall@50, where in session 3 both lost to
it. `dense` still holds nDCG@10.

**The controlled comparison is the number worth having.** Three things changed
at once, so the table above cannot attribute the gain. Moving the checkpoint
aside and re-running `hybrid_rerank` with everything else identical
(`docs/ablation_offtheshelf.md`):

| `hybrid_rerank` | Recall@50 | nDCG@10 | MRR |
|---|---:|---:|---:|
| off-the-shelf | 0.1254 | 0.0467 | 0.0912 |
| fine-tuned | **0.1388** | **0.0522** | **0.1287** |
| | +10.7% | +11.8% | **+41.1%** |

This resolves the falsifier recorded in session 3: `w_fts=0.5` was justified by
fusion being candidate generation for a reranker, contingent on that reranker
actually working. It does, so the setting stands.

### 3. Lexical term pruning — faster *and* better

Dropping near-universal terms before the OR (`gene` is in 47.8% of passages,
`express` 40.1%, `chromatin` 24.9%), plus joining `documents` after the LIMIT
instead of doing ~40k index lookups to decorate 100 rows:

| | before | after | |
|---|---:|---:|---|
| Recall@50 | 0.0462 | 0.0501 | +8.5% |
| nDCG@10 | 0.0146 | 0.0172 | +17.6% |
| p50 | 1531 ms | 175 ms | **8.7× faster** |

Benchmarked as a speed-for-quality trade; it wasn't one. Those terms were also
polluting `ts_rank_cd`'s cover density, so removing them sharpens ranking as
well as shrinking the scan. `scripts/bench_fts.py` refuses to print latency
without quality, which is why this was caught as a win rather than assumed.

### 4. What the audits found

Two review agents were run because a lot had been committed on the strength of
*outcomes* — a benchmark improved, tests passed — rather than line-by-line
review. Both notes files are in `docs/agent-notes/`.

**Fixed this session:**

- **Socket directory was 2755, not the `0700` its own comment claimed.**
  `pg_start.sh` chmods PGDATA and the password file but never RUNDIR, and that
  directory is the stated justification for `--auth-local=trust`. Any
  group member reaching the node had password-free superuser access to the
  database; only the group-restricted parent was actually protecting it. Now
  chmodded explicitly.
- **`_retrieval_fingerprint()` was blind to the settings that determine its own
  training pool.** It exists so "a stale checkpoint is identifiable rather than
  silently wrong", and recorded the fusion settings but none of the lexical
  pruning ones. The shipped checkpoint is stale by its own criterion: its
  negatives were mined the day before pruning landed, and measured over the
  same 171 queries the fts arm's top-50 retained a mean of 0.265 of its
  passages — ~80% turnover in one of the two inputs those negatives came from.
  **The trap generalises: pruning was filed as a latency fix and changed the
  training distribution as a side effect.**
- **Published OpenAPI text said the reranker was off-the-shelf**, along with
  README, START_HERE and NOTES. The honesty constraint failing in the direction
  nobody watches — understating what had been done, in text served to callers.
- **`DocumentModel.source` documented three sources; there are eight.** Clients
  treating it as an enum would break on ~6% of documents.

**Found and NOT fixed** — carried forward:

- **"Recall@50" is recall over ~35 documents, and the arms get unequal slots.**
  `search(k=50)` returns 50 *passages*; the collapse to documents happens
  after. `dense` averages 33.3 distinct docs in its top-50 (min 18), `fts`
  41.4. So dense is graded on ~20% fewer slots in a table built for arm-vs-arm
  comparison — and inconsistently with nDCG@10 in the same table, which does
  get 10 genuine documents. The bias runs *against* the winner, so "dense
  wins nDCG" is conservative rather than inflated, but this belongs in the
  harness docstring and the README.
- **`split_tsquery`'s premise is false.** It assumes
  `websearch_to_tsquery` only emits a top-level conjunction; the English word
  "or" produces a `|`. 6 of 790 eval queries hit this, and pruning silently
  never applies inside the OR branch.
- **`assemble_tsquery` hoists a negation out of an OR branch** —
  `chromatin or cancer -gene` becomes `('chromatin'|'cancer') & !'gene'`, so
  the exclusion leaks onto the chromatin side. Unreachable from the eval set
  (0/790 queries use `!`), so no published number is affected, but reachable
  from `regsearch search` and `/search`, which take free-form text.
- **`fts_min_terms=3` disables pruning for any short query**, not just
  all-common ones. Fires on 14/171 test queries.
- **The tsquery builders and `rrf_fuse` have no tests at all.** 95 tests pass
  and none touch that path.

### 5. Also this session

- `slurm/eval.sbatch` moves the ablation off the 1-CPU node: 15 min → 5:29 on
  8 cores. **Latency columns are consequently not comparable across sessions** —
  the `fts` speedup is same-node, same-harness and real; the `dense` and
  `hybrid_rerank` deltas partly reflect 8× the threads.
- `scripts/start.sh` had three bugs, including mode 644 — so `./scripts/start.sh`
  died with "Permission denied" instead of reaching the guard explaining it must
  be sourced. The guard was unreachable by exactly the mistake it existed for.
- The judging pool is generated: 704 candidates, 40 realistic queries.
- The first fine-tune launch died on `KeyError: 'EPOCHS'` — `: "${X:=1}"` sets a
  shell variable, and the training block is a child process reading `os.environ`.
  It failed after passing preflight and allocating a GPU.

---

## Session 3 — 2026-08-07 (late)

Theme of this session: **the two cheap wins from session 2 were attempted, and
one of them refuted itself.** Weighted RRF was written, swept, and shipped at a
setting that is *worse* on ranking quality than doing nothing — for a reason
that only makes sense once you say out loud what fusion is actually for here.
Alongside that: a FastAPI service over the existing arms, and the cross-encoder
fine-tune (written, never run).

Work was split across the main session and two agents. Their running notes
(`docs/agent-notes/api.md`, `docs/agent-notes/reranker.md`) are folded into this
entry and deleted — they were scratch coordination files, not permanent docs.

### State at end of session

| Thing | Status |
|---|---|
| Weighted RRF (`rrf_weights`) | **shipped** at `{"fts": 0.5, "dense": 1.0}` — see the sweep below |
| FastAPI service (`src/regsearch/api/`) | **built**, verified against the live DB, **not** wired to the CLI |
| Cross-encoder fine-tune (`retrieve/train_rerank.py`) | **written, never actually trained** — smoke run only |
| `slurm/finetune_rerank.sbatch` | written, **never submitted** |
| `data/models/reranker/` | still does not exist → `hybrid_rerank` is still the public checkpoint |
| `docs/ablation.md` | **`hybrid` / `hybrid_rerank` rows now stale** — measured under unweighted fusion |
| Unit tests | **95 passing** (47 chunk+metrics, 27 API, 21 train_rerank) |
| `slurm/embed.sbatch` stale comment | **fixed** |
| Manual judged eval set | still not started — every number is still weak supervision |

### 1. Weighted RRF — the sweep refuted its own premise

Session 2's read was: `hybrid` loses to `dense` because RRF gives both inputs an
equal vote and the lexical arm is much weaker, so down-weighting `fts` should at
minimum stop fusion from being harmful. `rrf_fuse()` now takes per-arm weights,
and `scripts/tune_rrf.py` sweeps them.

**The sweep is cheap by construction.** Retrieval does not depend on the fusion
weight, so the script runs each arm once, caches the ranked lists, and re-fuses
them in memory across all nine settings. That is one retrieval pass plus
arithmetic instead of nine full passes — which matters because the lexical arm
costs ~1.4 s per query and there are 171 of them.

Swept on the 171-query test split, `dense` pinned at 1.0 (only the ratio can
reorder anything). Full table lives in the `rrf_weights` comment in
`src/regsearch/config.py`:

| w_fts | Recall@50 | nDCG@10 | MRR | |
|---:|---:|---:|---:|---|
| 0.00 | 0.1236 | **0.0536** | **0.1100** | dense alone |
| 0.10 | 0.1241 | 0.0522 | 0.1017 | |
| 0.20 | 0.1250 | 0.0514 | 0.1001 | |
| 0.50 | **0.1267** | 0.0449 | 0.0916 | best recall — **shipped** |
| 1.00 | 0.0992 | 0.0339 | 0.0792 | unweighted, what shipped before |

**There is no sweet spot. The trade is monotone.** Every increment of lexical
weight buys depth-recall and pays for it at the top of the ranking, all the way
across the sweep. **No weight beats dense alone on nDCG@10 or MRR.** The premise
the work was written to serve — "tuning will make fusion stop being harmful" —
is false on this corpus, and the honest thing is to record that rather than pick
the metric that flatters it. `tune_rrf.py` prints an explicit warning when no
weight beats the dense-only baseline, so this cannot be quietly tuned into a win
later.

**0.5 ships anyway, and the reason is what the arm is for.** Fusion's job in
this pipeline is **candidate generation for the cross-encoder**, not final
ranking. The reranker exists precisely to fix ordering, so what it needs from
fusion is the largest possible pool of true positives — and 0.5 maximises
Recall@50 (0.1267, the best in the sweep, above dense's 0.1236). The
consequence, worth saying out loud: **the standalone `hybrid` row in the
ablation is a candidate generator being scored as if it were a final ranking.**
Its poor nDCG is that mismatch, not only arm weakness.

If the reranker fine-tune fails to beat the off-the-shelf baseline, this
justification collapses and `w_fts` should go to 0. That is the test.

**Consequence for the published table: `hybrid` and `hybrid_rerank` in
`docs/ablation.md`, `docs/ablation_nocanon.md` and `README.md` were measured
under unweighted fusion and no longer describe the committed default.** They are
marked stale in place rather than deleted or guessed at; `fts` and `dense` are
unaffected because neither reads `rrf_weights`. Re-measure after the GPU
fine-tune, in one pass, so the new fusion and the trained reranker land in the
same table.

### 2. FastAPI service

`src/regsearch/api/`: `GET /health`, `GET /search?q=&arm=&k=`, `GET /doc/{id}`.

**Every ranked response goes through `retrieve.search.search()`.** Nothing in
the package re-implements or re-tunes retrieval, and that is the whole design
constraint: the ablation table only describes *this service* if the service runs
the code the eval measured. A convenience reimplementation would silently
invalidate the published numbers.

**Handlers are plain `def`, not `async def`.** Starlette runs a sync handler in
its worker threadpool and an `async` one directly on the event loop. The
retrieval path blocks end to end (psycopg, plus torch for `dense` and
`hybrid_rerank`), so an `async` handler would pin the event loop for the whole
query — a single 6.5-second `hybrid_rerank` call would stall every other request
in the process, and nothing in the code would look wrong afterwards. A test pins
this (`test_handlers_are_sync_so_starlette_offloads_them`) rather than a comment,
because a comment does not survive a refactor.

**Binds loopback by default.** On a shared cluster node an unauthenticated
search service on `0.0.0.0` is reachable by every other user on the box. Port-
forward instead.

Verified against the live database, not just mocks — server run on port 8077,
curled, killed:

| check | result |
|---|---|
| `/health` | 200, counts match `regsearch stats` exactly |
| `/search?arm=dense` | 200, warm `latency_ms` 12–50 ms |
| `/search?arm=fts` | 200, 122–784 ms |
| `/search?arm=hybrid` | 200, 156 ms, hits carry per-arm `components` |
| `/search?arm=hybrid_rerank` | 200, 6558 ms (one call) |
| `/search?arm=bm25` | 422, body names the four legal arms |
| `/doc/999999` / `/doc/0` / `/doc/abc` | 404 / 422 / 422 |
| `/search?q=%20&arm=fts` | 200 with `hits: []` — whitespace does not 500 |

The warm `dense` number is consistent with the ablation's 38 ms p50 / 50 ms p95,
which is the actual evidence that the service and the eval run the same arm.

**Honest caveats — these are real and none of them are fixed:**

1. **Cold start costs ~12 s and it lands inside the reported `latency_ms`.** The
   first `dense` request after a restart reported `latency_ms: 12358.23`; the
   second, same query, `50.43`. That is `sentence-transformers` loading
   `bge-small-en-v1.5` on CPU *inside the request*, because `embed.get_model()`
   is lazy and `@lru_cache`d. **Any latency a caller measures on a fresh process
   is meaningless.** Same applies to the cross-encoder. Left as is deliberately:
   eager loading contradicts the lazy startup that lets the process come up (and
   `/health` report the outage) while Postgres is down, which on this cluster is
   the normal state between allocations. The fix, when it happens, is a
   `lifespan` warm-up behind a settings boolean — matching the project's
   every-fix-gets-a-toggle convention.
2. **The `dense` arm returns a bare 500 on model-load failure.** Only
   `psycopg.Error` is caught in `app.py`. If `sentence-transformers` cannot load
   (HF cache missing, offline, OOM), `/search?arm=dense` returns an undetailed
   500 — while `hybrid_rerank` **degrades gracefully**, because `rerank()`
   catches and returns the fused order unchanged. So the two arms behave
   inconsistently under the same failure. A blanket `except Exception` was
   deliberately *not* added: it would convert genuine bugs into tidy 503s and
   hide them. The right fix is a narrow catch in `dense_search`/`embed_query`.
3. **No load testing was done.** One CPU on this node, and it was deliberately
   not run. Everything about concurrency here is reasoning from code, not
   measurement: the Starlette threadpool defaults to 40 workers against a
   connection pool of `max_size=8`, so overflow surfaces as `PoolTimeout` → 503
   (an honest failure, at least — verified that `PoolTimeout` really does
   subclass `psycopg.Error`, so it cannot leak out as a 500). On one CPU the CPU
   binds long before the pool does. Revisit before calling anything production.
4. **The `fts` arm looked faster here (122–784 ms) than the ablation's 1434 ms
   p50.** Two short keyword queries against a warm buffer cache is not a
   measurement. Recorded as an observation. **Do not quote it.**
5. **Not wired to the CLI.** `api.app.run(host, port, reload)` exists but no
   `regsearch serve` command calls it. When wiring it, **import
   `regsearch.api.app` lazily inside the command body** — `regsearch.api` exists
   precisely so `fastapi` stays in the `serve` extra, and a module-level import
   in `cli.py` would make every `regsearch ingest` on a login node require it.
6. **`/docs` needs the browser to have internet** (Swagger UI comes from a CDN).
   `/openapi.json` is the self-contained artefact. **`GET /` is a 404** — no root
   route.
7. **The RAG answer endpoint was not built.** `anthropic` is declared in the
   `serve` extra, but the endpoint needs a key-handling story and a decision
   about whether generated answers belong anywhere near an evaluation this
   project is trying to keep honest. Still open.

Three corrections made to claims that were stated confidently and were not true:
the `canonical_doc_id` schema description said NULL meant "no duplicate known"
(on this corpus `nulls 0 | self_ref 18,409 | merged 1,382`, so NULL essentially
never appears and a client branching on it would be branching on nothing); the
`components` description documented two of its four keys (`fts`/`dense` are
arms, `rrf`/`rerank` are stages, and a half-documented map is worse than an
undocumented one because a client will treat it as closed); and both 503 paths
re-raised without `from exc`, burying the psycopg error under the HTTP one in
exactly the log you read when the database has fallen over.

**One test was hardened because it could rot into passing vacuously.** The
OpenAPI honesty test asserted `spec.count("bm25") == spec.count("not bm25")` —
which is `0 == 0` if the denial is ever deleted, i.e. it would go green exactly
when the guard it exists to provide is gone. Same failure mode as session 2's
chunker `fold` test. It now asserts `"not bm25" in spec` first, and that was
negative-controlled by tampering the description to `-- BM25-like` and
confirming both assertions fail.

### 3. Cross-encoder fine-tune — written, **never run**

`src/regsearch/retrieve/train_rerank.py` + `slurm/finetune_rerank.sbatch`.

**Up front: no real training has happened.** This node has one CPU and no GPU.
Everything was validated on a deliberately tiny smoke run (4 queries → 16 pairs,
2 optimiser steps, CPU) whose only purpose was to prove the code path executes
and writes a loadable checkpoint. **There is no loss curve, no accuracy number,
and no ablation row.** The smoke run wrote to a scratch directory and
**deliberately not** to `data/models/reranker/` — leaving a 2-step model there
would silently flip the `hybrid_rerank` arm to an untrained checkpoint and
quietly corrupt the next ablation. `data/models/` still does not exist, so the
arm is still the public baseline. The real run is
`sbatch slurm/finetune_rerank.sbatch` on an A100.

**Negatives are mined from the system's own output**, not sampled randomly:
`search(q, arm="hybrid", k=candidate_k)` per training query, keeping candidates
whose document is not in that query's qrels, in rank order so the cap retains
the *hardest* ones. Random negatives make the task trivial — a random passage
out of 99,567 differs from a positive in organism, assay and decade, and a
cross-encoder separates it on surface vocabulary while learning nothing. At
inference the model only ever sees the fused top-k, where every candidate
already survived both arms; training on easy negatives and serving hard ones is
a train/serve mismatch whose symptom is a beautiful training loss and a flat
nDCG.

**The trap I did not see coming: the query's own paper.** The queries *are paper
titles*. Search a corpus for a paper's own title and it returns that paper's own
abstract at rank ~1 — and its doc_id is **never in its own qrels**, because a
paper does not cite itself. So the naive rule labels a near-exact lexical match
to the query as a **hard negative**, teaching the model that near-exact title
matches are irrelevant. That is close to the worst possible lesson for a
reranker.

Fixing it needs the citing document, which `eval_queries` has no foreign key to
— recovered by joining `citation_contexts.context_text = eval_queries.query_text`.
**Coverage is 619/619 train queries, each resolving to exactly one citing
document**, so no fallback path was needed. Measured on the smoke run at
`candidate_k=10`: **11 of 40 candidates were blocked as self-matches**, ~2.75 per
query (the counter counts blocked *passages*, and one paper contributes several
abstract chunks — so roughly one self-paper per query, appearing 2-3 times). I
expected the effect to exist. I did not expect it to be a quarter of the pool.

Two more exclusions, each of which otherwise produces a **contradictory** label
rather than a hard one:

- **Canonical twins.** A preprint of a cited paper has a different doc_id and
  near-identical text, so under the naive rule it becomes a hard negative — a
  passage almost identical to a positive, labelled the opposite way. At 1,335
  clusters / 7% of the corpus that is enough to teach noise directly. Everything
  is therefore compared at *cluster* level via `db.load_canonical_map()`, which
  omits self-mappings so a missing key means identity — matching the eval
  harness's convention rather than inventing a second one. The same mapping does
  double duty on the positive side: a retrieved twin of a cited paper counts as
  a positive.
- **Repeat clusters**, so one document cannot spend the whole negative budget on
  near-identical chunks of itself.

**The honest weakness: a mined negative is unjudged, not judged irrelevant.**
The qrels only contain cited papers that happen to be *in this corpus*, and a
citing paper's reference list is mostly outside it — so a genuinely relevant
paper that was never cited gets a 0. Hard-negative mining maximises this risk on
purpose, because the candidates most likely to be false negatives are exactly
the highest-ranked ones. `skip_top_n_negatives` (the standard RocketQA-style
denoising trick) is implemented but **defaults to 0**: turning on an unmeasured
mitigation without a GPU to measure it is just a second untested choice.

**Positives:** if the positive document is in the mined pool, its retrieved
passage is used (the realistic case — right paper, ranked too low, exactly what
the reranker must fix); otherwise its lead passage. Recall is ~12%, so the
fallback is the common case. The risk in that scheme is a *shape* shortcut —
positives always lead chunks, negatives always mid-document ones, model learns
"abstract-shaped text = relevant". Checked: every passage in this corpus is an
abstract chunk, so both pools come from the same text distribution. Re-check if
full text is ever ingested. `max_positives_per_query` (8) caps a 119-qrel
outlier (mean 10.2, min 3, max 119) that would otherwise contribute more pairs
than ten typical queries combined.

**Training uses `CrossEncoder.old_fit()`, the deprecated pre-4.0 pure-PyTorch
loop, and that is deliberate.** The current `fit()` routes through
`CrossEncoderTrainer`, which requires `datasets` and `accelerate` — **neither is
installed, and neither is in any extra in `pyproject.toml`.** `old_fit()` needs
only torch + transformers, both already provided by the `--extra embed` the GPU
job installs. The alternative was adding two dependencies (plus pyarrow) to a
shared virtualenv while another agent was actively running against it, *and*
editing `pyproject.toml` so the GPU job reproduces the install. Verified
`old_fit` runs to completion on this exact torch 2.13 / transformers 5.14 pair —
it is deprecated, not broken. Cost of the choice: no built-in eval loop or
checkpointing; `use_amp=True` is passed when CUDA is present, which is the only
one of those that matters here. Swapping to the trainer later touches ~15 lines
in `_fit`.

Loss is `BCEWithLogitsLoss` on a single-logit head — the standard pointwise
setup, and what `rerank.py`'s `model.predict()` expects to read back. Pointwise
rather than a margin loss because the labels are binary and noisy; a margin loss
would take weak supervision more literally than it deserves.

**The training set is only valid for the fusion config it was mined under.** The
mining code calls `search()` and consumes what it returns — it hardcodes no
weight, arm list or `rrf_k` — so it picks up whatever fusion is current. That is
the point (the model trains against its real candidate distribution) but it
means a checkpoint goes stale if fusion changes. `training_meta.json` records
`rrf_weighted`, `rrf_weights` and `fts_or_semantics` at mining time so a
checkpoint can be traced to the retriever that generated its negatives. If the
weights change after training, re-mine and retrain — do not reuse. (This was
live during the session: `rrf_weights` was `{"fts": 0.3}` when the mining code
was read and `{"fts": 0.5}` by the time the smoke run recorded its fingerprint,
because the sweep was landing underneath. Nothing broke, which is the evidence
that the decoupling holds.)

**Cost warning:** mining is one full hybrid search per query, and `hybrid` p50
is ~1.4 s, so 619 queries is ~15-25 min of Postgres-bound wall clock **before
the first gradient step**. The sbatch budgets 2 h almost entirely for that. There
is no pairs cache; a `--pairs-cache` JSONL would fix it if hyperparameters ever
get swept.

The sbatch **asserts after training that `rerank.model_path()` actually resolves
to the checkpoint directory** — that catches a save which produced no
`config.json`, which is otherwise invisible until the eval quietly reports
baseline numbers a second time and nobody notices.

Hyperparameters (1 epoch, lr 2e-5, batch 32) are the conventional MS MARCO
cross-encoder defaults, **not tuned here** and not tunable without a GPU. They
are env-overridable at submit time:
`EPOCHS=2 MAX_NEG=16 SKIP_TOP_N=3 sbatch slurm/finetune_rerank.sbatch`.

`tests/test_train_rerank.py` is 21 tests over pure functions, no GPU, no model
download, no database. The three that carry weight: a candidate whose canonical
twin is a qrel is not emitted as a negative; the citing document is not emitted
as a negative; negatives are cluster-deduped and rank-ordered so the cap keeps
the hardest ones. The DB and `search()` boundaries are deliberately *not* mocked
— mining is a thin loop with the judgement extracted out of it, and a mock there
would assert the shape of my own SQL and pass whether or not the logic was
right.

### 4. `slurm/embed.sbatch` — stale comment fixed

The comment claimed GPU jobs reach Postgres "over the shared-filesystem Unix
socket. That works because /vast is visible from every node." That is **exactly
the misreading that cost two failed GPU jobs in session 1**: a Unix socket is
local IPC, and a compute node seeing the socket inode on shared storage still
gets ECONNREFUSED. The code was fixed to TCP at the time; the comment was never
updated and had been documenting the broken design as if it were the design ever
since.

It now describes the TCP path (`config.resolved_host` ← `data/run/pg_host`,
scram-sha-256, `pg_hba` limited to RFC1918). Worth a line in the log because the
fine-tune sbatch is modelled on this file — leaving it would have propagated the
wrong mental model into a second GPU job.

### Found but not fixed

- **`lru_cache` on `embed.get_model` / `rerank.get_cross_encoder` does not
  lock.** Two concurrent first requests in the threadpool can each construct the
  model — two copies loaded, one discarded. Wasteful, not incorrect, invisible
  with one client, and would disappear under a lifespan warm-up.
- **`k` is never silently truncated — by coincidence.** `search()` uses
  `candidate_k = max(bm25_topk, dense_topk) = 100` and the endpoint caps `k` at
  100, so they line up exactly. Nothing enforces that. If `dense_topk` ever drops
  below 100, `?k=100` starts returning fewer than 100 hits with no error.
- **`eval_queries` has no `citing_doc_id` column.** Recovering the citing paper
  means an unindexed text equality on `context_text` against an 8,075-row table.
  Fast enough at this size; the right schema is a real column, populated in
  `eval/build.py`.
- **`/health` returning `degraded` against a genuinely dead Postgres was not
  verified** — only the monkeypatched unit test covers it. Postgres was in use by
  other work and was not stopped to check.

### Honesty items — still binding, unchanged

1. **The lexical arm is `ts_rank_cd`, not BM25.** Cover-density TF-IDF, no k1/b
   saturation. Named `fts` everywhere for this reason. The OpenAPI description
   says "not BM25" explicitly and a test now asserts that string is present.
2. **`hybrid_rerank` is still the off-the-shelf `ms-marco-MiniLM-L-6-v2`.**
   `data/models/reranker/` does not exist. Writing the fine-tune did not change
   this; only running it will.
3. **Every number in the repo is citation-derived weak supervision.** The same
   signal will train the reranker. `load_eval_set` still defaults to
   `origin='manual'`; `export_pool_for_judging` still has never been run.

### Where to pick up — next session

```bash
scripts/pg_start.sh
regsearch stats        # 19,791 / 99,567 / 0 unembedded / 8,075 contexts
```

**1. Run the fine-tune on a GPU.** `sbatch slurm/finetune_rerank.sbatch`, with
Postgres already up (the job preflights and exits otherwise). This is the only
remaining step that turns `hybrid_rerank` into a trained row.

**2. Re-run the ablation and replace the stale rows** —
`regsearch eval --split test --origin citation --out docs/ablation.md`. One pass
covers both changes: `w_fts=0.5` fusion *and* the trained checkpoint. Until then
`hybrid` / `hybrid_rerank` in `docs/ablation.md`, `docs/ablation_nocanon.md` and
`README.md` are marked stale in place. **The decision that hangs on this: if the
fine-tuned reranker does not beat the baseline, the candidate-generation
justification for `w_fts=0.5` collapses and the weight should go to 0.**

**3. Wire `regsearch serve`** — lazy import in the command body, see §2.

**4. Cap the lexical candidate set.** Still open from session 2, and now it also
gates fine-tune iteration speed: `hybrid` at ~1.4 s per query is what makes
negative mining a 15-25 minute phase.

**5. Manual judged eval set.** Still the thing that would let any number here be
called a result rather than weak supervision. Needs a human.

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

**Also now justified by the numbers, and both cheap:**

**Weight the RRF arms.** `hybrid` currently *loses* to `dense` because RRF gives
the much weaker `fts` arm an equal vote. A weighted fusion, or gating fts out
when its top score is low, should at minimum stop fusion from being harmful.
This is the cheapest available win in the table.

**Cap the lexical candidate set.** `fts` p50 is 1434 ms because ORing terms
takes it from ranking 1 passage to ranking ~32k. Restrict candidates before
`ts_rank_cd` scores them — the recall came from OR'ing, but nothing requires
ranking every match.

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

Three runs, all n=171, split=test, origin=citation. **Pre-fix** is kept as the
evidence for bug 2, not as a result.

**Pre-fix — do not quote:**

| arm | Recall@50 | nDCG@10 | MRR | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| `fts` | 0.0013 | 0.0013 | 0.0019 | 5.7 | 20.5 |
| `dense` | 0.1221 | 0.0483 | 0.1033 | 59.3 | 146.0 |
| `hybrid` | 0.1224 | 0.0470 | 0.0943 | 45.5 | 67.9 |
| `hybrid_rerank` | 0.1262 | 0.0432 | 0.0849 | 5676.7 | 7776.8 |

**Corrected, `--canonicalize` (current `docs/ablation.md`):**

| arm | Recall@50 | nDCG@10 | MRR | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| `fts` | 0.0462 | 0.0146 | 0.0430 | 1434.4 | 3265.0 |
| `dense` | **0.1236** | **0.0536** | **0.1100** | **38.3** | **50.3** |
| `hybrid` | 0.0992 | 0.0339 | 0.0792 | 1409.1 | 3369.9 |
| `hybrid_rerank` | 0.1212 | 0.0513 | 0.0971 | 9630.2 | 15192.2 |

`docs/ablation_nocanon.md` holds the same run with `--no-canonicalize`, so each
fix can be attributed separately.

**What the three runs say:**

**The lexical fix worked: `fts` Recall@50 went 0.0013 → 0.0462, ~35×.** It also
cost two orders of magnitude of latency — p50 5.7 ms → 1434 ms — because the
arm went from ranking 1 matched passage to ranking ~32k. Worse than the ~700 ms
I estimated from a single warm query. **This is now the arm's real problem** and
the obvious next lever on it: cap the candidate set before `ts_rank_cd` sees it
rather than ranking every OR match.

**Canonicalisation moves the rank-sensitive metrics, not recall.** nDCG@10
+11-18% and MRR +6-21% across arms, while Recall@50 barely shifts (dense
0.1221 → 0.1236). That is the expected shape: at depth 50 a twin was usually
counted either way, but near the top it was stealing a slot from a distinct
paper. It confirms the fix targets what it claimed to.

**The headline result is awkward and worth keeping that way: plain `dense`
wins every metric, and it is also the fastest arm by ~40×.** Both of the
"sophisticated" arms are worse than the simple one:

- `hybrid` (RRF fusion) *loses* to `dense` — 0.0992 vs 0.1236 Recall@50. Fusing
  a much weaker lexical arm into a strong dense one drags it down; RRF weights
  the two arms equally by construction, so a bad arm gets an equal vote.
- `hybrid_rerank` does not rescue it (0.1212), because it reranks the already
  degraded fused candidate set, and with an off-the-shelf checkpoint.

So the current honest summary is: **fusion and reranking as configured make
retrieval worse, and the ablation is what proves it.** That is a better
portfolio result than a tidy monotone improvement — it is the table doing its
job. Two concrete follow-ups fall straight out of it: weight the RRF arms
rather than fusing equally, and fine-tune the reranker so its row is a trained
model rather than a public baseline.

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
