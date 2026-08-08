# Agent notes — FastAPI serving layer

Scope of this agent: validate and finish `src/regsearch/api/` and
`tests/test_api.py`. Written as I went, so the order is chronological rather
than tidy. **Nothing is committed** — everything is left in the working tree.

Owned by the main session or the other agent, and **not touched**: `cli.py`,
`retrieve/search.py`, `config.py`, `NOTES.md`, `README.md`, `START_HERE.md`,
`retrieve/train_rerank.py`, `slurm/*`.

---

## Starting state, and the first surprise

The draft was on disk and had never been run. First thing I did was run it
rather than read further:

```
$ .venv/bin/python -m pytest tests/ -q
70 passed, 1 warning in 1.15s
```

That was the surprise. The brief was "unverified draft, expect breakage" and it
was already green — 47 pre-existing + 23 new. So the useful work here was not
"make it run". It was:

1. prove the endpoints work **against the real database**, which no test covers;
2. find the tests and docstrings that pass or read as true **for the wrong
   reason**.

Postgres confirmed up first, with the expected numbers:

```
$ .venv/bin/regsearch stats
documents 19,791 | passages 99,567 | unembedded 0
citation contexts 8,075 | eval queries 790 | eval qrels 8,075
```

---

## What I VERIFIED against the live service

Server: `.venv/bin/uvicorn regsearch.api.app:app --port 8077`, backgrounded,
curled, killed. Port 8077 is free again. All four arms hit real Postgres and,
for `dense`/`hybrid_rerank`, real torch on the one CPU.

| check | result |
|---|---|
| `GET /health` | 200, `status: ok`, `database: true`, counts match `regsearch stats` exactly |
| `GET /search?arm=dense` | 200, sensible hits, warm `latency_ms` 12.4 → **50.4 ms** |
| `GET /search?arm=fts` | 200, `latency_ms` 122–784 ms depending on query |
| `GET /search?arm=hybrid` | 200, 156 ms, hits carry `{"fts": …, "dense": …}` |
| `GET /search?arm=hybrid_rerank` | 200, **6558 ms** (one call only, as instructed) |
| `GET /search?arm=bm25` | **422**, body names the four legal arms |
| `GET /doc/999999` | **404**, `no document with doc_id 999999` |
| `GET /doc/20482` | 200, metadata + 4 passages, ordinals `[0,1,2,3]` |
| `GET /doc/0` | 422 (`ge=1` on the path param) |
| `GET /doc/abc` | 422 |
| `GET /search?q=%20&arm=fts` | 200 with `hits: []` — whitespace does **not** 500 |
| `POST /search` | 405 |
| `GET /openapi.json`, `GET /docs` | 200 |

Two of those are worth pulling out.

**Cold start costs 12 seconds and it lands inside `latency_ms`.** The very first
`dense` request after a restart reported `latency_ms: 12358.23`; the second, same
query, reported `50.43`. The 12 s is `sentence-transformers` loading
`bge-small-en-v1.5` on CPU inside the request, because `embed.get_model()` is
`@lru_cache`d and lazy. The warm 50 ms is consistent with the ablation table's
38 ms p50 / 50 ms p95, so the arm in the service really is the arm the eval
measured — which is the whole point of routing through `search()`. But the first
request is not, and any latency a caller measures on a fresh process is
meaningless. Same applies to `hybrid_rerank`'s cross-encoder. **Not fixed** — see
"left undone" below.

**The `fts` arm is faster here than the ablation says** (122 ms and 784 ms vs a
1434 ms p50). Two short keyword queries against a warm buffer cache is not a
measurement, and I was told not to run load tests, so I am recording it as an
observation, not a result. Do not quote it.

Claims in the code I checked rather than took on faith:

- **"PoolTimeout subclasses OperationalError"** (`app.py`, justifying the 503).
  True: `PoolTimeout → OperationalError → DatabaseError → Error`, and
  `issubclass(psycopg_pool.PoolTimeout, psycopg.Error)` is `True`. So a pool
  exhaustion really does come out as 503 and not as a 500.
- **"handlers are plain `def` so Starlette offloads them"**. True for all three.
  I turned it into a test rather than leaving it as a comment (below).
- **`k` is never silently truncated.** `search()` uses
  `candidate_k = max(bm25_topk, dense_topk) = max(100, 100) = 100` and the
  endpoint caps `k` at 100, so the two line up exactly. This is a coincidence of
  config values, not a constraint anything enforces — if `dense_topk` ever drops
  below 100, `?k=100` starts returning fewer than 100 hits with no error. Noted,
  not guarded, because the guard would have to live in `search.py`, which I do
  not own.

## What I ASSUMED (not verified)

- **`/health` returning `degraded` when Postgres is actually gone.** Only
  covered by the unit test, which monkeypatches `db.stats` to raise. I did not
  stop Postgres to check the real path — the main session and the other agent
  are both using it. The docstring's further claim that a first call against a
  dead database *blocks until the pool gives up* is likewise untested; it is
  plausible from `ConnectionPool(open=True)` semantics but I did not time it.
- **Concurrency.** No load test was run (instructed not to, and this node has
  one CPU). Everything I say below about the threadpool is reasoning from the
  code, not measurement.
- **`/docs` rendering.** It returns 200, but the HTML pulls Swagger UI from
  `cdn.jsdelivr.net`. It will render only if the *browser* has internet — fine
  over an SSH port-forward from a laptop, blank on an isolated node. Not a bug,
  but do not assume `/docs` proves anything offline; `/openapi.json` is the
  self-contained artefact.

---

## What I changed, and why

Four changes. Three are corrections to things that were stated confidently and
were not true; one hardens a test that could rot into passing vacuously.

### 1. `models.py` — `canonical_doc_id` description was wrong

It said *"NULL means no duplicate is known; readers should treat that as the
doc itself."* A client reading that would branch on NULL. On this corpus:

```
nulls 0 | self_ref 18,409 | merged 1,382 | total 19,791
```

`rebuild_canonical_docs()` writes the representative for **every** document with
a normalised title ≥ 10 chars, so the overwhelmingly common value is the
document's own id, and NULL never appears. Rewritten to say that, and to say
NULL is reachable only where clustering cannot apply (no title, or a normalised
title under 10 characters). The live `/doc/20482` response shows
`canonical_doc_id: 20482`, which is what sent me looking.

### 2. `models.py` — `components` description omitted half the keys

It described `components` as per-arm provenance. For `hybrid_rerank` the real
payload is:

```json
"components": {"fts": 0.5, "dense": 0.8599…, "rrf": 0.01852…, "rerank": 9.5217…}
```

`rrf` and `rerank` are stages, not arms, and `rerank.py` writes them
deliberately so pre- and post-rerank ordering can be compared. A schema that
documents two of the four keys is worse than one that documents none, because
a client will treat the map as closed. Now describes all four and says which
are arms and which are stages.

### 3. `app.py` — `raise HTTPException(...) from exc`

Both 503 paths re-raised inside an `except` block without chaining, which
produces the "During handling of the above exception, another exception
occurred" double traceback in the uvicorn log and buries the psycopg error
under the HTTP one. Behaviour is unchanged; the log is readable. Trivial, but
this is the code path you read when the database has fallen over, so it is the
one that should not be noisy.

### 4. `tests/test_api.py` — the BM25 test could pass vacuously

The honesty test asserted `spec.count("bm25") == spec.count("not bm25")`. That
is `0 == 0` if the description is ever rewritten without the explicit denial —
the test would go green precisely when the guard it exists to provide has been
deleted. This is the same failure mode NOTES.md records for the chunker `fold`
test, so it seemed worth not repeating. Added `assert "not bm25" in spec` first.

Negative control, run to prove the hardened assertion bites — I patched the
description from `-- not BM25` to `-- BM25-like` and re-generated the schema:

```
after tampering: 'not bm25' present? False | bm25 count: 1
```

Both assertions fail on that input. Before the change, only the count equality
existed and it would have failed too (1 != 0) — but it would *not* have failed
had the mention been dropped entirely, which is the more likely edit.

Live schema today: `bm25: 1 | not bm25: 1 | ts_rank_cd: 2`. The single
occurrence is the denial. The lexical arm is `ts_rank_cd` everywhere.

---

## Tests I added (4, bringing the suite 70 → 74)

All still database-free. `.venv/bin/python -m pytest tests/ -q` → **74 passed**.

**`test_handlers_are_sync_so_starlette_offloads_them`** — asserts none of the
three endpoints is a coroutine function. Starlette runs a plain `def` handler in
its worker threadpool and an `async def` one directly on the event loop, and the
retrieval path is blocking end to end (psycopg, plus torch for `dense` and
`hybrid_rerank`). Making a handler `async` would let the single 6.5-second
`hybrid_rerank` call I measured stall every other request in the process, and
nothing about the code would look wrong afterwards. That is exactly the kind of
decision that needs a test rather than a comment. It also asserts it actually
checked all three paths, so renaming a route cannot make the loop pass by
iterating over nothing.

**Three tests for `api/queries.get_document`** — this was the only first-party
SQL in the package with zero coverage. `db.connection` is swapped for a
recording fake, so:

- the metadata row and the passage rows really are merged into one dict, both
  statements are parameterised on `doc_id`, and neither interpolates the id into
  the SQL text;
- a missing document returns `None` **without** running the passage query (one
  `execute`, not two) — the 404 path is the common bad path and should not pay
  for a scan;
- exactly one pool checkout is taken. The docstring claims this and the pool
  maxes out at 8 connections, so a second checkout per request would halve the
  concurrency the service can sustain. Worth pinning, since "add a `with
  db.connection()`" is a natural-looking edit.

---

## Broken / risky things found but NOT fixed

Out of my scope, or a design call above my pay grade. Listed so they are not
lost.

**1. First-request model load blocks for ~12 s and pollutes `latency_ms`.**
Verified, above. Two options, both cheap, both changing behaviour I was not
asked to change: a FastAPI `lifespan` that calls `embed.get_model()` at startup
(moves the 12 s to boot, where a health check can gate on it), or splitting
`latency_ms` into `retrieval_ms` and `total_ms`. My preference is the lifespan
warm-up, guarded by a settings boolean so a Postgres-less or torch-less deploy
can skip it — that also matches the project's existing "every fix gets a
toggle" convention. **Not done: it changes startup semantics and the app's
docstring explicitly commits to lazy startup so the process comes up while
Postgres is down.**

**2. `lru_cache` on `embed.get_model` / `rerank.get_cross_encoder` does not lock.**
`functools.lru_cache` does not hold a lock across the wrapped call, so two
concurrent first requests in the threadpool can both construct the model — two
copies loaded, one thrown away. Wasteful, not incorrect, and invisible with a
single client. Would disappear on its own if the warm-up above is added. Lives
in `retrieve/`, which I do not own.

**3. Threadpool (default 40) versus pool `max_size=8`.** 40 concurrent requests
can queue on 8 connections; the overflow surfaces as `PoolTimeout` → 503, which
is at least an honest failure rather than a hang. On a one-CPU node the CPU
binds long before the pool does, so this is theoretical here. Worth revisiting
before anything is called production; needs a load test, which I was told not
to run.

**4. Non-database failures in the dense arm return a bare 500.** Only
`psycopg.Error` is caught. If `sentence-transformers` cannot load — HF cache
missing, offline, OOM — `/search?arm=dense` returns 500 "Internal Server Error"
with no detail. `hybrid_rerank` already degrades gracefully (`rerank()` catches
and returns the fused order), so the two arms behave inconsistently under model
failure. I deliberately did **not** add a blanket `except Exception`: it would
convert genuine bugs into tidy 503s and hide them. The right fix is a narrow
catch in `dense_search`/`embed_query`, which is in `retrieve/`.

**5. `GET /` is a 404.** No root route. Harmless; a one-line redirect to `/docs`
would be friendlier. Not added — it is unrequested surface area.

**6. `run()` in `app.py` is dead code.** Never called; there is no CLI entry
point. See below.

**7. Not mine, noticed in passing:** `START_HERE.md` is modified and
`scripts/start.sh` is untracked in the working tree. Both belong to the main
session. I did not touch either.

---

## For the main session: CLI wiring

`app.py` already carries the entry point, unwired as instructed. Signature:

```python
from regsearch.api.app import run
run(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None
```

It calls `uvicorn.run("regsearch.api.app:app", host=host, port=port, reload=reload)`
— the import-string form, so `--reload` works. A Typer command would be:

```python
@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Run the HTTP API."""
    from regsearch.api.app import run     # lazy: keeps fastapi optional
    run(host=host, port=port, reload=reload)
```

**Import it lazily inside the command body.** `regsearch.api` deliberately
exists so `fastapi` stays in the `serve` extra; a module-level import in
`cli.py` would make every `regsearch ingest` on a login node require it.

The loopback default is deliberate and I would keep it: on a shared cluster node
an unauthenticated search service on `0.0.0.0` is reachable by every other user
on the box. Port-forward instead.

---

## Things deliberately NOT built

`NOTES.md` §2 lists a RAG answer endpoint alongside `/search` and `/doc/{id}`.
Not built — outside the brief I was given, and `anthropic` is declared in the
`serve` extra but the endpoint would need a key-handling story (and a decision
about whether generated answers are allowed anywhere near an evaluation this
project is trying to keep honest). Flagging it as still open.
