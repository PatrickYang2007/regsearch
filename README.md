# regsearch

Hybrid retrieval and reranking over regulatory-genomics literature.

Given a query — a gene, a TF motif, a variant — return ranked, cited evidence
passages from the literature. Built to compare retrieval strategies honestly:
every arm is measured on the same labelled query set and reported in one
ablation table.

> **Status: in progress.** Corpus loaded (19,791 documents / 99,567 passages),
> fully embedded, HNSW index built, all four arms run end to end, and a FastAPI
> service serves them over HTTP. First ablation table is below.
>
> The cross-encoder has been fine-tuned on citation-derived pairs and the
> ablation re-run against it.
>
> **One caveat applies to every number here: the labels are citation-derived
> weak supervision, not human relevance judgements.** A query is a paper's
> title and its positives are the papers it cites. The reranker trains on that
> same signal, so it is scored against the family of labels it learned from. A
> hand-judged evaluation set is the outstanding work that would remove this
> asterisk. See [START_HERE.md](START_HERE.md) for orientation and next steps,
> or [NOTES.md](NOTES.md) for the full dev log.

## Why this exists

Most retrieval demos report a single configuration and no baseline. The point of
this one is the comparison: how much does dense retrieval actually add over
lexical search on domain text, and how much does a trained reranker add on top
of that — measured, not asserted.

## Architecture

```
Europe PMC ──► ingest ──► Postgres 17 + pgvector
                            ├── passages.tsv        (GIN, lexical arm)
                            └── passages.embedding  (HNSW, dense arm)
                                      │
                            reciprocal rank fusion
                                      │
                            cross-encoder reranker
                                      │
                            grounded answer + citations
```

Single store for both arms: the lexical and dense indexes live on the same
table, so fusion is one query rather than a join across two systems.

## Retrieval arms

| Arm | Implementation |
|---|---|
| `fts` | Postgres full-text search, `ts_rank_cd` over a `GENERATED` tsvector |
| `dense` | `bge-small-en-v1.5` embeddings, HNSW cosine index via pgvector |
| `hybrid` | Weighted reciprocal rank fusion over the two above (`w_fts=0.5`) |
| `hybrid_rerank` | Cross-encoder reranking the fused top-k |

**On "BM25":** the lexical arm is Postgres `ts_rank_cd`, a length-normalised
TF-IDF variant. It is *not* BM25 — there are no k1/b term-saturation
parameters. It is a fair lexical baseline and is labelled `fts` everywhere
rather than being called BM25, because the difference is real and a reviewer
who knows retrieval will notice. Swapping in true BM25 (via a dedicated index)
is a tracked follow-up.

## Reranker training data

Hand-labelling enough pairs to train a cross-encoder is not feasible for a side
project, so labels are harvested from citations: a paper's title is treated as a
query whose positives are the papers it cites that are also in this corpus.
8,075 such pairs over 790 queries, at no annotation cost.

**Hard negatives are mined from the system's own fused output** — one real
`hybrid` search per training query, keeping high-ranked candidates that the
weak labels say are wrong. Random negatives would make the task trivial: at
inference the reranker only ever sees the fused top-k, so training on easy
negatives and serving hard ones is a train/serve mismatch. Three things are
excluded because they yield a *contradictory* label rather than a hard one:
canonical twins of a cited paper (a preprint has a different id and near-
identical text), repeat chunks of one document, and — the non-obvious one — the
query's own paper, which search returns at rank ~1 because the query *is* its
title, and which is never in its own qrels.

Weak labels train the model; a separate hand-judged query set evaluates it.
These are kept apart by `eval_queries.origin` so training signal never leaks
into the reported numbers. The hand-judged set does not exist yet, which is why
every number in this repo is labelled weak supervision.

## Results

| arm | Recall@50 | nDCG@10 | MRR | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| `fts` | 0.0501 | 0.0172 | 0.0448 | 174.0 | 609.4 |
| `dense` | 0.1236 | **0.0536** | 0.1100 | **20.5** | **29.7** |
| `hybrid` | 0.1284 | 0.0500 | 0.1010 | 187.6 | 564.5 |
| `hybrid_rerank` | **0.1388** | 0.0522 | **0.1287** | 1185.2 | 1907.0 |

_n=171 test queries. **Weak supervision:** labels are citation-derived — a query
is a paper's title, its positives are the papers it cites — not human relevance
judgements. The cross-encoder is fine-tuned on that same signal, so its row is
scored against the family of labels it learned from. Latency measured on 8
cores; the `fts` figure is the one reflecting a real code change (see below).
Reproduce with `regsearch eval --split test --origin citation`._

**The full pipeline wins on recall and first-hit rank; plain `dense` still wins
nDCG@10 and is ~58× faster.** Reranking pulls more correct documents into the
top 50 and gets the first one higher, but has not overtaken dense on the graded
top-10 measure.

### What the fine-tune bought

The table above changed three things at once — a trained reranker, weighted
fusion, and lexical term pruning — so it cannot attribute the gain. This is the
control: same queries, same fusion weights, same lexical config, only the
checkpoint differs (`docs/ablation_offtheshelf.md`).

| `hybrid_rerank` | Recall@50 | nDCG@10 | MRR |
|---|---:|---:|---:|
| off-the-shelf `ms-marco-MiniLM-L-6-v2` | 0.1254 | 0.0467 | 0.0912 |
| fine-tuned on citation pairs | **0.1388** | **0.0522** | **0.1287** |
| | +10.7% | +11.8% | **+41.1%** |

**Known limitation:** those negatives were mined before lexical term pruning
landed, and pruning turned over ~80% of the lexical arm's top-50. The model is
therefore trained against a candidate distribution it no longer sees at
inference. The comparison above is still sound — both rows ran under the
current configuration — but retraining on the current pool is worth doing.

### The fusion sweep, and why it under-reads the result

Sweeping the RRF weights showed a monotone trade with no sweet spot — **no
weight beat dense alone on nDCG@10 or MRR**:

| w_fts | Recall@50 | nDCG@10 | MRR |
|---:|---:|---:|---:|
| 0.00 (dense alone) | 0.1236 | **0.0536** | **0.1100** |
| 0.20 | 0.1250 | 0.0514 | 0.1001 |
| 0.50 (shipped) | **0.1267** | 0.0449 | 0.0916 |
| 1.00 (unweighted) | 0.0992 | 0.0339 | 0.0792 |

This sweep predates term pruning, so its `w_fts=0.5` row (0.1267) is lower than
the current `hybrid` row (0.1284) for the same setting — the lexical arm it
fused was the slower, weaker one. Re-running it is a tracked follow-up.

`w_fts=0.5` ships anyway, because fusion's job in this pipeline is **candidate
generation for the cross-encoder, not final ranking** — the reranker exists to
fix ordering, so what it needs from fusion is the largest pool of true positives,
and 0.5 maximises Recall@50. Read the standalone `hybrid` row as a candidate
generator being scored as if it were a final ranking.

That justification was written with an explicit falsifier: if the fine-tuned
reranker failed to beat the off-the-shelf baseline, fusion would not be earning
its keep and `w_fts` should go to 0. It beat it on all three metrics, so the
setting stands.

The lexical arm's latency was its own open problem, and is fixed. ORing the
query terms is what buys the recall — a ten-word title ANDed matched 1 passage
in 99,567 — but it took the candidate set to a median of ~40,000 passages, and
`ts_rank_cd` had to score every one. Dropping near-universal terms before the
OR (`gene` occurs in 47.8% of passages, and discriminates nothing) cuts that to
a median of ~8,600: **p50 1531 ms → 175 ms, an 8.7× speedup, measured on the
same node and harness.**

Quality went *up* rather than down — Recall@50 +8.5%, nDCG@10 +17.6%. Those
terms were also polluting the cover-density score, so removing them sharpens
ranking as well as shrinking the scan. It was measured as a trade and turned
out not to be one.

Metrics are computed at the document level (passages collapse to their parent
document in first-appearance order), so an arm cannot win by returning several
passages from one paper. Duplicate records — a preprint and its published
version are separate Europe PMC rows — collapse to one canonical document, or
retrieving the right paper under the wrong id would score as a miss. Latency is
wall-clock per query at p50/p95 rather than a mean, which hides the tail.

## Setup

Requires Apptainer (no Docker or root needed) and Python 3.11+.

```bash
scripts/pull_image.sh     # Postgres 17 + pgvector -> containers/
scripts/pg_start.sh       # initdb on first run, then start the server
uv sync                   # ingestion + retrieval deps
uv sync --extra embed     # adds torch + sentence-transformers (GPU box)
uv sync --extra serve     # adds fastapi + uvicorn (HTTP API)
```

`scripts/pg_psql.sh` opens a shell against the running database.

Postgres runs as your own uid inside the container, with `PGDATA` on shared
storage. Same-node clients connect over a Unix socket in a `0700` directory —
that mode is what makes the socket's `trust` auth safe, and `pg_start.sh` now
sets it explicitly rather than inheriting whatever the parent had.
Slurm jobs on other nodes cannot use the socket — it is local IPC — so the server
also listens on TCP with `scram-sha-256`, and `pg_start.sh` records its hostname
in `data/run/pg_host` for clients to find. `pg_hba` is restricted to RFC1918
ranges, so it is reachable from compute nodes and not from off-cluster.

## HTTP API

`GET /health`, `GET /search?q=&arm=&k=`, `GET /doc/{doc_id}`. Every ranked
response goes through the same `retrieve.search.search()` the evaluation calls,
so the ablation table describes the service and not a parallel implementation.

```bash
uv run --extra serve uvicorn regsearch.api.app:app --port 8000
```

Loopback by default: on a shared cluster node an unauthenticated search service
on `0.0.0.0` is reachable by every other user on the box. Port-forward instead.

Two caveats worth knowing before timing it. The first request after a restart
pays ~12 s of lazy model loading, and that lands **inside** the reported
`latency_ms` — a latency measured on a fresh process is meaningless. And
`hybrid_rerank` takes roughly 5-10 seconds per query on CPU, which is why it is
not the default arm.

## Layout

```
scripts/     Apptainer + Postgres lifecycle, RRF weight sweep
src/regsearch/
  config.py  Settings (env-overridable, REGSEARCH_ prefix)
  db/        schema.sql + access layer
  ingest/    Europe PMC client, chunking
  retrieve/  the four arms, fusion, reranking, reranker fine-tune
  eval/      eval-set construction, harness, metrics
  api/       FastAPI service over the arms
slurm/       GPU batch jobs (embedding, reranker training)
```

## License

MIT
