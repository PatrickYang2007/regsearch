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
> Two caveats that apply to every number here. **The labels are citation-derived
> weak supervision**, not human relevance judgements — a hand-judged set is
> still outstanding. And **the reranker is an off-the-shelf checkpoint**, not a
> fine-tuned one; the fine-tuning script exists but that training run has not
> happened yet. See
> [START_HERE.md](START_HERE.md) for orientation and the next steps, or
> [NOTES.md](NOTES.md) for the full dev log.

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

> ⚠️ **Two rows below are stale.** `hybrid` and `hybrid_rerank` were measured
> under *unweighted* reciprocal rank fusion. The committed default is now
> weighted (`w_fts=0.5`), so neither number describes the current code.
> **Replacement numbers have not been measured yet, and none are guessed at
> here** — the table is re-run once the reranker fine-tune completes. `fts` and
> `dense` are unaffected; neither arm reads the fusion weights.

| arm | Recall@50 | nDCG@10 | MRR | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| `fts` | 0.0462 | 0.0146 | 0.0430 | 1434.4 | 3265.0 |
| `dense` | **0.1236** | **0.0536** | **0.1100** | **38.3** | **50.3** |
| `hybrid` ⚠️ stale | 0.0992 | 0.0339 | 0.0792 | 1409.1 | 3369.9 |
| `hybrid_rerank` ⚠️ stale | 0.1212 | 0.0513 | 0.0971 | 9630.2 | 15192.2 |

_n=171 test queries. **Weak supervision:** labels are citation-derived — a query
is a paper's title, its positives are the papers it cites — not human relevance
judgements. `hybrid_rerank` uses an off-the-shelf `ms-marco-MiniLM-L-6-v2`, not
a fine-tuned model; the fine-tune is written but has never been run. **The two
fused rows predate weighted fusion and are pending re-measurement.** Reproduce
with `regsearch eval --split test --origin citation`._

**Plain dense retrieval wins every metric, and is ~40× faster than the arms
built on top of it.** As measured, both of the more elaborate arms are worse:

- `hybrid` (reciprocal rank fusion) *loses* to `dense`. Unweighted RRF gives its
  inputs an equal vote, so fusing a much weaker lexical arm into a strong dense
  one drags the result down.
- `hybrid_rerank` does not recover it, because it reranks the already-degraded
  fused candidate set — with a public checkpoint rather than a trained one.

That is the ablation doing its job rather than a tidy result.

**The follow-up it pointed at has since been done, and it refuted itself.** RRF
now takes per-arm weights, and sweeping them showed there is no sweet spot: the
trade is monotone, and **no weight beats dense alone on nDCG@10 or MRR**.
Lexical evidence buys depth-recall and pays for it at the top of the ranking.

| w_fts | Recall@50 | nDCG@10 | MRR |
|---:|---:|---:|---:|
| 0.00 (dense alone) | 0.1236 | **0.0536** | **0.1100** |
| 0.20 | 0.1250 | 0.0514 | 0.1001 |
| 0.50 (shipped) | **0.1267** | 0.0449 | 0.0916 |
| 1.00 (unweighted) | 0.0992 | 0.0339 | 0.0792 |

`w_fts=0.5` ships anyway, because fusion's job in this pipeline is **candidate
generation for the cross-encoder, not final ranking** — the reranker exists to
fix ordering, so what it needs from fusion is the largest pool of true positives,
and 0.5 maximises Recall@50. Read the standalone `hybrid` row as a candidate
generator being scored as if it were a final ranking. If the fine-tuned reranker
turns out not to beat the off-the-shelf baseline, that justification collapses
and the weight should go to 0.

The remaining follow-up is fine-tuning the reranker so that row reflects a
trained model. The training script and its Slurm job exist; the GPU run has not
happened.

The lexical arm's latency is its own open problem: ORing query terms takes it
from ranking one matched passage to ranking ~32k, which is what buys the recall
and what costs the 1.4 s. Capping the candidate set before `ts_rank_cd` is the
next lever.

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
storage. Same-node clients connect over a Unix socket in a `0700` directory;
Slurm jobs on other nodes cannot — a Unix socket is local IPC — so the server
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
