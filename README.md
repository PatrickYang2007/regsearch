# regsearch

Hybrid retrieval and reranking over regulatory-genomics literature.

Given a query — a gene, a TF motif, a variant — return ranked, cited evidence
passages from the literature. Built to compare retrieval strategies honestly:
every arm is measured on the same labelled query set and reported in one
ablation table.

> **Status: in progress.** Corpus is loaded (19,791 documents / 99,567
> passages) and lexical retrieval works. Embedding, the reranker fine-tune, and
> the eval table are outstanding. **There are no results yet** — the ablation
> table below is empty on purpose, and nothing here should be quoted as a
> measurement until `regsearch eval` has run. See [NOTES.md](NOTES.md) for the
> dev log and current state.

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
| `hybrid` | Reciprocal rank fusion over the two above |
| `hybrid+rerank` | Cross-encoder reranking the fused top-k |

**On "BM25":** the lexical arm is Postgres `ts_rank_cd`, a length-normalised
TF-IDF variant. It is *not* BM25 — there are no k1/b term-saturation
parameters. It is a fair lexical baseline and is labelled `fts` everywhere
rather than being called BM25, because the difference is real and a reviewer
who knows retrieval will notice. Swapping in true BM25 (via a dedicated index)
is a tracked follow-up.

## Reranker training data

Hand-labelling enough pairs to train a cross-encoder is not feasible for a side
project, so labels are harvested from citation contexts: a sentence in paper A
that cites paper B is treated as a query whose positive is B, with hard
negatives sampled from the lexical top-50. This yields tens of thousands of
training pairs at no annotation cost.

Weak labels train the model; a separate hand-judged query set evaluates it.
These are kept in different tables (`eval_queries.origin`) so training signal
never leaks into the reported numbers.

## Results

Not measured yet. The harness (`regsearch eval`) fills this in across all four
arms on a shared query set:

| arm | Recall@50 | nDCG@10 | MRR | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| `fts` | — | — | — | — | — |
| `dense` | — | — | — | — | — |
| `hybrid` | — | — | — | — | — |
| `hybrid+rerank` | — | — | — | — | — |

Metrics are computed at the document level (passages collapse to their parent
document in first-appearance order), so an arm cannot win by returning several
passages from one paper. Latency is wall-clock per query at p50/p95 rather than
a mean, which hides the tail.

## Setup

Requires Apptainer (no Docker or root needed) and Python 3.11+.

```bash
scripts/pull_image.sh     # Postgres 17 + pgvector -> containers/
scripts/pg_start.sh       # initdb on first run, then start on a Unix socket
uv sync                   # ingestion + retrieval deps
uv sync --extra embed     # adds torch + sentence-transformers (GPU box)
```

`scripts/pg_psql.sh` opens a shell against the running database.

Postgres runs as your own uid inside the container, with `PGDATA` on shared
storage and connections over a Unix socket in a `0700` directory — no TCP
listener, no port collisions with other users on a shared node.

## Layout

```
scripts/     Apptainer + Postgres lifecycle
src/regsearch/
  config.py  Settings (env-overridable, REGSEARCH_ prefix)
  db/        schema.sql + access layer
  ingest/    Europe PMC client, chunking
slurm/       GPU batch jobs (embedding, reranker training)
```

## License

MIT
