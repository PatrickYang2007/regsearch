# regsearch

Hybrid retrieval and reranking over regulatory-genomics literature.

Given a query — a gene, a TF motif, a variant — return ranked, cited evidence
passages from the literature. Built to compare retrieval strategies honestly:
every arm is measured on the same labelled query set and reported in one
ablation table.

> **Status: in progress.** Corpus loaded (19,791 documents / 99,567 passages),
> fully embedded, HNSW index built, and all four arms run end to end. First
> ablation table is below.
>
> Two caveats that apply to every number here. **The labels are citation-derived
> weak supervision**, not human relevance judgements — a hand-judged set is
> still outstanding. And **the reranker is an off-the-shelf checkpoint**, not a
> fine-tuned one; that training run has not happened yet. See
> [NOTES.md](NOTES.md) for the dev log and current state.

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

| arm | Recall@50 | nDCG@10 | MRR | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| `fts` | 0.0462 | 0.0146 | 0.0430 | 1434.4 | 3265.0 |
| `dense` | **0.1236** | **0.0536** | **0.1100** | **38.3** | **50.3** |
| `hybrid` | 0.0992 | 0.0339 | 0.0792 | 1409.1 | 3369.9 |
| `hybrid_rerank` | 0.1212 | 0.0513 | 0.0971 | 9630.2 | 15192.2 |

_n=171 test queries. **Weak supervision:** labels are citation-derived — a query
is a paper's title, its positives are the papers it cites — not human relevance
judgements. `hybrid_rerank` uses an off-the-shelf `ms-marco-MiniLM-L-6-v2`, not
a fine-tuned model. Reproduce with `regsearch eval --split test --origin
citation`._

**Plain dense retrieval wins every metric, and is ~40× faster than the arms
built on top of it.** Both of the more elaborate arms are worse:

- `hybrid` (reciprocal rank fusion) *loses* to `dense`. RRF weights its inputs
  equally by construction, so fusing a much weaker lexical arm into a strong
  dense one gives the weak arm an equal vote and drags the result down.
- `hybrid_rerank` does not recover it, because it reranks the already-degraded
  fused candidate set — with a public checkpoint rather than a trained one.

That is the ablation doing its job rather than a tidy result. The two follow-ups
it points at are weighting the RRF arms instead of fusing them equally, and
fine-tuning the reranker so that row reflects a trained model.

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
