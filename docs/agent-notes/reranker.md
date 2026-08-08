# Agent notes — reranker fine-tune

Scope of this work: `src/regsearch/retrieve/train_rerank.py`,
`slurm/finetune_rerank.sbatch`, `tests/test_train_rerank.py`, and a one-line
comment fix in `slurm/embed.sbatch`. Nothing else was touched.

Written as I went. Newest material appended, so the ordering is roughly the
order I discovered things.

---

## 0. The headline caveat, up front

**Real training was never run.** This node has one CPU and no GPU, and a second
agent is working on it concurrently. Everything below about *training* was
validated on a deliberately tiny smoke run (a handful of pairs, one optimiser
step, CPU) whose only purpose is to prove the code path executes end to end and
writes a loadable checkpoint. **No loss curve, no accuracy number, and no
ablation row exists yet.** The real run is `sbatch slurm/finetune_rerank.sbatch`
on an A100.

Second caveat, permanent: the labels are **citation-derived weak supervision**.
Query = a citing paper's title (tier 1 of the harvest); positives = the papers
it cites that happen to be in this corpus. Nobody judged anything. The
docstrings say this; so should any table that quotes a result.

---

## 1. What I verified vs what I assumed

**VERIFIED (ran it and looked at the output):**

- `regsearch stats` matches START_HERE: 19,791 / 99,567 / 0 unembedded /
  8,075 contexts / 790 queries / 8,075 qrels. Postgres was already up.
- 619 training queries exist for `origin='citation' AND split='train'`, via
  `harness.load_eval_set('train', 'citation')`.
- Every one of those 619 queries joins to **exactly one** citing document
  through `citation_contexts.context_text = eval_queries.query_text`. Coverage
  is 619/619 — no fallback needed. (Section 3 explains why I needed this.)
- Mean qrels per train query is **10.2**, min 3, max **119**. The 119 matters:
  an uncapped positive set would let one query dominate a batch.
- `sentence_transformers` 5.6.1 is installed, but **`datasets` and `accelerate`
  are NOT**. See section 5 — this decided the training API.
- `CrossEncoder.old_fit(...)` runs to completion on this box with the installed
  versions (torch 2.13.0+cu130, transformers 5.14.1). `CrossEncoder.fit()` does
  not, because it routes through `CrossEncoderTrainer`.
- `rerank.py` gates on `data/models/reranker/config.json` existing. That
  directory does **not** exist yet, so every rerank number in `docs/ablation.md`
  is the public baseline, exactly as NOTES.md says.
- All passages are abstract chunks (`section` defaults to `'abstract'`,
  ~5 chunks/doc). There is no full-text/abstract mixture to worry about.
- pytest: 47 pre-existing tests (chunk + metrics) green before I started and
  green after; 95 total in `tests/` now, which includes the other agent's API
  tests.
- The self-match filter earns its keep, with numbers. Smoke run over 4 queries
  at `candidate_k=10`: **11 of 40 candidates were blocked as self-matches**,
  i.e. ~2.75 per query. (The counter counts blocked *passages*, and one paper
  contributes several abstract chunks, so that is roughly one self-paper per
  query appearing 2-3 times.) Without the filter a large share of the mined
  negatives would have been chunks of the query's own paper. I expected this
  effect to exist; I did not expect it to be that large.
- The mined checkpoint loads back through `CrossEncoder(path)` and scores
  sensibly (`+3.06` for an on-topic passage, `-11.18` for an off-topic patent),
  and `save_pretrained` does write the `config.json` that `rerank.model_path()`
  gates on. The sbatch asserts that gate explicitly after training.
- `settings.rrf_weights` was `{"fts": 0.3, ...}` when I read config.py and
  `{"fts": 0.5, ...}` by the time the smoke run recorded its fingerprint --
  the main session's sweep moving underneath me, mid-task. Nothing broke,
  which is the evidence that section 4 holds.

**ASSUMED (did not or could not check):**

- That the hyperparameters (1 epoch, lr 2e-5, batch 32) are reasonable. They
  are the conventional MS MARCO cross-encoder defaults, not tuned here, and
  cannot be tuned without a GPU.
- That an A100 in the `gpu` partition is what the real run gets. I copied the
  partition/`--gpus=1` lines from `slurm/embed.sbatch` and did **not** submit
  anything.
- That the mined candidate pool will look the same at real-training time as it
  does now. It will not, and that is by design — see section 4 on `rrf_weights`.
- Whether the fine-tuned checkpoint actually beats the off-the-shelf one. That
  is the entire open question and it needs the GPU run plus a re-run of
  `regsearch eval`.

---

## 2. Hard negatives — the central design decision

Random negatives would make the task trivial. Pick a random passage out of
99,567 and it is almost always about a different organism, assay, and decade;
a cross-encoder separates that from a positive on surface vocabulary alone and
learns nothing that helps at inference time. At inference the model only ever
sees the fused top-k, where *every* candidate already survived both retrieval
arms. Training on easy negatives and testing on hard ones is a train/serve
mismatch, and the usual symptom is a model whose training loss looks great and
whose nDCG does not move.

So negatives are mined from the system's own output: run
`search(query, arm="hybrid", k=candidate_k)` for each training query, and keep
the returned passages whose document is **not** in that query's qrels. These
are, by construction, passages the deployed retriever ranks highly and the
weak labels say are wrong — precisely the decisions the reranker exists to fix.

Ordering: negatives are taken in rank order, hardest (highest ranked) first,
capped at `max_negatives_per_query`.

**Tradeoffs, stated plainly:**

- *Cost.* Mining is one full hybrid search per query. `hybrid` p50 is ~1.4 s on
  this corpus (the lexical arm dominates), so 619 queries is roughly 15 minutes
  of wall clock **before a single gradient step**, and it is CPU/Postgres-bound,
  not GPU-bound. Random negatives would be free. This is the price of the
  design and the sbatch budgets for it.
- *False negatives.* This is the honest weakness. A mined negative is
  **unjudged**, not judged irrelevant. The qrels only contain the cited papers
  that happen to be *in this corpus* — a citing paper's reference list is mostly
  outside it — so a genuinely relevant paper that was simply never cited (or
  was cited but is out-of-corpus) gets labelled 0. Hard-negative mining
  maximises this risk on purpose, because the candidates most likely to be
  falsely negative are exactly the ones ranked highest.
- *Mitigation available, off by default.* `skip_top_n_negatives` drops the
  first N candidates before sampling, the standard denoising trick (RocketQA
  and successors): the very top of the pool is where false negatives
  concentrate. Default 0 — I did not enable it because I have no way to measure
  whether it helps without a GPU, and turning on an unmeasured mitigation is
  just a second untested choice. It is there for the sweep.
- *Coupling to the retriever.* The negatives are defined by whatever `hybrid`
  currently returns, so the training set is a function of the retrieval
  configuration. That is the point (the model is trained against its real
  candidate distribution) but it means the checkpoint is stale if fusion
  changes materially. Recorded in `training_meta.json` next to the checkpoint.

**One trap I did not expect and had to fix — the query's own paper.** The
queries are paper titles. Searching a corpus for a paper's own title returns
that paper's own abstract at or near rank 1. Its doc_id is not in the qrels (a
paper does not cite itself), so the naive rule labels it a **hard negative** —
and it is a near-exact lexical match to the query. Training on that teaches the
model that near-exact title matches are irrelevant, which is close to the worst
possible lesson for a reranker. The citing document is therefore excluded from
the negative pool (with its canonical twins). This is why I needed the
`citation_contexts.context_text` join in section 1; it resolves for all 619.

---

## 3. Canonical twins — false negatives by another name

1,335 duplicate clusters (preprint + published record of the same paper, 7% of
the corpus). A qrel names one record; retrieval frequently surfaces the other.
Under the naive rule the twin's doc_id is "not in qrels", so a **preprint of a
cited paper becomes a hard negative** — a passage whose text is nearly
identical to a positive, labelled the opposite way. That is not a hard negative,
it is a contradictory label, and at 7% corpus prevalence there would be enough
of them to teach the model noise directly.

Fix: everything is compared at cluster level. `db.load_canonical_map()` gives
`{doc_id: canonical_doc_id}` for merged documents only (self-mappings omitted,
so a missing key means identity — I match the eval harness's convention rather
than inventing a second one). Both sides of the comparison are mapped through
it: qrel doc_ids become a set of positive *clusters*, and a candidate is a
negative only if its cluster is absent from that set.

The same mapping does double duty on the positive side: a candidate whose
cluster is in the qrel set counts as a positive even when the exact record
retrieved is the twin the qrel does not name.

Tradeoff: clustering is on the normalised title, so it inherits whatever
false-merge rate that has. Merging two genuinely different papers would drop a
real negative rather than create a false one — the failure is a slightly
smaller training set, not a corrupted one. That asymmetry is why I was
comfortable applying it aggressively.

Deduping negatives by cluster (not by doc_id) also stops one document
contributing three near-identical passages to the same query's negatives.

---

## 4. Do not hardcode fusion internals

The main session is mid-sweep on `settings.rrf_weights` (weighted RRF,
currently `{"fts": 0.3, "dense": 1.0}`). The mining code calls `search()` and
consumes what it returns; it reads nothing about how fusion works and hardcodes
no weight, arm list, or `rrf_k`. When the weight is finalised the candidate
pool shifts and mining just picks up the new pool on the next run.

Consequence worth flagging: **the training set is only valid for the fusion
config it was mined under.** `training_meta.json` records `rrf_weighted`,
`rrf_weights` and `fts_or_semantics` at mining time so a checkpoint can be
traced to the retriever that produced its negatives. If the weights change
after training, the honest move is to re-mine and retrain, not to reuse.

---

## 5. Which training API — and why the deprecated one

`sentence-transformers` 5.6.1 offers two paths:

- `CrossEncoderTrainer` / `CrossEncoder.fit()` — current API. Requires
  `datasets` and `accelerate`. **Neither is installed**, and neither is in any
  extra in `pyproject.toml`.
- `CrossEncoder.old_fit()` — pre-4.0 pure-PyTorch loop. Needs only torch +
  transformers, both already present and both already in the `embed` extra that
  the GPU job installs.

I used `old_fit`. Reasoning: the alternative is adding two dependencies (plus
pyarrow) to a shared virtualenv while another agent is actively running against
it, and editing `pyproject.toml` to make the GPU job reproduce the same
install. Trading a deprecation warning for zero dependency risk and an sbatch
that works with the extras the repo already declares is the right side of that
trade for a portfolio project. Verified `old_fit` runs on this exact
torch/transformers pair — it is deprecated, not broken.

Cost of the choice: no `CrossEncoderTrainingArguments`, so no built-in eval
loop, checkpointing, or fp16 flag plumbing. `use_amp=True` is passed when CUDA
is present, which is the only one of those that materially matters here.
If someone later wants the trainer, `uv add datasets accelerate` and swap the
~15 lines in `_fit`; the data pipeline above it is untouched.

Loss: `BCEWithLogitsLoss` on a single-logit head (`num_labels=1`), the standard
pointwise cross-encoder setup and what `rerank.py`'s `model.predict()` expects
to read back out. Pointwise rather than a listwise/margin loss mostly because
the label set here is binary and noisy; a margin loss would take the weak
labels more literally than they deserve.

---

## 6. Positives — a smaller decision, still a decision

qrels name documents, but the model scores *passages*, so each positive
document needs a passage.

- If the positive document appears in the mined candidate pool, its retrieved
  passage is used. That is the realistic case: the right paper is in the pool
  but ranked too low, which is exactly what the reranker must fix.
- Otherwise (recall is ~12%, so this is the common case) the document's lead
  passage — lowest `ordinal` — is used.

The risk in a scheme like this is a *shape* shortcut: if positives were always
lead chunks and negatives always mid-document chunks, the model could learn
"abstract-shaped text = relevant" and ignore the query. I checked, and it does
not apply here — every passage in this corpus is an abstract chunk, so the two
pools are drawn from the same text distribution. Worth re-checking if full text
is ever ingested.

`max_positives_per_query` (default 8) caps the 119-qrel outlier. Uncapped, that
one query would contribute more pairs than 10 typical queries combined.
Positives are taken lowest-doc_id-first so the selection is deterministic
across runs.

---

## 7. Tests

`tests/test_train_rerank.py` — 21 tests, no GPU, no model download, no
database, 0.28 s. Everything under test is a pure function over plain
dataclasses: `to_canonical`, `positive_clusters`, `select_hard_negatives`,
`build_pairs`.

The three that carry weight:

- a candidate whose *canonical twin* is a qrel is not emitted as a negative
  (the section-3 poisoning case);
- the citing document is not emitted as a negative (the section-2 self-match
  trap);
- negatives are cluster-deduped and rank-ordered, so the cap keeps the hardest
  ones rather than an arbitrary set.

The DB and `search()` boundaries are deliberately *not* mocked. Mining is a
thin loop with all the judgement extracted out of it; a mock there would assert
the shape of my own SQL and pass whether or not the logic was right.

---

## 8. Found but not fixed

- `slurm/embed.sbatch` said the GPU job reaches Postgres "over the
  shared-filesystem Unix socket". A Unix socket is local IPC and is not
  connectable from another node; the code was fixed to TCP in session 1 but the
  comment was not. **Fixed** — it now describes the TCP path and points at
  `config.resolved_host` / `data/run/pg_host`. Called out here because it is the
  bug that cost two GPU jobs, and I nearly propagated it by copying the file.
- `eval_queries` has no foreign key to the citing document. Recovering it means
  joining on `context_text`, an unindexed text equality against an 8,075-row
  table. It works and it is fast enough at this size, but a
  `citing_doc_id` column on `eval_queries` would be the right schema. Not
  changed — schema migrations are the main session's call, and `eval/build.py`
  is the natural place to populate it.
- Mining re-runs the hybrid search from scratch every time
  (~15 min for 619 queries) with no cache. If hyperparameters get swept this
  will be the annoying part. A `--pairs-cache` JSONL would fix it; skipped as
  gold-plating for a single planned run.

---

## 9. Exact commands

Smoke test actually run on this node (tiny; proves execution, nothing else).
It wrote to a scratch directory, **not** to `data/models/reranker` — leaving a
2-step checkpoint there would silently flip the `hybrid_rerank` arm to an
untrained model and quietly corrupt the next ablation:

```bash
.venv/bin/python -c "
from regsearch.retrieve.train_rerank import train_reranker
print(train_reranker(epochs=1, batch_size=8, max_negatives_per_query=3,
                     max_positives_per_query=1, limit=4, candidate_k=10,
                     max_length=128, output_dir='/tmp/smoke-reranker'))"
```

Result: 4 queries → 16 pairs (4 positives, 12 negatives), 2 optimiser steps on
CPU, checkpoint written and reloadable. Mining stats from that run:
`queries_used=4, queries_without_negatives=0, twins_blocked=0,
self_matches_blocked=11`. `twins_blocked=0` is expected at this sample size —
7% of the corpus is duplicated, so 4 queries is far too few to hit one; the
twin path is covered by unit tests, not by this run.

Real run (do not run on the dev node):

```bash
scripts/pg_start.sh          # must be up BEFORE submitting; the job preflights it
sbatch slurm/finetune_rerank.sbatch
```

Hyperparameters are env-overridable at submit time, e.g.
`EPOCHS=2 MAX_NEG=16 SKIP_TOP_N=3 sbatch slurm/finetune_rerank.sbatch`.

Then, to get the trained row into the table:

```bash
regsearch eval --split test --origin citation --out docs/ablation.md
```
