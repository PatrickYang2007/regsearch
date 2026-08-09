# How to judge the pool

This is the one task in the project that cannot be automated. Everything else
has been built around it.

**File:** `data/judging_pool.csv` — 704 candidates across 40 queries.
**Time:** roughly 1.5–2.5 hours. It does not have to be done in one sitting.

---

## Why you, and not a model

Every relevance number in this repo is currently **citation-derived weak
supervision**: a query is a paper's title, and its "correct answers" are the
papers it cites. That is a reasonable free training signal, and it is what the
cross-encoder was fine-tuned on.

It is *not* a measurement of relevance. Using a model to fill in this CSV would
score the system against the same family of signal it learned from — the result
would be circular and would measure nothing. That is why `load_eval_set`
defaults to `origin='manual'` and refuses to fall back.

Once this file is judged and imported, the project has **one table it can
defend without an asterisk**. That is the difference between "I built a search
system" and "I measured one."

---

## The scale

Fill the `relevance` column with a number 0–3:

| | meaning |
|---|---|
| **3** | Directly answers the query. If you were searching this, you'd want this at the top. |
| **2** | Clearly relevant and useful, but not the best possible answer — covers part of it, or is about a closely related case. |
| **1** | Marginal. Same general area, mentions the topic, but doesn't really address the query. |
| **0** | Not relevant. Different topic, or only shares vocabulary. |

**Leave a cell blank if you genuinely can't tell.** Blank means "unjudged" and
is skipped on import — it is *not* treated as 0. Guessing 0 on something you
didn't read invents a negative label, which is worse than having no label.

---

## Practical advice

- **Judge query by query, not row by row.** The CSV is grouped by query. Read
  the query once, then rate its ~18 candidates together. Re-orienting for every
  row is what makes this feel long.
- **Judge the passage, not the paper.** The `passage` column is what retrieval
  actually returned. A great paper represented by an irrelevant chunk is not a
  good result.
- **Ignore the `found_by` column while judging.** It records which arms
  retrieved each candidate, and knowing that biases you toward whichever arm you
  expect to win. It exists for later analysis, not for you.
- **Don't aim for a particular distribution.** If a query has nine good answers,
  give nine good scores. Calibrating toward some imagined spread is a way of
  fitting the labels to an expectation.
- Expect a lot of 0s and 1s. That is normal and correct — the pool deliberately
  includes each arm's top hits, and the whole point is that some arms retrieve
  badly.

---

## When you're done

```bash
cd /vast/palmer/pi/garg/Patrick/regsearch
source .venv/bin/activate
scripts/pg_start.sh                              # if Postgres isn't up

regsearch import-qrels data/judging_pool.csv
regsearch eval --origin manual --out docs/ablation_manual.md
```

That last command produces **the first table in this project that is not weak
supervision.** It is the number that belongs on a résumé, and it is the only one
that does.

---

## What the pool is, technically

Candidates are **pooled** across the `fts`, `dense`, and `hybrid` arms — the
union of each arm's top 10, deduplicated. Pooling matters because judging only
one arm's output biases the qrels toward it: anything the other arms uniquely
found would be scored irrelevant purely because nobody looked at it. In this
pool `fts` uniquely contributed 259 of the 704 candidates, so judging only the
dense output would have thrown away a third of the evidence.

The queries are realistic searches, not paper titles, and were written against
the corpus's actual topic distribution so the pool is not full of questions
nothing can answer. No query was adjusted after seeing what came back —
tuning queries to retrieved output would make the evaluation circular in a
second, subtler way.

The pool was generated *after* the final retrieval changes landed
(term pruning, weighted fusion, the fine-tuned reranker), so it does not need
regenerating. Judgements made now stay valid.
