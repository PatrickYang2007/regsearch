# The fine-tuned reranker is a pool discriminator, not a relevance model

Found: 2026-08-10, incidentally — a GPU verification job asserted that an
on-topic passage outscores an off-topic one, and the assertion failed.

**This does not invalidate the +41% MRR result.** That number was measured in
the only setting the arm ever runs in, and it stands. What follows is a
statement about where the checkpoint is valid, which is narrower than "it ranks
relevant things higher".

---

## What was observed

Query: `ATAC-seq chromatin accessibility in single cells`. Same four passages
scored by `data/models/reranker` (the shipped fine-tune) and by
`cross-encoder/ms-marco-MiniLM-L-6-v2` (the base model it was fine-tuned from):

| passage | fine-tuned | off-the-shelf |
|---|---:|---:|
| on-topic: single-cell ATAC-seq across tissues | **-2.224** | 8.527 |
| on-topic: scATAC-seq enhancer accessibility | **-0.433** | 7.627 |
| off-topic: a GNN for QM9 molecular properties | **8.189** | -11.345 |
| off-topic: *a cake recipe* | **8.917** | -11.344 |

The fine-tuned model's ordering is inverted on obvious topicality: it scores a
cake recipe above a passage that restates the query. The base model separates
the two classes by ~19 points in the right direction.

A second, in-distribution probe over 8 test queries — a genuinely cited
document's passage against a passage drawn at random from the corpus:

| model | cited beats random | mean cited | mean random | gap |
|---|---:|---:|---:|---:|
| fine-tuned | 6/8 | 3.114 | 1.334 | 1.78 |
| off-the-shelf | 6/8 | -6.445 | -9.814 | 3.37 |

Same hit rate, but the fine-tune's margin is *narrower*, and its absolute
scores sit high for both classes.

## Why this is the expected outcome of how it was trained

Every negative the fine-tune saw was a **hard** negative: mined from the
`hybrid` arm's top-50 for that query (`training_meta.json`: 4,952 negatives,
`candidate_k: 50`). A passage that reaches a retrieval arm's top-50 for a query
is, by construction, already on-topic. The model was therefore trained on a
diet in which *everything is topical* and the only thing separating a positive
from a negative is whether the citing paper happened to cite it.

So it never had a reason to keep the feature "is this even about the same
field?", and gradient descent does not preserve features that never reduce
loss. It specialised into ranking *within* an on-topic pool — which is exactly
the production task, since `hybrid_rerank` only ever sees the fused top-100 —
and its behaviour on anything outside that pool is extrapolation.

## What follows from it

1. **The ablation number is unaffected.** `hybrid_rerank` reranks the fused
   candidate pool and nothing else, so the model is in-distribution for every
   measurement in `docs/ablation.md`. Both rows of the off-the-shelf comparison
   ran under that same condition.
2. **The scores are not calibrated relevance and must never be thresholded.**
   Any future "is this good enough to show / to cite" cutoff, in particular the
   proposed `/answer` endpoint, cannot use this model's raw score. A cake recipe
   scoring 8.9 is what that failure mode looks like.
3. **It is not a general-purpose reranker** and should not be described as one.
   "Fine-tuned to discriminate within our retrieved candidate pool" is the
   honest phrasing; "fine-tuned for relevance" is not.
4. **This is a good interview answer, not an embarrassment.** It is a concrete,
   measured instance of hard-negative mining narrowing a model's competence —
   the trade was real and it bought +41% MRR on the task that matters.

## Limits of this evidence

Small and deliberately cheap: one hand-written query with four passages, and
8 query/passage pairs for the in-distribution probe. Enough to establish the
direction of the effect and to disqualify score thresholding; **not** enough to
quantify it. A proper version would score the full test split against a matched
sample of random corpus passages and report the separation as a distribution.

Worth re-checking after the retrain on the current candidate pool — the effect
may get stronger, since better-mined hard negatives are, if anything, more
uniformly on-topic.
