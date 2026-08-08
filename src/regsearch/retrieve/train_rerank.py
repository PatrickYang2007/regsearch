"""Fine-tune the cross-encoder reranker on citation-derived weak supervision.

The `hybrid_rerank` arm currently runs an off-the-shelf public checkpoint, so
the ablation table contains no trained model at all while 619 training queries
sit unused. This module turns those queries into (query, passage, label) pairs
and trains a cross-encoder on them, writing the result where `rerank.py` already
looks for it.

**These labels are WEAK SUPERVISION, not human relevance judgements.** A query
is a citing paper's title; its positives are the papers it cites that happen to
be present in this corpus. Nobody judged anything. Two consequences that shape
every design choice below: the positive set is *incomplete* (most of a
reference list is out of corpus), and "not a positive" therefore does not mean
"irrelevant".

Pipeline
--------
1. Load `origin='citation' AND split='train'` queries + qrels via the eval
   harness, so training and evaluation read labels through the same code path.
2. For each query, run the real `hybrid` retrieval arm to get the candidate
   pool the reranker will actually face at inference.
3. Positives from qrels; negatives from that pool, minus the qrels, minus
   canonical twins of the qrels, minus the query's own citing paper.
4. Train a single-logit CrossEncoder with BCE and save to `data/models/reranker`.

The pure selection logic (steps 1 and 3) is deliberately separated from the
database and the GPU so it can be unit-tested without either.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Importing config also sets HF_HOME, which must happen before anything pulls in
# huggingface_hub -- it resolves its cache location once, at first import.
from regsearch.config import get_settings
from regsearch.db import client as db
from regsearch.eval.harness import load_eval_set
from regsearch.retrieve.rerank import BASE_MODEL, MODEL_DIR
from regsearch.retrieve.search import search

log = logging.getLogger(__name__)

POSITIVE_LABEL = 1.0
NEGATIVE_LABEL = 0.0


# --------------------------------------------------------------- data types
@dataclass(frozen=True)
class Candidate:
    """One retrieved passage, stripped down to what pair-building needs."""

    passage_id: int
    doc_id: int
    text: str
    rank: int = 0


@dataclass(frozen=True)
class TrainingPair:
    query_id: int
    query_text: str
    passage_id: int
    doc_id: int
    passage_text: str
    label: float
    # "positive_retrieved" | "positive_fallback" | "hard_negative" -- kept so a
    # mined set can be audited without re-running the mining.
    kind: str = ""


@dataclass
class MiningStats:
    queries_seen: int = 0
    queries_used: int = 0
    queries_without_negatives: int = 0
    positives: int = 0
    negatives: int = 0
    twins_blocked: int = 0
    self_matches_blocked: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


# ------------------------------------------------------- pure selection logic
def to_canonical(doc_id: int, canonical_map: Mapping[int, int]) -> int:
    """Cluster representative for a document.

    A missing key means identity. That is `db.load_canonical_map`'s convention
    (it omits self-mappings to stay small) and the eval harness's convention,
    and matching it here means training and evaluation agree on what "the same
    paper" means.
    """
    return canonical_map.get(doc_id, doc_id)


def positive_clusters(
    qrel_doc_ids: Iterable[int], canonical_map: Mapping[int, int]
) -> set[int]:
    """Qrel documents collapsed to their canonical clusters.

    Everything downstream compares clusters, never raw doc_ids, so that a
    preprint twin of a cited paper can never be mistaken for a negative.
    """
    return {to_canonical(d, canonical_map) for d in qrel_doc_ids}


def select_hard_negatives(
    candidates: Sequence[Candidate],
    positives: set[int],
    canonical_map: Mapping[int, int],
    max_negatives: int,
    exclude_docs: Iterable[int] = (),
    skip_top_n: int = 0,
    stats: MiningStats | None = None,
) -> list[Candidate]:
    """Pick hard negatives out of a real retrieval run.

    HARD, NOT RANDOM -- this is the most important decision in this module.
    A random passage from 99,567 is almost always about a different organism,
    assay and decade, so a cross-encoder separates it from a positive on
    surface vocabulary alone and learns nothing transferable. At inference the
    reranker only ever sees the fused top-k, where every candidate already
    survived both retrieval arms. Training on easy negatives while serving hard
    ones is a train/serve mismatch whose usual symptom is a beautiful training
    loss and a flat nDCG. So negatives are drawn from the system's own output:
    passages `search(q, arm="hybrid")` ranked highly that the weak labels do
    not call positive. Those are exactly the decisions the reranker exists to
    fix.

    `candidates` are consumed in rank order, so the cap keeps the HARDEST
    negatives rather than an arbitrary subset.

    Three exclusions, each of which would otherwise inject a *contradictory*
    label rather than a hard one:

    1. `positives` (as canonical clusters, not doc_ids). A preprint and its
       published record are separate rows with separate doc_ids and near
       identical text; 1,335 such clusters exist here, 7% of the corpus. If a
       qrel names the published record and retrieval returns the preprint, the
       naive "not in qrels" rule labels a near-copy of a positive as a
       negative.
    2. `exclude_docs` -- in practice the query's own citing paper. The queries
       are paper titles, so searching for one returns that paper's own abstract
       at rank ~1, and it is never in its own qrels because papers do not cite
       themselves. Left in, it becomes a near-exact lexical match to the query
       labelled irrelevant, which is about the worst lesson a reranker can be
       taught.
    3. Repeat clusters. One document contributing several near-identical
       passages to the same query would over-weight it against the cap.

    `skip_top_n` discards the first N candidates before sampling. False
    negatives (relevant but never cited, or cited but out of corpus) are
    concentrated at the very top of the pool, and dropping that head is the
    standard denoising trick. Default 0: it trades hardness for cleanliness and
    which way that trade goes is an empirical question this project has not
    measured yet.
    """
    blocked = {to_canonical(d, canonical_map) for d in exclude_docs}
    seen: set[int] = set()
    out: list[Candidate] = []

    for cand in candidates[skip_top_n:]:
        if len(out) >= max_negatives:
            break
        cluster = to_canonical(cand.doc_id, canonical_map)
        if cluster in positives:
            if stats is not None and cand.doc_id not in positives:
                stats.twins_blocked += 1
            continue
        if cluster in blocked:
            if stats is not None:
                stats.self_matches_blocked += 1
            continue
        if cluster in seen:
            continue
        seen.add(cluster)
        out.append(cand)

    return out


def build_pairs(
    query_id: int,
    query_text: str,
    qrel_doc_ids: Sequence[int],
    candidates: Sequence[Candidate],
    canonical_map: Mapping[int, int],
    fallback_passages: Mapping[int, Candidate],
    max_negatives: int = 8,
    max_positives: int = 8,
    exclude_docs: Iterable[int] = (),
    skip_top_n: int = 0,
    stats: MiningStats | None = None,
) -> list[TrainingPair]:
    """Turn one query's qrels + retrieval run into labelled pairs.

    Positives need a *passage*, but qrels name *documents*. Where the positive
    document turned up in the candidate pool, its retrieved passage is used --
    that is the realistic case the reranker must fix, the right paper present
    but ranked too low. Otherwise (Recall@50 is ~0.12, so this is the common
    case) the document's lead passage from `fallback_passages` stands in.

    A query with no usable negatives contributes nothing: positives alone give
    the loss no contrast and just bias the model toward predicting 1.

    Selection is deterministic -- positives ordered by doc_id, negatives by
    retrieval rank -- so two mining runs over the same database agree.
    """
    positives = positive_clusters(qrel_doc_ids, canonical_map)

    negatives = select_hard_negatives(
        candidates,
        positives,
        canonical_map,
        max_negatives=max_negatives,
        exclude_docs=exclude_docs,
        skip_top_n=skip_top_n,
        stats=stats,
    )
    if not negatives:
        return []

    # Best retrieved passage per positive cluster: first in rank order wins.
    retrieved_by_cluster: dict[int, Candidate] = {}
    for cand in candidates:
        cluster = to_canonical(cand.doc_id, canonical_map)
        if cluster in positives and cluster not in retrieved_by_cluster:
            retrieved_by_cluster[cluster] = cand

    pos_pairs: list[TrainingPair] = []
    used_clusters: set[int] = set()
    for doc_id in sorted(set(qrel_doc_ids)):
        if len(pos_pairs) >= max_positives:
            break
        cluster = to_canonical(doc_id, canonical_map)
        if cluster in used_clusters:
            continue
        cand = retrieved_by_cluster.get(cluster)
        kind = "positive_retrieved"
        if cand is None:
            cand = fallback_passages.get(doc_id)
            kind = "positive_fallback"
        if cand is None:  # document has no passages -- nothing to train on
            continue
        used_clusters.add(cluster)
        pos_pairs.append(
            TrainingPair(
                query_id=query_id,
                query_text=query_text,
                passage_id=cand.passage_id,
                doc_id=cand.doc_id,
                passage_text=cand.text,
                label=POSITIVE_LABEL,
                kind=kind,
            )
        )

    if not pos_pairs:
        return []

    neg_pairs = [
        TrainingPair(
            query_id=query_id,
            query_text=query_text,
            passage_id=c.passage_id,
            doc_id=c.doc_id,
            passage_text=c.text,
            label=NEGATIVE_LABEL,
            kind="hard_negative",
        )
        for c in negatives
    ]

    if stats is not None:
        stats.positives += len(pos_pairs)
        stats.negatives += len(neg_pairs)

    return pos_pairs + neg_pairs


# ------------------------------------------------------------- database side
def load_citing_docs(query_texts: Sequence[str]) -> dict[str, list[int]]:
    """Map each query text back to the paper(s) whose citations produced it.

    `eval_queries` carries no foreign key to the citing document, so the join
    is on the text itself -- `citation_contexts.context_text` is what
    `eval/build.py` copied into `query_text`. Verified to resolve for all 619
    training queries. Needed so the citing paper can be kept out of its own
    negative pool; see `select_hard_negatives`.
    """
    if not query_texts:
        return {}
    with db.connection() as conn:
        rows = conn.execute(
            """
            SELECT context_text, array_agg(DISTINCT citing_doc_id) AS citing
            FROM citation_contexts
            WHERE context_text = ANY(%s)
            GROUP BY context_text
            """,
            (list(query_texts),),
        ).fetchall()
    return {r["context_text"]: list(r["citing"]) for r in rows}


def load_lead_passages(doc_ids: Iterable[int]) -> dict[int, Candidate]:
    """Lowest-ordinal passage per document, for positives retrieval missed.

    Lowest ordinal because chunking is sequential over the abstract, so
    ordinal 0 is the opening of the abstract -- the chunk most likely to state
    what the paper is about.
    """
    ids = sorted(set(doc_ids))
    if not ids:
        return {}
    with db.connection() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT ON (doc_id) passage_id, doc_id, text
            FROM passages
            WHERE doc_id = ANY(%s)
            ORDER BY doc_id, ordinal
            """,
            (ids,),
        ).fetchall()
    return {
        r["doc_id"]: Candidate(
            passage_id=r["passage_id"], doc_id=r["doc_id"], text=r["text"]
        )
        for r in rows
    }


# -------------------------------------------------------------------- mining
def mine_training_pairs(
    split: str = "train",
    origin: str = "citation",
    max_negatives_per_query: int = 8,
    max_positives_per_query: int = 8,
    candidate_k: int = 50,
    limit: int | None = None,
    skip_top_n_negatives: int = 0,
    arm: str = "hybrid",
    progress_every: int = 25,
) -> tuple[list[TrainingPair], MiningStats]:
    """Build the full training set by running real retrieval per query.

    Cost warning: this is one hybrid search per query, and `hybrid` p50 is
    ~1.4 s on this corpus (the lexical arm dominates). 619 queries is roughly
    15 minutes of Postgres-bound wall clock *before the first gradient step*.
    That is the price of hard negatives and the sbatch budgets for it.

    Note this calls `search()` and consumes whatever it returns. It reads
    nothing about how fusion works internally -- no weights, no rrf_k, no arm
    list -- so a change to RRF weighting shifts the candidate pool and mining
    simply picks up the new pool on the next run.
    """
    eval_set = load_eval_set(split=split, origin=origin)
    if not eval_set:
        raise RuntimeError(
            f"no training queries for split={split!r} origin={origin!r}. "
            "Run `regsearch build-evalset` first."
        )
    if limit is not None:
        eval_set = eval_set[:limit]

    canonical_map = db.load_canonical_map()
    if not canonical_map:
        log.warning(
            "no duplicate clusters recorded; preprint twins of cited papers "
            "will be mined as hard negatives. Run `regsearch dedup-docs` first"
        )

    citing = load_citing_docs([item["query_text"] for item in eval_set])

    # One round-trip for every positive that retrieval might miss, rather than
    # one per query: recall is ~12%, so most positives take the fallback path.
    lead_passages = load_lead_passages(
        d for item in eval_set for d in item["qrels"]
    )

    stats = MiningStats()
    pairs: list[TrainingPair] = []

    for i, item in enumerate(eval_set, start=1):
        stats.queries_seen += 1
        resp = search(item["query_text"], arm=arm, k=candidate_k)  # type: ignore[arg-type]
        candidates = [
            Candidate(
                passage_id=h.passage_id, doc_id=h.doc_id, text=h.text, rank=h.rank
            )
            for h in resp.hits
        ]

        got = build_pairs(
            query_id=item["query_id"],
            query_text=item["query_text"],
            qrel_doc_ids=list(item["qrels"]),
            candidates=candidates,
            canonical_map=canonical_map,
            fallback_passages=lead_passages,
            max_negatives=max_negatives_per_query,
            max_positives=max_positives_per_query,
            exclude_docs=citing.get(item["query_text"], []),
            skip_top_n=skip_top_n_negatives,
            stats=stats,
        )
        if got:
            stats.queries_used += 1
            pairs.extend(got)
        else:
            stats.queries_without_negatives += 1

        if progress_every and i % progress_every == 0:
            log.info("mined %d/%d queries -> %d pairs", i, len(eval_set), len(pairs))

    log.info("mining done: %s", stats.as_dict())
    return pairs, stats


# ------------------------------------------------------------------ training
def train_reranker(
    epochs: int = 1,
    batch_size: int = 32,
    lr: float = 2e-5,
    max_negatives_per_query: int = 8,
    limit: int | None = None,
    *,
    max_positives_per_query: int = 8,
    candidate_k: int = 50,
    skip_top_n_negatives: int = 0,
    split: str = "train",
    origin: str = "citation",
    base_model: str = BASE_MODEL,
    output_dir: Path | str = MODEL_DIR,
    max_length: int = 512,
    warmup_ratio: float = 0.1,
    seed: int = 0,
    pairs: Sequence[TrainingPair] | None = None,
) -> dict[str, Any]:
    """Mine hard negatives, fine-tune a cross-encoder, save the checkpoint.

    Trains on CITATION-DERIVED WEAK SUPERVISION (see module docstring): the
    labels are "paper A cited paper B", not human relevance judgements. Any
    number produced from the resulting model must be reported as such.

    Writes to `data/models/reranker/` by default, which is where
    `rerank.model_path()` already looks -- it prefers that directory and falls
    back to the public checkpoint only when `config.json` is absent there. So
    the `hybrid_rerank` arm switches over the moment this finishes, with no
    other code change.

    Args:
        epochs: passes over the mined pairs.
        batch_size: pairs per optimiser step.
        lr: AdamW peak learning rate.
        max_negatives_per_query: cap on hard negatives per query, taken hardest
            (highest ranked) first.
        limit: train on only the first N queries. For smoke tests.
        pairs: skip mining and train on these instead. For tests, and for
            re-training on a cached set.

    Returns a summary dict; the same content is written to
    `training_meta.json` beside the checkpoint.
    """
    random.seed(seed)
    out_dir = Path(output_dir)

    if pairs is None:
        pairs, stats = mine_training_pairs(
            split=split,
            origin=origin,
            max_negatives_per_query=max_negatives_per_query,
            max_positives_per_query=max_positives_per_query,
            candidate_k=candidate_k,
            limit=limit,
            skip_top_n_negatives=skip_top_n_negatives,
        )
    else:
        pairs = list(pairs)
        stats = MiningStats(
            queries_used=len({p.query_id for p in pairs}),
            positives=sum(1 for p in pairs if p.label > 0),
            negatives=sum(1 for p in pairs if p.label <= 0),
        )

    if not pairs:
        raise RuntimeError("no training pairs mined; refusing to train on nothing")

    meta = _fit(
        pairs,
        base_model=base_model,
        out_dir=out_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        max_length=max_length,
        warmup_ratio=warmup_ratio,
        seed=seed,
    )
    meta.update(
        {
            "n_pairs": len(pairs),
            "mining": stats.as_dict(),
            "split": split,
            "origin": origin,
            "max_negatives_per_query": max_negatives_per_query,
            "max_positives_per_query": max_positives_per_query,
            "candidate_k": candidate_k,
            "skip_top_n_negatives": skip_top_n_negatives,
            "supervision": (
                "citation-derived WEAK supervision (query = citing paper title, "
                "positives = cited papers present in corpus). NOT human judgements."
            ),
            # The negatives are whatever `hybrid` returned at mining time, so the
            # checkpoint is only valid for this retrieval configuration. Recorded
            # so a stale checkpoint is identifiable rather than silently wrong.
            "retrieval_config": _retrieval_fingerprint(),
        }
    )

    (out_dir / "training_meta.json").write_text(json.dumps(meta, indent=2))
    log.info("wrote fine-tuned reranker to %s (%d pairs)", out_dir, len(pairs))
    return meta


def _retrieval_fingerprint() -> dict[str, Any]:
    s = get_settings()
    return {
        "arm": "hybrid",
        "rrf_weighted": s.rrf_weighted,
        "rrf_weights": dict(s.rrf_weights),
        "rrf_k": s.rrf_k,
        "fts_or_semantics": s.fts_or_semantics,
        "bm25_topk": s.bm25_topk,
        "dense_topk": s.dense_topk,
    }


def _fit(
    pairs: Sequence[TrainingPair],
    base_model: str,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    max_length: int,
    warmup_ratio: float,
    seed: int,
) -> dict[str, Any]:
    """The actual gradient descent. Imported lazily so mining needs no torch.

    Uses `CrossEncoder.old_fit`, the pre-4.0 pure-PyTorch loop, rather than the
    current `CrossEncoderTrainer`. The trainer needs `datasets` and
    `accelerate`, neither of which is installed nor declared in any extra in
    pyproject.toml; `old_fit` needs only torch + transformers, which the
    `embed` extra the GPU job already installs provides. Deprecated, not
    broken. Swapping to the trainer later touches only this function.
    """
    import torch
    from sentence_transformers import CrossEncoder, InputExample
    from torch.utils.data import DataLoader

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # num_labels=1 -> a single logit, scored with BCEWithLogitsLoss. That is
    # what `rerank.py`'s model.predict() reads back as a relevance score, so the
    # head shape here and the inference path have to agree.
    model = CrossEncoder(base_model, num_labels=1, max_length=max_length, device=device)

    examples = [
        InputExample(texts=[p.query_text, p.passage_text], label=float(p.label))
        for p in pairs
    ]
    # Shuffled so a batch is not all-positive-then-all-negative: pairs come out
    # of mining grouped by query, and BCE on a single-class batch is a poor
    # gradient.
    random.shuffle(examples)

    loader = DataLoader(
        examples,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=model.smart_batching_collate,
    )
    steps = max(len(loader) * epochs, 1)

    log.info(
        "training %s on %d pairs / %d steps (device=%s, bs=%d, lr=%g, epochs=%d)",
        base_model, len(examples), steps, device, batch_size, lr, epochs,
    )

    model.old_fit(
        train_dataloader=loader,
        epochs=epochs,
        optimizer_params={"lr": lr},
        warmup_steps=int(steps * warmup_ratio),
        use_amp=(device == "cuda"),  # no-op on CPU, ~2x on an A100
        show_progress_bar=False,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))

    return {
        "base_model": base_model,
        "output_dir": str(out_dir),
        "device": device,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "max_length": max_length,
        "steps": steps,
        "seed": seed,
    }
