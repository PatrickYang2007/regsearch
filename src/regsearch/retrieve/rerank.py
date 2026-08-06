"""Cross-encoder reranking of fused candidates.

A bi-encoder scores query and passage independently, so it never sees the two
together. A cross-encoder concatenates them and runs full attention across the
pair, which catches relevance a dot product misses -- at the cost of one forward
pass per candidate, hence reranking only the fused top-k rather than the corpus.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from regsearch.config import PROJECT_ROOT
from regsearch.retrieve.search import Hit

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

log = logging.getLogger(__name__)

MODEL_DIR = PROJECT_ROOT / "data" / "models" / "reranker"
# Fallback until the fine-tune has run, so the hybrid_rerank arm is usable (and
# the eval table has an off-the-shelf baseline to beat) from day one.
BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_warned = False


def model_path() -> str:
    """Prefer the fine-tuned checkpoint; fall back to the public baseline."""
    global _warned
    if (MODEL_DIR / "config.json").exists():
        return str(MODEL_DIR)
    if not _warned:
        log.warning(
            "no fine-tuned reranker at %s; falling back to %s. "
            "Eval rows using this are the OFF-THE-SHELF baseline, not the trained model.",
            MODEL_DIR,
            BASE_MODEL,
        )
        _warned = True
    return BASE_MODEL


@lru_cache(maxsize=2)
def get_cross_encoder(path: str) -> CrossEncoder:
    from sentence_transformers import CrossEncoder
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("loading cross-encoder %s on %s", path, device)
    return CrossEncoder(path, max_length=512, device=device)


def rerank(query: str, hits: list[Hit], batch_size: int = 64) -> list[Hit]:
    """Rescore candidates with the cross-encoder. Returns a new sorted list.

    Degrades to the identity ranking if the model cannot load -- a reranker
    failing should cost relevance, not take the whole search endpoint down.
    """
    if not hits:
        return hits

    try:
        model = get_cross_encoder(model_path())
    except Exception:
        log.exception("cross-encoder unavailable; returning fused order unchanged")
        return hits

    pairs = [(query, h.text) for h in hits]
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)

    for h, sc in zip(hits, scores, strict=True):
        # Keep the fusion score under its own key: the eval harness compares
        # pre- and post-rerank ordering, which needs both.
        h.components["rrf"] = h.score
        h.components["rerank"] = float(sc)
        h.score = float(sc)

    out = sorted(hits, key=lambda h: (-h.score, h.passage_id))
    for i, h in enumerate(out):
        h.rank = i + 1
    return out
