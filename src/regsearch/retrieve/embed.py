"""Embedding: model loading, corpus encoding, query encoding.

Kept out of search.py so the lexical arm and the FastAPI service can be imported
without pulling torch into the process.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from regsearch.config import get_settings
from regsearch.db import client as db

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

# bge-* models were trained with an asymmetric setup: queries carry an
# instruction prefix, documents do not. Omitting it measurably hurts retrieval,
# and it is a silent failure -- results are merely worse, never wrong-looking.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer

    s = get_settings()
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("loading %s on %s", s.embed_model, device)
    model = SentenceTransformer(s.embed_model, device=device)
    model.max_seq_length = 512
    return model


def embed_query(query: str) -> list[float]:
    model = get_model()
    vec = model.encode(
        QUERY_PREFIX + query,
        normalize_embeddings=True,  # cosine == dot product once normalised
        show_progress_bar=False,
    )
    return vec.tolist()


def embed_texts(texts: list[str], batch_size: int | None = None) -> Any:
    s = get_settings()
    model = get_model()
    return model.encode(
        texts,  # documents: no prefix, per the bge asymmetric setup
        batch_size=batch_size or s.embed_batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )


def embed_corpus(batch_size: int | None = None, limit: int | None = None) -> int:
    """Encode every passage with a NULL embedding and write it back.

    Streams in batches rather than loading all passages: the corpus is far
    larger than a comfortable working set, and this way an interrupted run
    simply resumes -- the WHERE embedding IS NULL filter is the checkpoint.
    """
    s = get_settings()
    batch_size = batch_size or s.embed_batch_size
    total = 0

    for rows in db.iter_unembedded(batch_size=batch_size):
        vecs = embed_texts([r["text"] for r in rows], batch_size=batch_size)
        db.update_embeddings(
            [(r["passage_id"], v) for r, v in zip(rows, vecs, strict=True)]
        )
        total += len(rows)
        log.info("embedded %d passages (total %d)", len(rows), total)
        if limit is not None and total >= limit:
            break

    return total
