"""Chunk abstracts into overlapping sentence windows.

Why not embed whole abstracts: a 250-word abstract averaged into one 384-dim
vector washes out the specific claim a query is usually about. Windows of a few
sentences keep claims locally coherent; the overlap stops a claim that straddles
a boundary from being split across two vectors and lost by both.

Deliberately regex-based rather than spaCy/NLTK: one less heavy dependency, and
the abbreviation guard below covers the cases that actually occur in abstracts.
"""

from __future__ import annotations

import re

# Don't split after these -- the period is part of the token, not a sentence end.
_ABBREVIATIONS = {
    "approx", "ca", "cf", "e.g", "et al", "etc", "fig", "figs", "i.e", "no",
    "p", "ref", "refs", "vs", "vol", "spp", "sp", "subsp", "str", "var",
    "dr", "prof", "mr", "mrs", "ms", "st", "jr", "sr", "inc", "ltd", "co",
}

_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")
_WS = re.compile(r"\s+")


def _looks_like_abbreviation(chunk: str) -> bool:
    m = re.search(r"([A-Za-z.]+)\.$", chunk.strip())
    if not m:
        return False
    tail = m.group(1).lower().rstrip(".")
    if tail in _ABBREVIATIONS:
        return True
    # Single letter before a period -> almost always an initial ("E. coli").
    return len(tail) == 1


def split_sentences(text: str) -> list[str]:
    text = _WS.sub(" ", (text or "").strip())
    if not text:
        return []

    parts = _SENT_END.split(text)
    merged: list[str] = []
    for part in parts:
        if merged and _looks_like_abbreviation(merged[-1]):
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return [p.strip() for p in merged if p.strip()]


def window_sentences(
    sentences: list[str],
    window: int = 3,
    stride: int = 2,
    min_chars: int = 80,
    max_chars: int = 1200,
) -> list[str]:
    """Slide a `window`-sentence window with the given `stride`.

    stride < window gives the overlap. A trailing window shorter than the stride
    is merged into its predecessor rather than emitted as a stub.
    """
    if not sentences:
        return []

    out: list[str] = []
    i = 0
    while i < len(sentences):
        chunk = " ".join(sentences[i : i + window]).strip()
        if len(chunk) > max_chars:
            chunk = chunk[:max_chars].rsplit(" ", 1)[0]
        if chunk:
            out.append(chunk)
        if i + window >= len(sentences):
            break
        i += stride

    # Fold a too-short tail into the previous window.
    if len(out) > 1 and len(out[-1]) < min_chars:
        out[-2] = f"{out[-2]} {out[-1]}"[:max_chars]
        out.pop()
    return out


def chunk_document(
    title: str, abstract: str | None, window: int = 3, stride: int = 2
) -> list[dict[str, str]]:
    """Produce passage rows for one document.

    Ordinal 0 is always title+lead: many queries are answered by the title alone,
    and it gives documents with no abstract at least one retrievable passage.
    """
    passages: list[dict[str, str]] = []
    sentences = split_sentences(abstract or "")

    lead = sentences[0] if sentences else ""
    head = f"{title.strip()} {lead}".strip()
    passages.append({"section": "title", "text": head[:1200]})

    for chunk in window_sentences(sentences, window=window, stride=stride):
        passages.append({"section": "abstract", "text": chunk})

    # Drop exact duplicates (short abstracts make the title window repeat the
    # first body window) while preserving order.
    seen: set[str] = set()
    deduped: list[dict[str, str]] = []
    for p in passages:
        k = p["text"]
        if k and k not in seen:
            seen.add(k)
            deduped.append(p)
    return deduped
