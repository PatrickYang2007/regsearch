"""Abstract chunking.

The chunker is regex-based rather than spaCy/NLTK, so its correctness rests
entirely on the abbreviation guard and the window bookkeeping. Both are easy to
break silently: a bad split produces a passage that still *looks* fine in a
search result while having lost half its claim.

Biology abstracts are the adversarial case here -- "E. coli", "e.g.", "Fig. 3",
and genus abbreviations all put a period mid-sentence.
"""

from __future__ import annotations

from regsearch.ingest.chunk import (
    chunk_document,
    split_sentences,
    window_sentences,
)


# -------------------------------------------------------- sentence splitting
def test_splits_on_plain_sentence_boundary():
    assert split_sentences("CTCF binds DNA. It loops chromatin.") == [
        "CTCF binds DNA.",
        "It loops chromatin.",
    ]


def test_does_not_split_species_initial():
    """"E. coli" is the canonical false boundary: single letter, then a period.

    The guard treats any single-letter token before a period as an initial.
    """
    assert split_sentences("We used E. coli cells. The result was clear.") == [
        "We used E. coli cells.",
        "The result was clear.",
    ]


def test_does_not_split_on_eg_abbreviation():
    assert split_sentences("Peaks were called (e.g. MACS2). Then we merged them.") == [
        "Peaks were called (e.g. MACS2).",
        "Then we merged them.",
    ]


def test_does_not_split_on_figure_reference():
    assert split_sentences("See Fig. 3 for details. Overall expression rose.") == [
        "See Fig. 3 for details.",
        "Overall expression rose.",
    ]


def test_requires_capital_or_digit_to_start_next_sentence():
    """A period followed by lowercase is decimal/abbreviation, not a boundary."""
    assert split_sentences("The value was 0.05 across replicates.") == [
        "The value was 0.05 across replicates."
    ]


def test_splits_on_question_and_exclamation():
    assert split_sentences("Does CTCF loop? Yes it does.") == [
        "Does CTCF loop?",
        "Yes it does.",
    ]


def test_collapses_whitespace_and_newlines():
    assert split_sentences("CTCF   binds\n\nDNA here.") == ["CTCF binds DNA here."]


def test_text_without_terminal_punctuation_is_one_sentence():
    assert split_sentences("No terminal punctuation here") == [
        "No terminal punctuation here"
    ]


def test_empty_and_none_inputs():
    assert split_sentences("") == []
    assert split_sentences("   ") == []
    assert split_sentences(None) == []  # type: ignore[arg-type]


# --------------------------------------------------------------- windowing
def _sents(n: int) -> list[str]:
    # Each comfortably over min_chars so no tail-folding is triggered.
    return [f"Sentence number {i} carries enough characters to stand alone here." for i in range(n)]


def test_window_overlaps_by_window_minus_stride():
    """window=3, stride=2 => consecutive windows share their boundary sentence.

    The overlap is the whole point: a claim straddling a boundary must survive
    in at least one window intact.
    """
    s = _sents(5)
    out = window_sentences(s, window=3, stride=2)
    assert len(out) == 2
    assert out[0] == " ".join(s[0:3])
    assert out[1] == " ".join(s[2:5])
    # s[2] appears in both -- that is the overlap.
    assert s[2] in out[0] and s[2] in out[1]


def test_single_sentence_yields_one_window():
    s = _sents(1)
    assert window_sentences(s, window=3, stride=2) == s


def test_fewer_sentences_than_window():
    s = _sents(2)
    assert window_sentences(s, window=3, stride=2) == [" ".join(s)]


def test_no_empty_trailing_window():
    """The loop breaks once the window reaches the end.

    Advancing past it would emit an empty final chunk, which becomes a passage
    row with no text and an embedding of nothing.
    """
    for n in range(1, 12):
        out = window_sentences(_sents(n), window=3, stride=2)
        assert all(chunk.strip() for chunk in out)


def test_short_tail_is_folded_into_previous_window():
    """A stub tail becomes part of its predecessor rather than its own passage.

    Four sentences at window=3/stride=2 give windows [0:3] and [2:4]; making the
    last two sentences tiny puts the tail under min_chars and triggers the fold.
    Without it the corpus gains a 5-character passage whose embedding is noise.
    """
    s = _sents(2) + ["A.", "B."]

    unfolded = window_sentences(s, window=3, stride=2, min_chars=0)
    assert len(unfolded) == 2 and unfolded[-1] == "A. B."  # the stub, unfolded

    out = window_sentences(s, window=3, stride=2, min_chars=80)
    assert len(out) == 1
    assert out[0].endswith("A. B.")
    assert all(len(c) >= 80 for c in out)


def test_fold_repeats_the_overlapping_sentence():
    """Documented wart, not a regression: folding concatenates two windows that
    already overlapped, so the shared sentence appears twice in the result."""
    s = _sents(2) + ["A.", "B."]
    out = window_sentences(s, window=3, stride=2, min_chars=80)
    assert out[0].count("A.") == 2


def test_fold_respects_max_chars():
    s = _sents(2) + ["A.", "B."]
    out = window_sentences(s, window=3, stride=2, min_chars=80, max_chars=100)
    assert all(len(c) <= 100 for c in out)


def test_oversized_window_truncated_on_word_boundary():
    long_sentence = "word " * 500
    out = window_sentences([long_sentence.strip()], window=3, stride=2, max_chars=100)
    assert len(out) == 1
    assert len(out[0]) <= 100
    # Truncation is rsplit(" ", 1), so it never ends mid-token.
    assert not out[0].endswith("wor")


def test_empty_sentence_list():
    assert window_sentences([], window=3, stride=2) == []


# --------------------------------------------------------- document chunking
def test_ordinal_zero_is_title_plus_lead():
    passages = chunk_document("CTCF and looping", "First claim here. Second claim here.")
    assert passages[0]["section"] == "title"
    assert passages[0]["text"].startswith("CTCF and looping First claim here.")


def test_document_with_no_abstract_still_gets_one_passage():
    """Otherwise the document is unretrievable by the dense arm entirely."""
    passages = chunk_document("CTCF and looping", None)
    assert len(passages) == 1
    assert passages[0]["section"] == "title"
    assert passages[0]["text"] == "CTCF and looping"


def test_empty_abstract_behaves_like_none():
    assert chunk_document("Title here", "") == chunk_document("Title here", None)


def test_exact_duplicate_passages_are_dropped():
    """A one-sentence abstract makes the title window and first body window
    identical in the body portion; the dedup keeps order and drops repeats."""
    passages = chunk_document("T", "Only one sentence in this abstract here.")
    texts = [p["text"] for p in passages]
    assert len(texts) == len(set(texts))


def test_all_passages_have_nonempty_text():
    passages = chunk_document("Title", "One. Two. Three. Four. Five. Six.")
    assert passages
    assert all(p["text"].strip() for p in passages)


def test_passages_respect_max_length():
    passages = chunk_document("T" * 50, "word " * 2000)
    assert all(len(p["text"]) <= 1200 for p in passages)
