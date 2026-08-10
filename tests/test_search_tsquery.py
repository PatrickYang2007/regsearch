"""Unit tests for the lexical arm's tsquery rewrite.

This path had no coverage at all, which is exactly how the two bugs pinned
below (a false top-level-AND assumption, and negations being hoisted out of the
OR branch they belonged to) survived a passing 95-test suite and a benchmark.

Everything here is database-free: `build_fts_tsquery` takes the text Postgres
rendered and returns the text it will run, so the whole rewrite is testable
without a connection. The `common` maps are hand-written rather than read from
lexeme_df, so a corpus rebuild cannot silently change what these assert.
"""

from __future__ import annotations

import pytest

from regsearch.retrieve.search import (
    TsqueryParseError,
    build_fts_tsquery,
    parse_tsquery,
)

# Stand-in for lexeme_df: {lexeme: ndoc} for lexemes over the df threshold.
COMMON = {"gene": 60000, "cell": 36560, "express": 50000, "chromatin": 24797}


class TestOrRewrite:
    """The base behaviour: a conjunction becomes a disjunction."""

    def test_plain_conjunction_becomes_disjunction(self):
        assert build_fts_tsquery("'enhanc' & 'promot'") == "('enhanc' | 'promot')"

    def test_single_term_keeps_its_bracket(self):
        assert build_fts_tsquery("'chromatin'") == "('chromatin')"

    def test_or_semantics_off_returns_the_parse_untouched(self):
        parsed = "'enhanc' & 'promot'"
        assert build_fts_tsquery(parsed, or_semantics=False) == parsed

    def test_empty_parse_yields_empty(self):
        # All-stop-word or whitespace input. Callers read '' as "matches
        # nothing"; Postgres cannot cast an empty tsquery at all.
        assert build_fts_tsquery("") == ""
        assert build_fts_tsquery("   ") == ""


class TestTopLevelOr:
    """Finding #3: websearch_to_tsquery does NOT only emit a top-level AND.

    The English word "or" produces a `|`, and `&` binds tighter than `|`. The
    old implementation split the rendered text on ' & ' and so handed back
    "'wapl' | 'yy1'" as a single opaque conjunct, which no lexeme could be read
    out of -- meaning pruning silently did not apply to any term inside an OR
    branch.
    """

    def test_or_branch_survives(self):
        # Shape taken from a real eval query: "... CTCF, cohesin, WAPL or YY1".
        assert (
            build_fts_tsquery("'ctcf' & 'cohesin' & 'wapl' | 'yy1'")
            == "('ctcf' | 'cohesin' | 'wapl') | 'yy1'"
        )

    def test_pruning_reaches_inside_an_or_branch(self):
        # 'gene' is corpus-common and sits in the OR branch. The old splitter
        # could not see it, so it was never pruned; here it must be.
        out = build_fts_tsquery(
            "'ctcf' & 'cohesin' | 'gene' & 'yy1'", common=COMMON, min_terms=1
        )
        assert "'gene'" not in out
        assert "'ctcf'" in out and "'cohesin'" in out and "'yy1'" in out

    def test_parenthesised_group_is_parsed_not_split(self):
        assert (
            build_fts_tsquery("('a' | 'b') & 'c'") == "('a' | 'b' | 'c')"
        )


class TestNegationScoping:
    """Finding #4: a negation must not escape the branch it was parsed into."""

    def test_top_level_negation_stays_conjunctive(self):
        # Folding !x into the OR would match every passage merely lacking x,
        # i.e. nearly the whole corpus: an exclusion would become the broadest
        # possible inclusion.
        assert (
            build_fts_tsquery("'chromatin' & 'cancer' & !'gene'")
            == "('chromatin' | 'cancer') & !'gene'"
        )

    def test_negation_inside_an_or_branch_is_not_hoisted(self):
        # "chromatin or cancer -gene" parses as chromatin | (cancer & !gene).
        # The old assemble step produced ('chromatin' | 'cancer') & !'gene',
        # which also dropped passages containing "chromatin" AND "gene" -- an
        # exclusion the user never asked for.
        out = build_fts_tsquery("'chromatin' | 'cancer' & !'gene'")
        assert out == "'chromatin' | (('cancer') & !'gene')"
        assert not out.endswith("& !'gene'")

    def test_negation_only_query_matches_nothing(self):
        # "-cancer" says only what it does not want, so it has no candidate set.
        assert build_fts_tsquery("!'cancer'") == ""

    def test_negations_are_never_pruned(self):
        # Pruning an exclusion would not make the query cheaper, it would make
        # it match more.
        out = build_fts_tsquery("'ctcf' & !'gene'", common=COMMON, min_terms=1)
        assert out == "('ctcf') & !'gene'"


class TestPhrases:
    """Phrase operands are opaque: never pruned, never descended into."""

    def test_phrase_survives_the_or_rewrite(self):
        assert (
            build_fts_tsquery("'enhanc' <-> 'promot' & 'loop'")
            == "('enhanc' <-> 'promot' | 'loop')"
        )

    def test_phrase_is_not_pruned_on_a_member_word(self):
        # 'gene' is common, but "gene expression" as a phrase is highly
        # selective -- pruning it would discard the best evidence in the query.
        out = build_fts_tsquery(
            "'gene' <-> 'express' & 'ctcf'", common=COMMON, min_terms=1
        )
        assert "'gene' <-> 'express'" in out

    def test_distance_operator_survives(self):
        assert (
            build_fts_tsquery("'enhanc' <2> 'promot'") == "('enhanc' <2> 'promot')"
        )


class TestPruning:
    def test_common_terms_are_dropped(self):
        out = build_fts_tsquery(
            "'ctcf' & 'cohesin' & 'gene' & 'cell'", common=COMMON, min_terms=2
        )
        assert out == "('ctcf' | 'cohesin')"

    def test_backstop_restores_the_rarest_dropped_terms(self):
        # Every term is common, so pruning would leave nothing and return zero
        # hits -- strictly worse than being slow. The rarest are restored until
        # min_terms survive: chromatin (24,797) and cell (36,560) before
        # express (50,000) and gene (60,000).
        out = build_fts_tsquery(
            "'gene' & 'express' & 'cell' & 'chromatin'", common=COMMON, min_terms=2
        )
        assert out == "('cell' | 'chromatin')"

    def test_prefix_and_weighted_operands_are_never_pruned(self):
        # 'gene':* is not the key lexeme_df is indexed on, so it has no df to
        # test and must be kept rather than guessed at.
        out = build_fts_tsquery("'gene':* & 'ctcf'", common=COMMON, min_terms=1)
        assert "'gene':*" in out

    def test_no_common_map_prunes_nothing(self):
        # An empty lexeme_df (never rebuilt) must degrade to slow, not to wrong.
        assert (
            build_fts_tsquery("'gene' & 'cell'", common={}) == "('gene' | 'cell')"
        )


class TestParser:
    def test_precedence_is_or_looser_than_and(self):
        tree = parse_tsquery("'a' & 'b' | 'c'")
        # OR at the root with two children, not a three-way AND.
        assert type(tree).__name__ == "_Or"
        assert len(tree.children) == 2

    def test_empty_input_parses_to_none(self):
        assert parse_tsquery("") is None

    @pytest.mark.parametrize("bad", ["'a' &", "('a'", "'a' ) 'b'", "& 'a'"])
    def test_malformed_input_raises(self, bad):
        with pytest.raises(TsqueryParseError):
            parse_tsquery(bad)

    def test_build_degrades_to_the_raw_parse_on_malformed_input(self):
        # Reached from a public endpoint, so answering narrowly beats raising.
        assert build_fts_tsquery("'a' &") == "'a' &"

    def test_escaped_quote_in_a_lexeme_round_trips(self):
        assert build_fts_tsquery("'it''s' & 'ctcf'") == "('it''s' | 'ctcf')"
