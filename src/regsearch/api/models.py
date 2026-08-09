"""Pydantic response models for the HTTP layer.

These exist so the OpenAPI schema is generated from real types rather than from
bare dicts -- a client can read /openapi.json and know what comes back. They
mirror the dataclasses in regsearch.retrieve.search; the conversion is one
function per type, at the edge, so the retrieval layer stays free of HTTP
concerns.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from regsearch.retrieve.search import Hit, SearchResponse


class Arm(str, Enum):
    """The four retrieval strategies, identical to the eval harness's arms.

    An Enum rather than a plain str parameter: FastAPI renders it as an OpenAPI
    enum and rejects anything else with 422 *before* the handler runs, so an
    unvalidated string can never reach search() and hit its ValueError as a 500.
    Values must stay in step with the Arm Literal in retrieve.search -- there is
    a test that fails if they drift.
    """

    fts = "fts"
    dense = "dense"
    hybrid = "hybrid"
    hybrid_rerank = "hybrid_rerank"


class HitModel(BaseModel):
    rank: int = Field(description="1-based position in this result list.")
    passage_id: int
    doc_id: int
    title: str = Field(description="Title of the document the passage came from.")
    text: str = Field(description="The passage itself, as chunked at ingest.")
    score: float = Field(
        description=(
            "Arm-specific and not comparable across arms: ts_rank_cd for fts, "
            "cosine similarity for dense, summed reciprocal ranks for hybrid, "
            "cross-encoder logit for hybrid_rerank. Higher is better in all four."
        )
    )
    components: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Provenance carried by Hit: which underlying arms produced this "
            "passage and what each scored it. `fts` and `dense` are the "
            "contributing arms -- a single-arm request carries one, a fused hit "
            "carries one per arm that found it (an arm missing from the map did "
            "not retrieve this passage). `hybrid_rerank` adds two more keys that "
            "are stages rather than arms: `rrf` preserves the fusion score the "
            "cross-encoder replaced, and `rerank` is the cross-encoder logit, so "
            "the reordering can be inspected after the fact."
        ),
    )

    @classmethod
    def from_hit(cls, h: Hit) -> HitModel:
        return cls(
            rank=h.rank,
            passage_id=h.passage_id,
            doc_id=h.doc_id,
            title=h.title,
            text=h.text,
            score=h.score,
            components=dict(h.components),
        )


class SearchResult(BaseModel):
    query: str
    arm: Arm
    k: int = Field(description="Number of hits requested.")
    latency_ms: float = Field(
        description="Server-side wall clock for the retrieval call itself."
    )
    hits: list[HitModel]

    @classmethod
    def from_response(cls, resp: SearchResponse, k: int) -> SearchResult:
        return cls(
            query=resp.query,
            arm=Arm(resp.arm),
            k=k,
            latency_ms=round(resp.latency_ms, 2),
            hits=[HitModel.from_hit(h) for h in resp.hits],
        )


class PassageModel(BaseModel):
    passage_id: int
    ordinal: int = Field(description="Position of the chunk within its document.")
    section: str
    text: str


class DocumentModel(BaseModel):
    doc_id: int
    source: str = Field(
        description=(
            "Europe PMC source database. Eight occur in this corpus: MED "
            "(PubMed, 15,256), PPR (preprints, 4,226), AGR (169), PAT "
            "(patents, 49), ETH (theses, 48), PMC (34), CBA (5), CTX (4). "
            "Treat it as an open string, not an enum."
        )
    )
    ext_id: str
    pmid: str | None = None
    pmcid: str | None = None
    doi: str | None = None
    title: str
    abstract: str | None = None
    journal: str | None = None
    pub_year: int | None = None
    cited_by: int = 0
    is_open_access: bool = False
    canonical_doc_id: int | None = Field(
        default=None,
        description=(
            "Cluster representative for duplicate records (a preprint and its "
            "published twin are separate Europe PMC records). Equal to doc_id "
            "when this document is its own representative, which is the common "
            "case -- on the current corpus 18,409 of 19,791 documents are "
            "self-referential and 1,382 point elsewhere. NULL only where "
            "clustering could not apply: no title, or a normalised title under "
            "10 characters, which is too generic to be evidence of identity. "
            "Treat NULL as the document itself."
        ),
    )
    passages: list[PassageModel]


class HealthResponse(BaseModel):
    status: str = Field(description="'ok' when the database answered, else 'degraded'.")
    version: str
    database: bool = Field(description="Whether the corpus database was reachable.")
    counts: dict[str, int] | None = Field(
        default=None,
        description="Corpus counts, or null when the database is unreachable.",
    )
    detail: str | None = Field(
        default=None, description="Error text when database is false."
    )
