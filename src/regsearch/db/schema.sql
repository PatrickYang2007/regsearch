-- regsearch schema. Idempotent: safe to re-run.
--
-- Design notes
-- ------------
-- * documents  : one row per Europe PMC record (abstract-level metadata).
-- * passages   : retrieval unit. Abstracts are chunked into overlapping
--                sentence windows, because whole-abstract embeddings wash out
--                the specific claim a query is actually about.
-- * The lexical arm uses Postgres FTS (ts_rank_cd), which is a length-normalised
--   TF-IDF variant -- NOT true BM25 (no k1/b saturation terms). It is a fair
--   lexical baseline and the eval reports it honestly as "fts". See README for
--   the upgrade path to real BM25.

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------- documents
CREATE TABLE IF NOT EXISTS documents (
    doc_id        BIGSERIAL PRIMARY KEY,
    source        TEXT NOT NULL,              -- MED | PMC | PPR (Europe PMC source)
    ext_id        TEXT NOT NULL,              -- id within that source
    pmid          TEXT,
    pmcid         TEXT,
    doi           TEXT,
    title         TEXT NOT NULL,
    abstract      TEXT,
    journal       TEXT,
    pub_year      INT,
    cited_by      INT DEFAULT 0,
    is_open_access BOOLEAN DEFAULT FALSE,
    ingested_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source, ext_id)
);

CREATE INDEX IF NOT EXISTS documents_pmid_idx  ON documents (pmid) WHERE pmid IS NOT NULL;
CREATE INDEX IF NOT EXISTS documents_pmcid_idx ON documents (pmcid) WHERE pmcid IS NOT NULL;
CREATE INDEX IF NOT EXISTS documents_year_idx  ON documents (pub_year);

-- ----------------------------------------------------------------- passages
CREATE TABLE IF NOT EXISTS passages (
    passage_id  BIGSERIAL PRIMARY KEY,
    doc_id      BIGINT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    ordinal     INT NOT NULL,                 -- position of the chunk in the doc
    section     TEXT NOT NULL DEFAULT 'abstract',
    text        TEXT NOT NULL,
    -- Generated column: FTS vector is always in sync with text, no trigger and
    -- no way for an ingest bug to leave the index stale.
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    embedding   vector(384),                  -- NULL until the embed step runs
    UNIQUE (doc_id, ordinal)
);

CREATE INDEX IF NOT EXISTS passages_tsv_idx ON passages USING GIN (tsv);
CREATE INDEX IF NOT EXISTS passages_doc_idx ON passages (doc_id);

-- HNSW for cosine distance. Created in build_indexes() AFTER bulk load --
-- building it on an empty table then inserting is markedly slower than
-- inserting first and indexing once.
-- CREATE INDEX passages_embedding_idx ON passages
--   USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=64);

-- -------------------------------------------------------- citation contexts
-- Weak supervision for the reranker: a sentence in doc A that cites doc B is
-- treated as a query whose positive is B. Free training data, no hand labels.
CREATE TABLE IF NOT EXISTS citation_contexts (
    ctx_id        BIGSERIAL PRIMARY KEY,
    citing_doc_id BIGINT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    cited_doc_id  BIGINT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    context_text  TEXT NOT NULL               -- the citing sentence == pseudo-query
);

-- Dedup on the hash rather than the full text: a table-level UNIQUE constraint
-- cannot contain an expression, and context_text can exceed the btree row
-- limit. This index is also what ON CONFLICT targets on insert.
CREATE UNIQUE INDEX IF NOT EXISTS citation_contexts_uniq_idx
    ON citation_contexts (citing_doc_id, cited_doc_id, md5(context_text));

CREATE INDEX IF NOT EXISTS citation_contexts_cited_idx ON citation_contexts (cited_doc_id);

-- ------------------------------------------------------------- eval harness
CREATE TABLE IF NOT EXISTS eval_queries (
    query_id   BIGSERIAL PRIMARY KEY,
    query_text TEXT NOT NULL UNIQUE,
    -- 'citation' = auto-derived weak label; 'manual' = hand-judged.
    -- Kept apart so the headline numbers can be reported on manual only.
    origin     TEXT NOT NULL DEFAULT 'manual',
    split      TEXT NOT NULL DEFAULT 'test'   -- train | dev | test
);

CREATE TABLE IF NOT EXISTS eval_qrels (
    query_id   BIGINT NOT NULL REFERENCES eval_queries(query_id) ON DELETE CASCADE,
    doc_id     BIGINT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    relevance  INT NOT NULL DEFAULT 1,        -- graded: 0 irrelevant .. 3 perfect
    PRIMARY KEY (query_id, doc_id)
);

-- --------------------------------------------------------------- ingest log
-- Lets ingest resume without re-hitting the API for queries already pulled.
CREATE TABLE IF NOT EXISTS ingest_log (
    query       TEXT PRIMARY KEY,
    n_docs      INT NOT NULL,
    fetched_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
