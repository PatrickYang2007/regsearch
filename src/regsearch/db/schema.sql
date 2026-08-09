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

-- One paper can enter the corpus several times: a bioRxiv preprint (source PPR)
-- and its published version (MED) are distinct Europe PMC records with distinct
-- DOIs, and patent families repeat wholesale. 1,335 such clusters exist here,
-- 7% of the corpus.
--
-- This matters for evaluation, not storage. A qrel naming the published record
-- scores a *miss* when retrieval surfaces the preprint, and both twins consume
-- separate slots in the top-k -- so every arm is deflated, and not equally,
-- which distorts the arm-vs-arm comparison the whole project exists to make.
--
-- Rather than delete rows (which would throw away real records and break the
-- citation graph's foreign keys), each document points at a cluster
-- representative and evaluation scores at cluster level. NULL means "no
-- duplicate known"; readers should COALESCE to doc_id.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS canonical_doc_id BIGINT
    REFERENCES documents(doc_id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS documents_canonical_idx ON documents (canonical_doc_id)
    WHERE canonical_doc_id IS NOT NULL;

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

-- ---------------------------------------------------------- lexeme statistics
-- Document frequency per lexeme, materialised once from ts_stat.
--
-- Why it exists: the lexical arm ORs its query terms, because ANDing the ~10
-- content words of a paper title matched 1 passage out of 99,567. ORing fixed
-- recall and cost ~250x latency, because ts_rank_cd then has to score every
-- match -- ~52k passages for a typical title query. Most of that candidate set
-- is contributed by a few terms that are nearly corpus-wide: on this corpus
-- 'gene' occurs in 47.8% of passages, 'express' 40.1%, 'chromatin' 24.9%. Those
-- terms select half the corpus and distinguish almost nothing, so the query
-- builder drops them from the OR.
--
-- ts_stat is a full scan of every tsvector, so it is materialised rather than
-- run per query. It is a STATISTIC, not a constraint: a stale table costs
-- ranking quality, never correctness, so it is rebuilt explicitly (alongside
-- ANALYZE) rather than maintained by a trigger on every insert.
--
-- All lexemes are stored, not just the common ones, so the pruning threshold
-- can be re-tuned without a rebuild.
CREATE TABLE IF NOT EXISTS lexeme_df (
    lexeme   TEXT PRIMARY KEY,
    ndoc     INT NOT NULL,      -- passages containing the lexeme at least once
    -- ndoc / total passages AT BUILD TIME. Stored rather than derived so the
    -- threshold means the same thing after the corpus grows and the table has
    -- not been rebuilt -- a silently shifting denominator would change which
    -- terms get pruned without anything being re-measured.
    df_frac  REAL NOT NULL,
    built_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS lexeme_df_frac_idx ON lexeme_df (df_frac DESC);

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
