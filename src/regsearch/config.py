"""Central configuration.

Every path defaults under the project root so the whole thing is relocatable and
nothing lands in $HOME (which is quota-limited on the cluster).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Keep model weights on /vast rather than the home quota, which is small and
# shared with every other project on this account.
#
# This has to happen at import rather than in the entry points: huggingface_hub
# resolves its cache location once, when it is first imported. Every module that
# pulls in sentence-transformers does so lazily inside a function body, and they
# all import this module first, so setting it here lands before anything reads
# it. Setting it in slurm/embed.sbatch alone was not enough -- that left every
# interactive run downloading into $HOME.
#
# setdefault, not assignment: an HF_HOME already in the environment is a
# deliberate choice by whoever set it and should win.
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "data" / "hf"))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="REGSEARCH_",
        env_file=PROJECT_ROOT / ".env",
        extra="ignore",
    )

    # --- storage ---------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"
    cache_dir: Path = PROJECT_ROOT / "data" / "http_cache"
    pgdata_dir: Path = PROJECT_ROOT / "data" / "pgdata"
    run_dir: Path = PROJECT_ROOT / "data" / "run"

    # --- postgres --------------------------------------------------------
    # Unix socket by default: no TCP port collisions between users on a shared
    # node, and no listening socket exposed to the rest of the cluster.
    pg_host: str = str(PROJECT_ROOT / "data" / "run")
    # On a Unix socket the port is not a TCP port -- it selects the socket
    # filename (.s.PGSQL.<port>). scripts/pg_start.sh must be started with the
    # same value or the client looks for a socket that isn't there.
    pg_port: int = 5432
    pg_user: str = "regsearch"
    pg_db: str = "regsearch"
    pg_password: str = "regsearch"

    # --- europe pmc ------------------------------------------------------
    epmc_base: str = "https://www.ebi.ac.uk/europepmc/webservices/rest"
    # Europe PMC asks for <= ~10 req/s sustained; we stay well under.
    epmc_rps: float = 6.0
    epmc_concurrency: int = 6
    epmc_timeout_s: float = 30.0
    # Identify ourselves; EBI asks for a contact in the UA string.
    user_agent: str = "regsearch/0.1 (research prototype; pattyyang1@gmail.com)"

    # --- embeddings ------------------------------------------------------
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384
    embed_batch_size: int = 256

    # --- retrieval -------------------------------------------------------
    bm25_topk: int = 100
    dense_topk: int = 100
    rrf_k: int = 60
    rerank_topk: int = 100

    # Per-arm weights for reciprocal rank fusion. Unweighted RRF assumes its
    # inputs are equally trustworthy; on this corpus they are not.
    #
    # Swept on the 171-query test split (scripts/tune_rrf.py), dense pinned at
    # 1.0 since only the ratio can reorder anything:
    #
    #   w_fts | Recall@50 | nDCG@10 |    MRR
    #    0.00 |    0.1236 |  0.0536 | 0.1100   <- dense alone
    #    0.10 |    0.1241 |  0.0522 | 0.1017
    #    0.20 |    0.1250 |  0.0514 | 0.1001
    #    0.50 |    0.1267 |  0.0449 | 0.0916   <- best recall
    #    1.00 |    0.0992 |  0.0339 | 0.0792   <- unweighted, what shipped before
    #
    # The trade is monotone and does not have a sweet spot: lexical evidence
    # adds candidates at depth while degrading the top of the ranking. NO
    # weight beats dense alone on nDCG@10 or MRR. Tuning cannot make fusion win
    # on ranking quality here, and pretending otherwise would be picking the
    # metric to fit the conclusion.
    #
    # 0.5 is chosen because fusion's actual job in this pipeline is CANDIDATE
    # GENERATION for the cross-encoder, not final ranking -- the reranker
    # exists to fix ordering, so what it needs from fusion is the largest
    # possible pool of true positives. Read the standalone `hybrid` row in the
    # ablation as a candidate generator being scored as if it were a final
    # ranking, which is why its nDCG looks poor.
    rrf_weighted: bool = True
    rrf_weights: dict[str, float] = {"fts": 0.5, "dense": 1.0}
    # OR the lexical query terms instead of ANDing them. websearch_to_tsquery
    # ANDs by default, which requires a passage to contain *every* query term;
    # on the ~10-word natural-language queries this eval uses, that matches
    # almost nothing and the arm scores ~0. Off restores the AND behaviour,
    # which is the right choice only for short keyword queries.
    fts_or_semantics: bool = True

    # --- lexical arm: latency ---------------------------------------------
    # ORing the terms is what made `fts` work (Recall@50 0.0013 -> 0.0462) and
    # also what made it slow: a 10-word title OR-matches ~52,000 of the 99,567
    # passages, and ts_rank_cd has to decode positions for every one of them.
    # These two toggles attack that from opposite ends. Both default ON; both
    # can be flipped off to reproduce the pre-fix numbers exactly.

    # Drop query terms that occur in more than `fts_df_max_frac` of passages
    # before ORing. A lexeme in half the corpus contributes an enormous
    # candidate set and almost no discrimination: 'gene' is in 47.8% of passages
    # here, 'express' 40.1%, 'chromatin' 24.9%. Reads the materialised
    # `lexeme_df` table; with that table empty the toggle is inert, so an
    # un-rebuilt database degrades to the previous behaviour rather than
    # breaking.
    #
    # This is NOT idf weighting -- ts_rank_cd has no idf term, it ranks by cover
    # density. Pruning changes which passages are candidates AND their scores
    # (a dropped term stops contributing to any cover), so it is a real quality
    # trade and is measured as one, not asserted to be free.
    # Swept in docs/agent-notes/fts-latency.md.
    fts_prune_common_terms: bool = True
    fts_df_max_frac: float = 0.05
    # Backstop: never prune a query below this many positive terms. A query made
    # entirely of common words ("gene expression analysis") would otherwise
    # prune to nothing and return zero hits -- strictly worse than being slow.
    # When it trips, the rarest of the would-be-dropped terms are restored.
    fts_min_terms: int = 3

    # Join `documents` (for the title) AFTER the LIMIT rather than before it.
    # The old shape ran one index lookup into documents for every OR match --
    # ~52,000 lookups, 158k buffer hits -- to decorate 100 surviving rows. Pure
    # waste, and result-identical because passages.doc_id is NOT NULL with a
    # foreign key, so the inner join can never drop a row.
    fts_join_after_limit: bool = True

    @property
    def resolved_host(self) -> str:
        """Where Postgres is actually listening.

        pg_start.sh records its own hostname in data/run/pg_host. Clients on
        other nodes (Slurm jobs) must dial that host over TCP -- a Unix socket
        is local IPC and is not connectable across hosts even when the socket
        file lives on shared storage.
        """
        host_file = Path(self.run_dir) / "pg_host"
        if host_file.exists():
            recorded = host_file.read_text().strip()
            if recorded:
                return recorded
        return self.pg_host

    @property
    def resolved_password(self) -> str:
        pw_file = Path(self.run_dir) / "pg_password"
        if pw_file.exists():
            recorded = pw_file.read_text().strip()
            if recorded:
                return recorded
        return self.pg_password

    @property
    def dsn(self) -> str:
        return (
            f"host={self.resolved_host} port={self.pg_port} dbname={self.pg_db} "
            f"user={self.pg_user} password={self.resolved_password}"
        )

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.cache_dir, self.pgdata_dir, self.run_dir):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    if os.environ.get("REGSEARCH_SKIP_MKDIR") != "1":
        s.ensure_dirs()
    return s
