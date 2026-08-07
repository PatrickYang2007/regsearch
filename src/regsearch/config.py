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
    # OR the lexical query terms instead of ANDing them. websearch_to_tsquery
    # ANDs by default, which requires a passage to contain *every* query term;
    # on the ~10-word natural-language queries this eval uses, that matches
    # almost nothing and the arm scores ~0. Off restores the AND behaviour,
    # which is the right choice only for short keyword queries.
    fts_or_semantics: bool = True

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
