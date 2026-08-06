"""regsearch CLI."""

from __future__ import annotations

import logging

import typer
from rich.console import Console
from rich.table import Table

from regsearch.db import client as db

app = typer.Typer(add_completion=False, help="regsearch: hybrid retrieval over regulatory genomics literature")
console = Console()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


@app.command("init-db")
def init_db(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Create tables and indexes. Idempotent."""
    _setup_logging(verbose)
    db.apply_schema()
    console.print("[green]schema applied[/green]")


@app.command()
def ingest(
    max_results: int = typer.Option(2000, help="Max records per seed query."),
    query: list[str] = typer.Option(None, "--query", "-q", help="Override seed queries."),
    refetch: bool = typer.Option(False, help="Re-ingest queries already in ingest_log."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Pull records from Europe PMC into the database."""
    _setup_logging(verbose)
    from regsearch.ingest.pipeline import run_ingest

    db.apply_schema()
    totals = run_ingest(
        queries=list(query) if query else None,
        max_results=max_results,
        skip_seen=not refetch,
    )
    console.print(
        f"[green]ingested[/green] {totals['documents']} documents, "
        f"{totals['passages']} passages "
        f"({totals['skipped']} queries skipped as already done)"
    )


@app.command()
def stats() -> None:
    """Show corpus counts."""
    t = Table(title="regsearch corpus", show_header=False)
    t.add_column("metric", style="cyan")
    t.add_column("count", justify="right")
    for k, v in db.stats().items():
        t.add_row(k.replace("_", " "), f"{v:,}")
    console.print(t)


@app.command()
def embed(
    batch_size: int = typer.Option(256, help="Encoder batch size."),
    limit: int = typer.Option(None, help="Stop after N passages (smoke tests)."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Encode passages that have no embedding yet. Resumable."""
    _setup_logging(verbose)
    from regsearch.retrieve.embed import embed_corpus

    n = embed_corpus(batch_size=batch_size, limit=limit)
    console.print(f"[green]embedded[/green] {n:,} passages")


@app.command()
def search(
    query: str = typer.Argument(..., help="Query text."),
    arm: str = typer.Option("hybrid", help="fts | dense | hybrid | hybrid_rerank"),
    k: int = typer.Option(10, "-k", help="Results to show."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run one retrieval arm and print the ranked hits."""
    _setup_logging(verbose)
    from regsearch.retrieve.search import search as run_search

    resp = run_search(query, arm=arm, k=k)  # type: ignore[arg-type]
    t = Table(title=f"{resp.arm}  ·  {resp.latency_ms:.1f} ms  ·  {query!r}")
    t.add_column("#", justify="right", style="dim")
    t.add_column("score", justify="right")
    t.add_column("title", max_width=48)
    t.add_column("passage", max_width=64)
    for h in resp.hits:
        t.add_row(str(h.rank), f"{h.score:.4f}", h.title, h.text[:200])
    console.print(t)


@app.command("build-index")
def build_index(
    m: int = typer.Option(16, help="HNSW m."),
    ef_construction: int = typer.Option(64, help="HNSW ef_construction."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Build the HNSW vector index. Run after embedding, not before."""
    _setup_logging(verbose)
    db.build_vector_index(m=m, ef_construction=ef_construction)
    db.analyze()
    console.print("[green]index built[/green]")


if __name__ == "__main__":
    app()
