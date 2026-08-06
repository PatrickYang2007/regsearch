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
