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


@app.command("harvest-citations")
def harvest_citations(
    limit: int = typer.Option(2000, help="Max citing documents to process."),
    min_refs: int = typer.Option(3, help="Skip docs with fewer resolvable refs."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Harvest weak relevance labels from the citation graph."""
    _setup_logging(verbose)
    from regsearch.ingest.citations import run_harvest

    res = run_harvest(limit=limit, min_refs=min_refs)
    console.print(
        f"[green]harvested[/green] {res['pairs']:,} pairs from {res['documents']:,} documents"
    )


@app.command("build-evalset")
def build_evalset(
    test_frac: float = typer.Option(0.2, help="Fraction of citing docs held out."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Turn harvested citations into queries + qrels (split by citing doc)."""
    _setup_logging(verbose)
    from regsearch.eval.build import build_citation_evalset

    res = build_citation_evalset(test_frac=test_frac)
    console.print(
        f"[green]built[/green] {res['queries']:,} queries / {res['qrels']:,} qrels"
    )


@app.command("dedup-docs")
def dedup_docs(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Cluster duplicate records (preprint/published twins) onto a canonical id."""
    _setup_logging(verbose)
    db.apply_schema()
    res = db.rebuild_canonical_docs()
    console.print(
        f"[green]clustered[/green] {res['merged']:,} duplicate documents "
        f"into {res['clusters']:,} canonical records"
    )


@app.command("eval")
def eval_cmd(
    split: str = typer.Option("test"),
    origin: str = typer.Option("citation", help="citation | manual"),
    arms: str = typer.Option("fts,dense,hybrid,hybrid_rerank"),
    out: str = typer.Option(None, help="Write the table to a markdown file."),
    canonicalize: bool = typer.Option(
        True,
        "--canonicalize/--no-canonicalize",
        help="Score duplicate records (preprint/published) as one document.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the ablation across retrieval arms."""
    _setup_logging(verbose)
    from regsearch.eval.harness import run_ablation

    results = run_ablation(
        arms=[a.strip() for a in arms.split(",") if a.strip()],  # type: ignore[arg-type]
        split=split,
        origin=origin,
        canonicalize=canonicalize,
    )

    t = Table(title=f"ablation · split={split} · origin={origin} · n={results[0].n_queries}")
    for col, just in [
        ("arm", "left"), ("Recall@50", "right"), ("nDCG@10", "right"),
        ("MRR", "right"), ("p50 ms", "right"), ("p95 ms", "right"),
    ]:
        t.add_column(col, justify=just)  # type: ignore[arg-type]
    for r in results:
        t.add_row(
            r.arm, f"{r.recall_at_50:.4f}", f"{r.ndcg_at_10:.4f}",
            f"{r.mrr:.4f}", f"{r.latency_p50_ms:.1f}", f"{r.latency_p95_ms:.1f}",
        )
    console.print(t)

    if out:
        from pathlib import Path

        lines = [
            f"| arm | Recall@50 | nDCG@10 | MRR | p50 ms | p95 ms |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        lines += [
            f"| `{r.arm}` | {r.recall_at_50:.4f} | {r.ndcg_at_10:.4f} | "
            f"{r.mrr:.4f} | {r.latency_p50_ms:.1f} | {r.latency_p95_ms:.1f} |"
            for r in results
        ]
        lines.append("")
        lines.append(
            f"_n={results[0].n_queries} queries, split={split}, origin={origin}, "
            f"canonicalize={canonicalize}._"
        )
        if origin == "citation":
            # The table must carry its own caveat: detached from this footer it
            # reads as a relevance measurement, which it is not.
            lines.append("")
            lines.append(
                "**Weak supervision.** Labels are citation-derived, not human "
                "relevance judgements: a query is a paper's title and its "
                "positives are the papers it cites. The same signal trains the "
                "reranker, so `hybrid_rerank` rows are scored against the "
                "family of labels they learn from. Not a human-judged result."
            )
        Path(out).write_text("\n".join(lines) + "\n")
        console.print(f"[green]wrote[/green] {out}")


@app.command("build-lexeme-stats")
def build_lexeme_stats(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Materialise per-lexeme document frequency for the lexical arm.

    The `fts` arm drops near-corpus-wide query terms before ORing them, which
    needs to know how common each lexeme is. Without this table the pruning is
    inert -- the arm still works, just slower and slightly worse -- so a fresh
    database degrades rather than breaks.

    Re-run after ingesting more documents: the stored frequencies are relative
    to the corpus size at build time, so a grown corpus makes them stale.
    """
    _setup_logging(verbose)
    db.apply_schema()
    res = db.rebuild_lexeme_df()
    console.print(
        f"[green]built[/green] document frequencies for {res['lexemes']:,} lexemes "
        f"over {res['n_passages']:,} passages"
    )


@app.command("export-pool")
def export_pool(
    out: str = typer.Option("data/judging_pool.csv", help="CSV to write."),
    depth: int = typer.Option(10, help="Top-n taken from each arm."),
    arms: str = typer.Option("fts,dense,hybrid", help="Arms to pool from."),
    limit: int = typer.Option(None, help="Use only the first N queries."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Emit pooled candidates for HUMAN relevance judging.

    Pooled across arms, not taken from one: judging a single arm's output biases
    the qrels toward it, because anything only the other arms found is scored
    irrelevant purely for never having been looked at.

    The `relevance` column is left blank on purpose. Fill it in by hand with
    0-3, then load it back with `regsearch import-qrels`. Do not generate these
    labels with a model -- the reranker trains on citation-derived weak
    supervision, and a model-labelled test set would be scored against the same
    family of signal it learned from, which measures nothing.
    """
    _setup_logging(verbose)
    from pathlib import Path

    from regsearch.eval.build import export_pool_for_judging
    from regsearch.eval.manual_queries import get_manual_queries

    queries = get_manual_queries(limit)
    n = export_pool_for_judging(
        queries=queries,
        out_path=Path(out),
        arms=[a.strip() for a in arms.split(",") if a.strip()],  # type: ignore[arg-type]
        depth=depth,
    )
    console.print(
        f"[green]wrote[/green] {n:,} candidates for {len(queries)} queries to {out}\n"
        f"Fill the [cyan]relevance[/cyan] column (0-3) by hand, then run: "
        f"[cyan]regsearch import-qrels {out}[/cyan]"
    )


@app.command("import-qrels")
def import_qrels_cmd(
    csv_path: str = typer.Argument(..., help="Hand-filled judging CSV."),
    split: str = typer.Option("test", help="Split to file these judgements under."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Load hand-filled judgements back as origin='manual'.

    Rows with an empty relevance cell are SKIPPED, not treated as irrelevant --
    "nobody judged this" and "a human judged this irrelevant" are different
    facts, and conflating them silently invents negative labels.
    """
    _setup_logging(verbose)
    from pathlib import Path

    from regsearch.eval.build import import_qrels

    res = import_qrels(Path(csv_path), split=split)
    console.print(
        f"[green]imported[/green] {res['qrels']:,} judgements as origin='manual'\n"
        f"Evaluate them with: [cyan]regsearch eval --origin manual[/cyan]"
    )


@app.command("train-reranker")
def train_reranker_cmd(
    epochs: int = typer.Option(1),
    batch_size: int = typer.Option(32),
    lr: float = typer.Option(2e-5),
    max_negatives_per_query: int = typer.Option(8),
    limit: int = typer.Option(None, help="Use only the first N training queries."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Fine-tune the cross-encoder on citation-derived weak supervision.

    Wants a GPU: `sbatch slurm/finetune_rerank.sbatch`. On CPU this is hours.

    Writes to data/models/reranker/, which `retrieve.rerank` prefers over the
    public checkpoint automatically -- so a successful run silently changes what
    the `hybrid_rerank` arm means. Re-run `regsearch eval` afterwards.
    """
    _setup_logging(verbose)
    # Imported inside the body: train_rerank pulls in torch and
    # sentence-transformers, and a module-level import would make every
    # `regsearch stats` on a login node pay for them.
    from regsearch.retrieve.train_rerank import train_reranker

    meta = train_reranker(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        max_negatives_per_query=max_negatives_per_query,
        limit=limit,
    )
    console.print(f"[green]trained[/green] -> {meta.get('output_dir')}")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind address."),
    port: int = typer.Option(8000),
    reload: bool = typer.Option(False, help="Auto-reload on code change (dev only)."),
) -> None:
    """Run the HTTP API.

    Loopback by default. On a shared cluster node, binding 0.0.0.0 exposes an
    unauthenticated search service to every other user on the box.
    """
    # Imported inside the body so `fastapi`/`uvicorn` are only required when
    # actually serving -- they live in the `serve` extra precisely so the
    # ingestion path stays installable without them.
    from regsearch.api.app import run

    run(host=host, port=port, reload=reload)


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
