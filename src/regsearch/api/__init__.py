"""HTTP serving layer.

Kept in its own package so `fastapi` stays an optional dependency: importing
regsearch for ingest, eval, or the CLI must not require the serve extra.
"""
