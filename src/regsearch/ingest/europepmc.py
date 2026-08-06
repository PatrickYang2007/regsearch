"""Async Europe PMC client.

Three things this handles that a naive `requests.get` loop does not:

1. Rate limiting  -- a token bucket capped at settings.epmc_rps. EBI does not
   publish a hard limit but throttles aggressively; we stay well under.
2. Retries        -- exponential backoff on 429/5xx/timeouts via tenacity.
                     4xx other than 429 are NOT retried (they won't fix
                     themselves and retrying just burns the budget).
3. Disk cache     -- responses keyed by a hash of the request. Re-running
                     ingest after a crash costs no API calls, which matters
                     when a full crawl is tens of thousands of requests.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import orjson
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from regsearch.config import Settings, get_settings

log = logging.getLogger(__name__)


class RetryableHTTPError(Exception):
    """429 / 5xx -- worth another attempt."""


@dataclass
class RateLimiter:
    """Token bucket. Shared across all coroutines using one client."""

    rate: float
    capacity: float = 1.0
    _tokens: float = field(default=0.0, init=False)
    _last: float = field(default_factory=time.monotonic, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def __post_init__(self) -> None:
        self.capacity = max(self.capacity, self.rate)
        self._tokens = self.capacity

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                self._tokens = min(
                    self.capacity, self._tokens + (now - self._last) * self.rate
                )
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                await asyncio.sleep((1.0 - self._tokens) / self.rate)


class DiskCache:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        h = hashlib.sha256(key.encode()).hexdigest()
        # Two-level fan-out: a single flat dir with 100k entries is painful on
        # network filesystems.
        d = self.root / h[:2] / h[2:4]
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{h}.json"

    def get(self, key: str) -> Any | None:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            return orjson.loads(p.read_bytes())
        except orjson.JSONDecodeError:
            p.unlink(missing_ok=True)  # truncated by an interrupted write
            return None

    def put(self, key: str, value: Any) -> None:
        p = self._path(key)
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(orjson.dumps(value))
        tmp.replace(p)  # atomic: readers never see a partial file


class EuropePMCClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.s = settings or get_settings()
        self.cache = DiskCache(self.s.cache_dir / "epmc")
        self.limiter = RateLimiter(rate=self.s.epmc_rps)
        self.sem = asyncio.Semaphore(self.s.epmc_concurrency)
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> EuropePMCClient:
        self._client = httpx.AsyncClient(
            base_url=self.s.epmc_base,
            timeout=self.s.epmc_timeout_s,
            headers={"User-Agent": self.s.user_agent, "Accept": "application/json"},
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @retry(
        retry=retry_if_exception_type(
            (RetryableHTTPError, httpx.TimeoutException, httpx.TransportError)
        ),
        wait=wait_exponential_jitter(initial=1, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        assert self._client is not None, "use `async with EuropePMCClient()`"
        key = f"{path}?{orjson.dumps(params, option=orjson.OPT_SORT_KEYS).decode()}"

        cached = self.cache.get(key)
        if cached is not None:
            return cached

        async with self.sem:
            await self.limiter.acquire()
            resp = await self._client.get(path, params=params)

        if resp.status_code == 429 or resp.status_code >= 500:
            raise RetryableHTTPError(f"{resp.status_code} for {path}")
        resp.raise_for_status()

        data = resp.json()
        self.cache.put(key, data)
        return data

    # ------------------------------------------------------------- searching
    async def search_page(
        self, query: str, cursor: str = "*", page_size: int = 1000
    ) -> dict[str, Any]:
        return await self._get(
            "/search",
            {
                "query": query,
                "format": "json",
                "pageSize": page_size,
                "cursorMark": cursor,
                # `core` is what carries abstractText; the default `lite` does not.
                "resultType": "core",
            },
        )

    async def search_all(
        self, query: str, max_results: int = 5000, page_size: int = 1000
    ) -> list[dict[str, Any]]:
        """Cursor-paginate a query up to max_results."""
        out: list[dict[str, Any]] = []
        cursor = "*"
        while len(out) < max_results:
            data = await self.search_page(query, cursor=cursor, page_size=page_size)
            results = data.get("resultList", {}).get("result", [])
            if not results:
                break
            out.extend(results)
            nxt = data.get("nextCursorMark")
            # Europe PMC repeats the cursor on the final page rather than
            # omitting it -- without this check we'd loop forever.
            if not nxt or nxt == cursor:
                break
            cursor = nxt
        return out[:max_results]

    async def references(self, source: str, ext_id: str) -> list[dict[str, Any]]:
        """Works this record cites."""
        data = await self._get(
            f"/{source}/{ext_id}/references",
            {"format": "json", "pageSize": 1000},
        )
        return data.get("referenceList", {}).get("reference", [])


# ------------------------------------------------------------ normalisation
def normalise_record(rec: dict[str, Any]) -> dict[str, Any] | None:
    """Europe PMC record -> a `documents` row. Returns None if unusable."""
    title = (rec.get("title") or "").strip()
    if not title:
        return None

    year_raw = rec.get("pubYear")
    try:
        pub_year = int(year_raw) if year_raw else None
    except (TypeError, ValueError):
        pub_year = None

    try:
        cited_by = int(rec.get("citedByCount") or 0)
    except (TypeError, ValueError):
        cited_by = 0

    return {
        "source": rec.get("source") or "MED",
        "ext_id": str(rec.get("id") or "").strip(),
        "pmid": rec.get("pmid"),
        "pmcid": rec.get("pmcid"),
        "doi": rec.get("doi"),
        "title": title,
        "abstract": (rec.get("abstractText") or "").strip() or None,
        "journal": (rec.get("journalInfo", {}) or {}).get("journal", {}).get("title")
        or rec.get("journalTitle"),
        "pub_year": pub_year,
        "cited_by": cited_by,
        "is_open_access": (rec.get("isOpenAccess") == "Y"),
    }
