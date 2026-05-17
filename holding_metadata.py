"""
Holding metadata resolution — sector & country lookup.

Layered approach:
1. Static mapping table (instant, covers MSCI ACWI top constituents)
2. Persistent file cache (instant, grows over time)
3. yfinance fallback (slow, only for cache misses)

Only the top N holdings are queried via yfinance to keep analysis fast.
"""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import yfinance as yf

from static_metadata import STATIC_METADATA

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache configuration
# ---------------------------------------------------------------------------

_CACHE_PATH = os.path.join(os.path.dirname(__file__), ".holdings_meta_cache.json")

# Maximum number of holdings to query via yfinance per analysis run
MAX_YFINANCE_LOOKUPS = 50


def _load_cache() -> dict[str, dict[str, str]]:
    """Load the holdings metadata cache from disk."""
    if not os.path.exists(_CACHE_PATH):
        return {}
    try:
        with open(_CACHE_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
        return {}
    except Exception as exc:
        logger.debug("Metadata cache load failed: %s", exc)
        return {}


def _save_cache(cache: dict[str, dict[str, str]]) -> None:
    """Persist the holdings metadata cache to disk."""
    try:
        with open(_CACHE_PATH, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, indent=2)
    except Exception as exc:
        logger.debug("Metadata cache save failed: %s", exc)


def _fetch_yfinance_metadata(ticker: str) -> dict[str, str] | None:
    """Fetch sector and country from yfinance for a single ticker.

    Returns {"sector": ..., "country": ...} or None on failure.
    """
    try:
        info = yf.Ticker(ticker).info
        sector = info.get("sector")
        country = info.get("country")
        if sector and country:
            return {"sector": sector, "country": country}
    except Exception:
        pass
    return None


def _lookup_static(ticker: str) -> dict[str, str] | None:
    """Look up ticker in the static mapping table.

    Tries exact match first, then without exchange suffix.
    """
    # Exact match
    if ticker in STATIC_METADATA:
        return STATIC_METADATA[ticker]

    # Try without exchange suffix (e.g. "AAPL.L" -> "AAPL")
    base_ticker = ticker.split(".")[0] if "." in ticker else None
    if base_ticker and base_ticker in STATIC_METADATA:
        return STATIC_METADATA[base_ticker]

    return None


def fetch_holdings_metadata(
    tickers: list[str],
    progress_callback=None,
) -> dict[str, dict[str, str]]:
    """Resolve sector and country for a list of holding tickers.

    Uses the layered approach: static table -> file cache -> yfinance.
    Only queries yfinance for the first MAX_YFINANCE_LOOKUPS cache misses.

    Parameters
    ----------
    tickers : list[str]
        List of holding tickers to resolve.
    progress_callback : callable, optional
        Called with (completed, total) for yfinance lookups.

    Returns
    -------
    dict[str, dict[str, str]]
        Mapping of ticker -> {"sector": str, "country": str}.
        Tickers that couldn't be resolved are omitted.
    """
    cache = _load_cache()
    results: dict[str, dict[str, str]] = {}
    to_fetch: list[str] = []

    # Layer 1 & 2: static table and cache
    for ticker in tickers:
        # Check static table first
        static_hit = _lookup_static(ticker)
        if static_hit:
            results[ticker] = static_hit
            continue

        # Check file cache
        if ticker in cache:
            results[ticker] = cache[ticker]
            continue

        # Also check cache with base ticker
        base = ticker.split(".")[0] if "." in ticker else None
        if base and base in cache:
            results[ticker] = cache[base]
            continue

        to_fetch.append(ticker)

    # Layer 3: yfinance fallback (limited to MAX_YFINANCE_LOOKUPS)
    to_fetch = to_fetch[:MAX_YFINANCE_LOOKUPS]

    if to_fetch:
        cache_updated = False

        def _fetch_one(t: str) -> tuple[str, dict[str, str] | None]:
            return t, _fetch_yfinance_metadata(t)

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(_fetch_one, t): t for t in to_fetch}
            completed = 0
            for future in as_completed(futures):
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(to_fetch))
                try:
                    ticker, meta = future.result()
                    if meta:
                        results[ticker] = meta
                        cache[ticker] = meta
                        cache_updated = True
                except Exception:
                    pass

        if cache_updated:
            _save_cache(cache)

    return results
