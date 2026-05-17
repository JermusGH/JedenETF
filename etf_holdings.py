"""
ETF Holdings Scraper — Auto-discovery and live fetching.

Supports iShares (BlackRock), Vanguard, Invesco, Xtrackers (DWS),
and Amundi (via justETF fallback).
Automatically discovers ISIN, provider, and provider-specific parameters
from a Yahoo Finance ticker. All discovered data is cached locally.
"""

import json
import logging
import os
import re
from io import StringIO

import pandas as pd
import requests

from models import FetchResult
from validation import is_valid_isin

try:
    from curl_cffi import requests as curl_requests
except ImportError:
    curl_requests = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".etf_cache.json")

# Shared browser User-Agent used across all HTTP requests.
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

_HTTP_HEADERS = {"User-Agent": _USER_AGENT}

_YF_EXCHANGE_TO_FT = {
    "L": ("LSE", "USD"),
    "DE": ("GER", "EUR"),
    "AS": ("AMS", "EUR"),
    "PA": ("PAR", "EUR"),
    "MI": ("MIL", "EUR"),
    "SW": ("SWX", "CHF"),
}

# iShares page-level AJAX identifier (stable across sessions; used in the
# CSV download URL pattern on ishares.com/uk).
# Last verified: 2024-06-20
_ISHARES_AJAX_ID = "1506575576011"
_ISHARES_BASE_URL = "https://www.ishares.com/uk/individual/en/products"

_VANGUARD_GRAPHQL_URL = "https://www.nl.vanguard/gpx/graphql"
_VANGUARD_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Origin": "https://www.nl.vanguard",
    "Referer": "https://www.nl.vanguard/professional/product/etf/equity/",
    "x-consumer-id": "nl-ui",
}

_INVESCO_HOLDINGS_API = (
    "https://dng-api.invesco.com/cache/v1/accounts/en_GB/shareclasses"
)
_INVESCO_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.invesco.com/uk/en/financial-products/etfs.html",
}

_XTRACKERS_BASE_URL = "https://etf.dws.com/en-gb"
_XTRACKERS_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept": "application/json, text/html, */*",
    "Referer": "https://etf.dws.com/en-gb/",
}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_cache() -> dict:
    """Load the discovery cache from disk.

    Returns an empty dict if the file does not exist, contains malformed JSON,
    is empty, or cannot be read for any reason.
    """
    if not os.path.exists(_CACHE_PATH):
        return {}
    try:
        with open(_CACHE_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
        return {}
    except Exception as exc:
        logger.debug("Cache load failed: %s", exc)
        return {}


def _save_cache(cache: dict) -> None:
    """Persist the discovery cache to disk.

    Silently continues on any write failure so that analysis is never
    interrupted by a caching error.
    """
    try:
        with open(_CACHE_PATH, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, indent=2)
    except Exception as exc:
        logger.debug("Cache save failed: %s", exc)


# ---------------------------------------------------------------------------
# Shared HTML table parser
# ---------------------------------------------------------------------------

def _parse_holdings_from_html(html: str, provider_label: str = "") -> pd.DataFrame | None:
    """Extract holdings (name + weight) from an HTML page containing tables.

    Searches for tables with at least 5 rows where one column contains
    percentage-like values and another contains company names.

    Parameters
    ----------
    html : str
        Raw HTML content of the page.
    provider_label : str
        Label for log messages (e.g. "Invesco", "Xtrackers").

    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns: ticker, name, weight. None if parsing fails.
    """
    rows: list[dict] = []

    try:
        tables = pd.read_html(StringIO(html))
        for table in tables:
            if table.shape[0] < 5:
                continue
            for col_idx in range(table.shape[1]):
                col = table.iloc[:, col_idx].astype(str)
                pct_matches = col.str.match(r"^\d+\.?\d*\s*%?$")
                if pct_matches.sum() >= 5:
                    # Found a weight column — find the name column
                    name_col_idx = None
                    for nc in range(table.shape[1]):
                        if nc == col_idx:
                            continue
                        nc_vals = table.iloc[:, nc].astype(str)
                        if nc_vals.str.contains(r"[A-Za-z]{3,}").sum() >= 5:
                            name_col_idx = nc
                            break
                    if name_col_idx is not None:
                        weights = pd.to_numeric(
                            col.str.replace("%", "").str.replace(",", "."),
                            errors="coerce",
                        ).fillna(0.0)
                        names = table.iloc[:, name_col_idx].astype(str)
                        for name, weight in zip(names, weights):
                            if name.strip() and weight > 0:
                                rows.append({
                                    "ticker": "N/A",
                                    "name": name.strip(),
                                    "weight": weight,
                                })
                        break
            if rows:
                break
    except Exception as exc:
        logger.debug("HTML table parsing failed for %s: %s", provider_label, exc)

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df = df[df["weight"] > 0]
    if df.empty:
        return None

    logger.info("Parsed %d holdings from %s page", len(df), provider_label or "HTML")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# ISIN discovery (via Financial Times)
# ---------------------------------------------------------------------------

def _discover_isin(yf_ticker: str) -> str | None:
    """Resolve ISIN from a Yahoo Finance ticker using the FT ETF summary page."""
    if "." in yf_ticker:
        ticker_part, exchange_suffix = yf_ticker.rsplit(".", 1)
    else:
        ticker_part, exchange_suffix = yf_ticker, "L"

    ft_exchange, ft_currency = _YF_EXCHANGE_TO_FT.get(exchange_suffix, ("LSE", "USD"))
    url = (
        f"https://markets.ft.com/data/etfs/tearsheet/summary"
        f"?s={ticker_part}:{ft_exchange}:{ft_currency}"
    )

    try:
        if curl_requests:
            resp = curl_requests.get(url, impersonate="chrome", timeout=15)
        else:
            resp = requests.get(url, headers=_HTTP_HEADERS, timeout=15)
        if not resp.ok:
            return None
    except Exception:
        return None

    # Structured match near the ISIN label
    match = re.search(r"ISIN[^<]*<[^>]*>([A-Z]{2}[A-Z0-9]{10})", resp.text)
    if match:
        candidate = match.group(1)
        return candidate if is_valid_isin(candidate) else None

    # Fallback: find fund-domicile ISINs
    fund_isins = list(set(re.findall(r"\b((?:IE|LU)[A-Z0-9]{10})\b", resp.text)))
    if len(fund_isins) == 1 and is_valid_isin(fund_isins[0]):
        return fund_isins[0]
    return None


# ---------------------------------------------------------------------------
# Provider detection (public — reused by app.py)
# ---------------------------------------------------------------------------

def detect_provider(fund_family: str) -> str:
    """Map fund family string to a provider key.

    Returns one of: 'ishares', 'vanguard', 'invesco', 'xtrackers', 'amundi', 'unknown'.
    """
    family = fund_family.lower()
    if "blackrock" in family or "ishares" in family:
        return "ishares"
    if "vanguard" in family:
        return "vanguard"
    if "invesco" in family:
        return "invesco"
    if "dws" in family or "xtrackers" in family:
        return "xtrackers"
    if "amundi" in family:
        return "amundi"
    return "unknown"


# Keep underscore alias for internal use and backwards compatibility with tests
_detect_provider = detect_provider


# ---------------------------------------------------------------------------
# iShares provider
# ---------------------------------------------------------------------------

def _discover_ishares_product_id(isin: str) -> str | None:
    """Look up the iShares numeric product ID from the justETF profile page."""
    url = f"https://www.justetf.com/en/etf-profile.html?isin={isin}"
    try:
        resp = requests.get(url, headers=_HTTP_HEADERS, timeout=15)
        if not resp.ok:
            return None
        match = re.search(r"ishares\.com/[^\"]*?/(?:products|produkte)/(\d+)/", resp.text)
        return match.group(1) if match else None
    except Exception:
        return None


def _fetch_ishares(product_id: str) -> FetchResult:
    """Download the full holdings CSV from iShares and return a normalised DataFrame."""
    url = (
        f"{_ISHARES_BASE_URL}/{product_id}/x/"
        f"{_ISHARES_AJAX_ID}.ajax?fileType=csv&fileName=holdings&dataType=fund"
    )
    try:
        resp = requests.get(url, headers=_HTTP_HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("iShares request failed: %s", exc)
        return FetchResult(holdings=None, used_justetf=False)

    lines = resp.text.splitlines()
    header_idx = next(
        (i for i, line in enumerate(lines) if "Ticker" in line and "Name" in line),
        None,
    )
    if header_idx is None:
        logger.warning("Could not locate CSV header in iShares response")
        return FetchResult(holdings=None, used_justetf=False)

    df = pd.read_csv(StringIO("\n".join(lines[header_idx:])), on_bad_lines="skip")
    if "Name" not in df.columns or "Weight (%)" not in df.columns:
        return FetchResult(holdings=None, used_justetf=False)

    df = df[df["Name"].notna() & (df["Name"].astype(str).str.strip() != "")]

    holdings = pd.DataFrame({
        "ticker": df["Ticker"].astype(str).values if "Ticker" in df.columns else "N/A",
        "name": df["Name"].values,
        "weight": pd.to_numeric(
            df["Weight (%)"].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        ).fillna(0.0).values,
    }).reset_index(drop=True)

    return FetchResult(holdings=holdings, used_justetf=False)


# ---------------------------------------------------------------------------
# Vanguard provider
# ---------------------------------------------------------------------------

def _fetch_vanguard(isin: str) -> FetchResult:
    """Query the Vanguard GraphQL API by ISIN and return a normalised DataFrame."""
    query = (
        '{ borHoldings(isins: ["%s"]) '
        "{ holdings(limit: 1500) { totalHoldings items "
        "{ ticker issuerName marketValuePercentage } } } }" % isin
    )
    try:
        resp = requests.post(
            _VANGUARD_GRAPHQL_URL,
            json={"query": query},
            headers=_VANGUARD_HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("Vanguard request failed: %s", exc)
        return FetchResult(holdings=None, used_justetf=False)

    if "errors" in data:
        logger.warning("Vanguard API error: %s", data["errors"][0].get("message", ""))
        return FetchResult(holdings=None, used_justetf=False)

    try:
        items = data["data"]["borHoldings"][0]["holdings"]["items"]
    except (KeyError, IndexError, TypeError):
        logger.warning("Unexpected Vanguard response structure")
        return FetchResult(holdings=None, used_justetf=False)

    if not items:
        return FetchResult(holdings=None, used_justetf=False)

    df = pd.DataFrame({
        "ticker": [it.get("ticker") or "N/A" for it in items],
        "name": [it.get("issuerName", "") for it in items],
        "weight": [it.get("marketValuePercentage", 0.0) for it in items],
    })
    holdings = df[df["name"].str.strip() != ""].reset_index(drop=True)
    if holdings.empty:
        return FetchResult(holdings=None, used_justetf=False)
    return FetchResult(holdings=holdings, used_justetf=False)


# ---------------------------------------------------------------------------
# Invesco provider
# ---------------------------------------------------------------------------

def _fetch_invesco(isin: str) -> FetchResult:
    """Fetch holdings from the Invesco DNG API for a given ISIN.

    Uses the public Invesco API endpoint that powers their product pages.
    Returns all holdings (not just top 10).
    """
    api_url = f"{_INVESCO_HOLDINGS_API}/{isin}/holdings/index?idType=isin"
    try:
        resp = requests.get(api_url, headers=_INVESCO_HEADERS, timeout=20)
        if not resp.ok:
            logger.warning("Invesco API returned %d", resp.status_code)
            return _fetch_justetf(isin)

        data = resp.json()
        items = data.get("holdings")
        if not items:
            logger.warning("Invesco API returned no holdings")
            return _fetch_justetf(isin)

        rows = []
        for item in items:
            name = item.get("name", "")
            weight = item.get("weight", 0.0)
            # Strip currency denomination (e.g. "NVIDIA CORP USD0.001" → "NVIDIA CORP")
            name = re.sub(r'\s+[A-Z]{3}\d[\d.]*$', '', name)
            if name and weight > 0:
                rows.append({"ticker": "N/A", "name": name, "weight": float(weight)})

        if not rows:
            logger.warning("No valid holdings parsed from Invesco API")
            return _fetch_justetf(isin)

        df = pd.DataFrame(rows)
        logger.info("Invesco API returned %d holdings", len(df))
        return FetchResult(holdings=df.reset_index(drop=True), used_justetf=False)

    except Exception as exc:
        logger.warning("Invesco API request failed: %s", exc)
        return _fetch_justetf(isin)


# ---------------------------------------------------------------------------
# Xtrackers (DWS) provider
# ---------------------------------------------------------------------------

def _fetch_xtrackers(isin: str) -> FetchResult:
    """Fetch holdings for an Xtrackers (DWS) ETF.

    Strategy:
    1. Try to find the correct product page slug from justETF, then scrape it.
    2. Try the Vanguard GraphQL API (sometimes has cross-provider data).
    3. Fall back to justETF top-10.
    """
    # --- Attempt 1: Find the correct DWS product page via justETF link ---
    try:
        justetf_url = f"https://www.justetf.com/en/etf-profile.html?isin={isin}"
        resp = requests.get(justetf_url, headers=_HTTP_HEADERS, timeout=15)
        if resp.ok:
            match = re.search(
                r'href="(https?://etf\.dws\.com/[^"]+?/' + re.escape(isin) + r'[^"]*)"',
                resp.text,
            )
            if match:
                dws_url = match.group(1)
                if curl_requests:
                    page_resp = curl_requests.get(dws_url, impersonate="chrome", timeout=20)
                else:
                    page_resp = requests.get(dws_url, headers=_XTRACKERS_HEADERS, timeout=20)
                if page_resp.ok:
                    df = _parse_holdings_from_html(page_resp.text, "Xtrackers")
                    if df is not None:
                        return FetchResult(holdings=df, used_justetf=False)
    except Exception:
        pass

    # --- Attempt 2: Vanguard GraphQL (sometimes has cross-provider data) ---
    result = _fetch_vanguard(isin)
    if result.holdings is not None:
        return result

    # --- Attempt 3: justETF fallback ---
    logger.info("Xtrackers direct fetch failed, falling back to justETF")
    justetf_result = _fetch_justetf(isin)
    if justetf_result.holdings is None:
        logger.warning(
            "No holdings data available. This may be a swap-based or money market ETF "
            "that does not hold individual securities."
        )
    return justetf_result


# ---------------------------------------------------------------------------
# justETF fallback (Amundi and other providers without a public API)
# ---------------------------------------------------------------------------

def _fetch_justetf(isin: str) -> FetchResult:
    """Scrape the top-10 holdings table from the justETF profile page."""
    url = f"https://www.justetf.com/en/etf-profile.html?isin={isin}"
    try:
        resp = requests.get(url, headers=_HTTP_HEADERS, timeout=15)
        if not resp.ok:
            return FetchResult(holdings=None, used_justetf=False)
        tables = pd.read_html(StringIO(resp.text))
    except Exception:
        return FetchResult(holdings=None, used_justetf=False)

    for table in tables:
        if table.shape[0] < 5 or table.shape[1] != 2:
            continue
        pct_col = table.iloc[:, 1].astype(str)
        if pct_col.str.contains("%").sum() < 5:
            continue
        # Skip performance / risk tables
        name_col = table.iloc[:, 0].astype(str)
        if name_col.str.contains(r"(?:YTD|month|year|Volatility|drawdown)", case=False).any():
            continue

        weights = pd.to_numeric(
            pct_col.str.replace("%", "").str.replace(",", "."), errors="coerce"
        ).fillna(0.0)
        df = pd.DataFrame({
            "ticker": "N/A",
            "name": name_col.values,
            "weight": weights.values,
        })
        df = df[df["weight"] > 0]
        if not df.empty:
            return FetchResult(holdings=df.reset_index(drop=True), used_justetf=True)

    return FetchResult(holdings=None, used_justetf=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class HoldingsFetcher:
    """Stateful holdings fetcher that tracks justETF usage per batch.

    Usage::

        fetcher = HoldingsFetcher()
        df = fetcher.fetch(ticker, fund_family)
        # After all fetches:
        justetf_tickers = fetcher.justetf_tickers
    """

    def __init__(self) -> None:
        self._justetf_used: set[str] = set()
        self._cache: dict | None = None

    @property
    def justetf_tickers(self) -> set[str]:
        """Tickers that were resolved via justETF (top 10 only) in this batch."""
        return self._justetf_used.copy()

    def clear(self) -> None:
        """Reset tracking state for a new batch."""
        self._justetf_used.clear()
        self._cache = None

    def _get_cache(self) -> dict:
        """Load cache once per batch, reusing across multiple fetch() calls."""
        if self._cache is None:
            self._cache = _load_cache()
        return self._cache

    def _persist_cache(self) -> None:
        """Write the in-memory cache to disk."""
        if self._cache is not None:
            _save_cache(self._cache)

    def _resolve_isin(self, yf_ticker: str) -> str | None:
        """Resolve ISIN from cache or discovery, updating cache on success."""
        cache = self._get_cache()
        isin = cache.get(yf_ticker, {}).get("isin")
        if isin:
            return isin

        logger.info("Discovering ISIN for %s...", yf_ticker)
        isin = _discover_isin(yf_ticker)
        if isin:
            cache.setdefault(yf_ticker, {})["isin"] = isin
            self._persist_cache()
            logger.info("Found ISIN: %s", isin)
            return isin

        logger.warning("Could not discover ISIN for %s", yf_ticker)
        return None

    def _handle_fetch_result(self, yf_ticker: str, result: FetchResult, provider: str, isin: str) -> pd.DataFrame | None:
        """Process a FetchResult: update cache and tracking, return holdings."""
        if result.holdings is not None:
            cache = self._get_cache()
            cache.setdefault(yf_ticker, {}).update({"provider": provider, "isin": isin})
            self._persist_cache()
            if result.used_justetf:
                self._justetf_used.add(yf_ticker)
        return result.holdings

    def fetch(self, yf_ticker: str, fund_family: str = "") -> pd.DataFrame | None:
        """
        Fetch live holdings for an ETF given its Yahoo Finance ticker.

        Auto-discovers the ISIN, detects the provider, and fetches holdings.
        All discovery results are cached in ``.etf_cache.json`` for speed.

        Parameters
        ----------
        yf_ticker : str
            Yahoo Finance ticker, e.g. ``'CSPX.L'``, ``'VHVE.L'``, ``'PRAM.DE'``.
        fund_family : str, optional
            Fund family string (from ``yfinance``). Helps route to the correct provider.

        Returns
        -------
        pd.DataFrame or None
            Columns: ``ticker``, ``name``, ``weight`` (percentage).
            Returns ``None`` if the fetch fails entirely.
        """
        cache = self._get_cache()

        # --- Resolve ISIN ---
        isin = self._resolve_isin(yf_ticker)
        if not isin:
            return None

        # --- Detect provider ---
        provider = cache.get(yf_ticker, {}).get("provider") or detect_provider(fund_family)

        # --- Fetch by provider ---
        if provider == "vanguard":
            result = _fetch_vanguard(isin)
            return self._handle_fetch_result(yf_ticker, result, "vanguard", isin)

        if provider == "invesco":
            result = _fetch_invesco(isin)
            return self._handle_fetch_result(yf_ticker, result, "invesco", isin)

        if provider == "xtrackers":
            result = _fetch_xtrackers(isin)
            return self._handle_fetch_result(yf_ticker, result, "xtrackers", isin)

        if provider == "amundi":
            logger.info("Using justETF fallback (top holdings only)")
            result = _fetch_justetf(isin)
            if result.holdings is not None:
                self._justetf_used.add(yf_ticker)
            return result.holdings

        # iShares or unknown — try product ID discovery
        product_id = cache.get(yf_ticker, {}).get("product_id")
        if not product_id:
            logger.info("Discovering iShares product ID for %s...", isin)
            product_id = _discover_ishares_product_id(isin)
            if product_id:
                entry = cache.setdefault(yf_ticker, {})
                entry.update({"product_id": product_id, "provider": "ishares", "isin": isin})
                self._persist_cache()
                logger.info("Found product ID: %s", product_id)
            else:
                # Fallback chain: Vanguard → justETF
                vanguard_result = _fetch_vanguard(isin)
                if vanguard_result.holdings is not None:
                    return self._handle_fetch_result(yf_ticker, vanguard_result, "vanguard", isin)
                logger.info("Trying justETF fallback...")
                result = _fetch_justetf(isin)
                if result.holdings is not None:
                    self._justetf_used.add(yf_ticker)
                return result.holdings

        result = _fetch_ishares(product_id)
        return result.holdings


# ---------------------------------------------------------------------------
# Module-level convenience API (backwards-compatible)
# ---------------------------------------------------------------------------

_default_fetcher = HoldingsFetcher()


def get_justetf_tickers() -> set[str]:
    """Return the set of tickers that were resolved via justETF (top 10 only)."""
    return _default_fetcher.justetf_tickers


def clear_justetf_tickers() -> None:
    """Clear the justETF tracking set. Call before a new analysis batch."""
    _default_fetcher.clear()


def fetch_holdings(yf_ticker: str, fund_family: str = "") -> pd.DataFrame | None:
    """Convenience wrapper using the module-level default fetcher."""
    return _default_fetcher.fetch(yf_ticker, fund_family)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("=== ETF Holdings Scraper ===\n")
    test_tickers = {
        "CSPX.L": "BlackRock",
        "VHVE.L": "Vanguard",
        "EQQQ.L": "Invesco",
        "XDWD.DE": "DWS Investment S.A. (ETF)",
        "PRAM.DE": "Amundi",
    }
    fetcher = HoldingsFetcher()
    for ticker, family in test_tickers.items():
        print(f"[{ticker}] ({family})")
        holdings = fetcher.fetch(ticker, fund_family=family)
        if holdings is not None:
            print(f"  {len(holdings)} holdings loaded")
            for _, row in holdings.nlargest(3, "weight").iterrows():
                print(f"    {row['ticker']:6s}  {row['name']:35s}  {row['weight']:.2f}%")
        else:
            print("  FAILED")
        print()
