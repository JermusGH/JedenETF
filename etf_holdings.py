"""
ETF Holdings Scraper — Auto-discovery and live fetching.

Supports iShares (BlackRock), Vanguard, and Amundi (via justETF fallback).
Automatically discovers ISIN, provider, and provider-specific parameters
from a Yahoo Finance ticker. All discovered data is cached locally.
"""

import json
import os
import re
from io import StringIO

import pandas as pd
import requests

try:
    from curl_cffi import requests as curl_requests
except ImportError:
    curl_requests = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".etf_cache.json")

_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

_YF_EXCHANGE_TO_FT = {
    "L": ("LSE", "USD"),
    "DE": ("GER", "EUR"),
    "AS": ("AMS", "EUR"),
    "PA": ("PAR", "EUR"),
    "MI": ("MIL", "EUR"),
    "SW": ("SWX", "CHF"),
}

_ISHARES_AJAX_ID = "1506575576011"
_ISHARES_BASE_URL = "https://www.ishares.com/uk/individual/en/products"

_VANGUARD_GRAPHQL_URL = "https://www.nl.vanguard/gpx/graphql"
_VANGUARD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Origin": "https://www.nl.vanguard",
    "Referer": "https://www.nl.vanguard/professional/product/etf/equity/",
    "x-consumer-id": "nl-ui",
}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_cache() -> dict:
    if os.path.exists(_CACHE_PATH):
        try:
            with open(_CACHE_PATH, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_cache(cache: dict) -> None:
    with open(_CACHE_PATH, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=2)


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
        return match.group(1)

    # Fallback: find fund-domicile ISINs
    fund_isins = list(set(re.findall(r"\b((?:IE|LU)[A-Z0-9]{10})\b", resp.text)))
    return fund_isins[0] if len(fund_isins) == 1 else None


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

def _detect_provider(fund_family: str) -> str:
    """Map fund family string to a provider key."""
    family = fund_family.lower()
    if "blackrock" in family or "ishares" in family:
        return "ishares"
    if "vanguard" in family:
        return "vanguard"
    if "amundi" in family:
        return "amundi"
    return "unknown"


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


def _fetch_ishares(product_id: str) -> pd.DataFrame | None:
    """Download the full holdings CSV from iShares and return a normalised DataFrame."""
    url = (
        f"{_ISHARES_BASE_URL}/{product_id}/x/"
        f"{_ISHARES_AJAX_ID}.ajax?fileType=csv&fileName=holdings&dataType=fund"
    )
    try:
        resp = requests.get(url, headers=_HTTP_HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception as exc:
        print(f"  [!] iShares request failed: {exc}")
        return None

    lines = resp.text.splitlines()
    header_idx = next(
        (i for i, line in enumerate(lines) if "Ticker" in line and "Name" in line),
        None,
    )
    if header_idx is None:
        print("  [!] Could not locate CSV header in iShares response")
        return None

    df = pd.read_csv(StringIO("\n".join(lines[header_idx:])), on_bad_lines="skip")
    if "Name" not in df.columns or "Weight (%)" not in df.columns:
        return None

    df = df[df["Name"].notna() & (df["Name"].astype(str).str.strip() != "")]

    return pd.DataFrame({
        "ticker": df["Ticker"].astype(str).values if "Ticker" in df.columns else "N/A",
        "name": df["Name"].values,
        "weight": pd.to_numeric(
            df["Weight (%)"].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        ).fillna(0.0).values,
    }).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Vanguard provider
# ---------------------------------------------------------------------------

def _fetch_vanguard(isin: str) -> pd.DataFrame | None:
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
        print(f"  [!] Vanguard request failed: {exc}")
        return None

    if "errors" in data:
        print(f"  [!] Vanguard API error: {data['errors'][0].get('message', '')}")
        return None

    try:
        items = data["data"]["borHoldings"][0]["holdings"]["items"]
    except (KeyError, IndexError, TypeError):
        print("  [!] Unexpected Vanguard response structure")
        return None

    if not items:
        return None

    df = pd.DataFrame({
        "ticker": [it.get("ticker") or "N/A" for it in items],
        "name": [it.get("issuerName", "") for it in items],
        "weight": [it.get("marketValuePercentage", 0.0) for it in items],
    })
    return df[df["name"].str.strip() != ""].reset_index(drop=True)


# ---------------------------------------------------------------------------
# justETF fallback (Amundi and other providers without a public API)
# ---------------------------------------------------------------------------

def _fetch_justetf(isin: str) -> pd.DataFrame | None:
    """Scrape the top-10 holdings table from the justETF profile page."""
    url = f"https://www.justetf.com/en/etf-profile.html?isin={isin}"
    try:
        resp = requests.get(url, headers=_HTTP_HEADERS, timeout=15)
        if not resp.ok:
            return None
        tables = pd.read_html(StringIO(resp.text))
    except Exception:
        return None

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
            return df.reset_index(drop=True)

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_holdings(yf_ticker: str, fund_family: str = "") -> pd.DataFrame | None:
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
        Columns: ``ticker``, ``name``, ``weight`` (percentage, e.g. 8.24 means 8.24%).
        Returns ``None`` if the fetch fails entirely.
    """
    cache = _load_cache()

    # --- Resolve ISIN ---
    isin = cache.get(yf_ticker, {}).get("isin")
    if not isin:
        print(f"  Discovering ISIN for {yf_ticker}...")
        isin = _discover_isin(yf_ticker)
        if isin:
            cache.setdefault(yf_ticker, {})["isin"] = isin
            _save_cache(cache)
            print(f"  Found ISIN: {isin}")
        else:
            print(f"  [!] Could not discover ISIN for {yf_ticker}")
            return None

    # --- Detect provider ---
    provider = cache.get(yf_ticker, {}).get("provider") or _detect_provider(fund_family)

    # --- Fetch by provider ---
    if provider == "vanguard":
        return _fetch_vanguard(isin)

    if provider == "amundi":
        print("  Using justETF fallback (top holdings only)")
        return _fetch_justetf(isin)

    # iShares or unknown
    product_id = cache.get(yf_ticker, {}).get("product_id")
    if not product_id:
        print(f"  Discovering iShares product ID for {isin}...")
        product_id = _discover_ishares_product_id(isin)
        if product_id:
            entry = cache.setdefault(yf_ticker, {})
            entry.update({"product_id": product_id, "provider": "ishares", "isin": isin})
            _save_cache(cache)
            print(f"  Found product ID: {product_id}")
        else:
            # Fallback chain: Vanguard → justETF
            result = _fetch_vanguard(isin)
            if result is not None:
                cache.setdefault(yf_ticker, {}).update({"provider": "vanguard", "isin": isin})
                _save_cache(cache)
                return result
            print("  Trying justETF fallback...")
            return _fetch_justetf(isin)

    return _fetch_ishares(product_id)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== ETF Holdings Scraper ===\n")
    test_tickers = {
        "CSPX.L": "BlackRock",
        "VHVE.L": "Vanguard",
        "PRAM.DE": "Amundi",
    }
    for ticker, family in test_tickers.items():
        print(f"[{ticker}] ({family})")
        holdings = fetch_holdings(ticker, fund_family=family)
        if holdings is not None:
            print(f"  {len(holdings)} holdings loaded")
            for _, row in holdings.nlargest(3, "weight").iterrows():
                print(f"    {row['ticker']:6s}  {row['name']:35s}  {row['weight']:.2f}%")
        else:
            print("  FAILED")
        print()
