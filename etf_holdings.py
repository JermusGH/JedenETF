"""
ETF Holdings Scraper — Auto-discovery and live fetching.

Supports iShares (BlackRock), Vanguard, Invesco, Xtrackers (DWS),
and Amundi (via justETF fallback).
Automatically discovers ISIN, provider, and provider-specific parameters
from a Yahoo Finance ticker. All discovered data is cached locally.
"""

import json
import os
import re
from io import StringIO

import pandas as pd
import requests

from validation import is_valid_isin

try:
    from curl_cffi import requests as curl_requests
except ImportError:
    curl_requests = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".etf_cache.json")

# Tracks tickers that were resolved via justETF (top 10 only) during the last
# batch of fetch_holdings calls. Cleared at the start of each fetch_holdings call.
_justetf_used: set[str] = set()
_last_fetch_used_justetf: bool = False

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

_INVESCO_BASE_URL = "https://www.invesco.com/uk/en/financial-products/etfs"
_INVESCO_HOLDINGS_API = (
    "https://dng-api.invesco.com/cache/v1/accounts/en_GB/shareclasses"
)
_INVESCO_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.invesco.com/uk/en/financial-products/etfs.html",
}

_XTRACKERS_BASE_URL = "https://etf.dws.com/en-gb"
_XTRACKERS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html, */*",
    "Referer": "https://etf.dws.com/en-gb/",
}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_cache() -> dict:
    """Load the discovery cache from disk.

    Returns an empty dict if the file does not exist, contains malformed JSON,
    is empty, or cannot be read for any reason (Requirement 12.6).
    """
    if not os.path.exists(_CACHE_PATH):
        return {}
    try:
        with open(_CACHE_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
        return {}
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    """Persist the discovery cache to disk.

    Silently continues on any write failure so that analysis is never
    interrupted by a caching error (Requirement 12.7).
    """
    try:
        with open(_CACHE_PATH, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, indent=2)
    except Exception:
        pass


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
# Provider detection
# ---------------------------------------------------------------------------

def _detect_provider(fund_family: str) -> str:
    """Map fund family string to a provider key."""
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
# Invesco provider
# ---------------------------------------------------------------------------

def _discover_invesco_slug(isin: str) -> str | None:
    """Discover the Invesco product page slug from justETF's link to invesco.com."""
    # Kept for potential future use but not currently needed — the DNG API works directly.
    url = f"https://www.justetf.com/en/etf-profile.html?isin={isin}"
    try:
        resp = requests.get(url, headers=_HTTP_HEADERS, timeout=15)
        if not resp.ok:
            return None
        match = re.search(
            r"invesco\.com/[^\"]*?/financial-products/etfs/([a-z0-9\-]+?)(?:\.html)?[\"?]",
            resp.text,
        )
        return match.group(1) if match else None
    except Exception:
        return None


def _fetch_invesco(isin: str) -> pd.DataFrame | None:
    """Fetch holdings from the Invesco DNG API for a given ISIN.

    Uses the public Invesco API endpoint that powers their product pages.
    Returns all holdings (not just top 10).
    """
    api_url = (
        f"{_INVESCO_HOLDINGS_API}/{isin}/holdings/index?idType=isin"
    )
    try:
        resp = requests.get(api_url, headers=_INVESCO_HEADERS, timeout=20)
        if not resp.ok:
            print(f"  [!] Invesco API returned {resp.status_code}")
            return _fetch_justetf(isin)

        data = resp.json()
        items = data.get("holdings")
        if not items:
            print("  [!] Invesco API returned no holdings")
            return _fetch_justetf(isin)

        rows = []
        for item in items:
            name = item.get("name", "")
            weight = item.get("weight", 0.0)
            # Strip currency denomination from name (e.g. "NVIDIA CORP USD0.001" → "NVIDIA CORP")
            name = re.sub(r'\s+[A-Z]{3}\d[\d.]*$', '', name)
            if name and weight > 0:
                rows.append({"ticker": "N/A", "name": name, "weight": float(weight)})

        if not rows:
            print("  [!] No valid holdings parsed from Invesco API")
            return _fetch_justetf(isin)

        df = pd.DataFrame(rows)
        print(f"  Invesco API returned {len(df)} holdings")
        return df.reset_index(drop=True)

    except Exception as exc:
        print(f"  [!] Invesco API request failed: {exc}")
        return _fetch_justetf(isin)


def _parse_invesco_holdings_page(html: str) -> pd.DataFrame | None:
    """Parse holdings data from an Invesco product page HTML.

    The page contains a holdings table with columns for name, CUSIP, ISIN, and weight.
    """
    # Try to find holdings data in the HTML
    # Pattern 1: Table rows with weight percentages (e.g. "8.78%")
    # The Invesco page shows holdings like: NAME | CUSIP | ISIN | WEIGHT%
    rows = []

    # Look for structured holdings data — the page renders holdings in a table
    # with security name and weight percentage
    holdings_pattern = re.findall(
        r'([A-Z][A-Z0-9 &.,/\-\'()]+?)\s+[A-Z]{3}\d[\d.]*\s*\|\s*[A-Z0-9]+\s*\|\s*[A-Z]{2}[A-Z0-9]{10}\s*\|\s*(\d+\.?\d*)\s*%',
        html,
    )
    if not holdings_pattern:
        # Try without the currency denomination part
        holdings_pattern = re.findall(
            r'([A-Z][A-Z0-9 &.,/\-\'()]+?)\s*\|\s*[A-Z0-9]+\s*\|\s*[A-Z]{2}[A-Z0-9]{10}\s*\|\s*(\d+\.?\d*)\s*%',
            html,
        )
    if holdings_pattern:
        for name, weight in holdings_pattern:
            name = name.strip()
            # Remove trailing currency denomination (e.g. "USD0.001")
            name = re.sub(r'\s+[A-Z]{3}\d[\d.]*$', '', name)
            if name and float(weight) > 0:
                rows.append({"ticker": "N/A", "name": name, "weight": float(weight)})

    # Pattern 2: Try parsing HTML tables with pd.read_html
    if not rows:
        try:
            tables = pd.read_html(StringIO(html))
            for table in tables:
                # Look for a table that has a percentage column and enough rows
                if table.shape[0] < 5:
                    continue
                # Check if any column contains percentage-like values
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
                            # Name column should have mostly alphabetic content
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
        except Exception:
            pass

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df = df[df["weight"] > 0]
    if df.empty:
        return None

    print(f"  Parsed {len(df)} holdings from Invesco page")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Xtrackers (DWS) provider
# ---------------------------------------------------------------------------

def _fetch_xtrackers(isin: str) -> pd.DataFrame | None:
    """Fetch holdings for an Xtrackers (DWS) ETF.

    Strategy:
    1. Try scraping the DWS product page via curl_cffi (bypasses consent wall).
    2. Try the Vanguard GraphQL API (sometimes has cross-provider data).
    3. Fall back to justETF.
    """
    # --- Attempt 1: DWS product page ---
    product_url = f"{_XTRACKERS_BASE_URL}/{isin}-msci-world-ucits-etf-1c/"
    try:
        if curl_requests:
            resp = curl_requests.get(product_url, impersonate="chrome", timeout=20)
            if resp.ok and "holdings" in resp.text.lower():
                df = _parse_xtrackers_holdings_page(resp.text)
                if df is not None:
                    return df
    except Exception:
        pass

    # --- Attempt 2: Try to find the correct product page slug from justETF ---
    try:
        justetf_url = f"https://www.justetf.com/en/etf-profile.html?isin={isin}"
        resp = requests.get(justetf_url, headers=_HTTP_HEADERS, timeout=15)
        if resp.ok:
            # Look for etf.dws.com product page links
            match = re.search(
                r'href="(https?://etf\.dws\.com/[^"]+?/' + re.escape(isin) + r'[^"]*)"',
                resp.text,
            )
            if match:
                dws_url = match.group(1)
                if curl_requests:
                    page_resp = curl_requests.get(
                        dws_url, impersonate="chrome", timeout=20
                    )
                else:
                    page_resp = requests.get(
                        dws_url, headers=_XTRACKERS_HEADERS, timeout=20
                    )
                if page_resp.ok:
                    df = _parse_xtrackers_holdings_page(page_resp.text)
                    if df is not None:
                        return df
    except Exception:
        pass

    # --- Attempt 3: Vanguard GraphQL (sometimes has cross-provider data) ---
    result = _fetch_vanguard(isin)
    if result is not None:
        return result

    # --- Attempt 4: justETF fallback ---
    print("  [!] Xtrackers direct fetch failed, falling back to justETF")
    result = _fetch_justetf(isin)
    if result is None:
        print(
            "  [!] No holdings data available. This may be a swap-based or money market ETF "
            "that does not hold individual securities."
        )
    return result


def _parse_xtrackers_holdings_page(html: str) -> pd.DataFrame | None:
    """Parse holdings data from a DWS/Xtrackers product page HTML."""
    rows = []

    # DWS pages may contain holdings in HTML tables
    try:
        tables = pd.read_html(StringIO(html))
        for table in tables:
            if table.shape[0] < 5:
                continue
            # Look for a column with percentage values and a name column
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
    except Exception:
        pass

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df = df[df["weight"] > 0]
    if df.empty:
        return None

    print(f"  Parsed {len(df)} holdings from Xtrackers page")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# justETF fallback (Amundi and other providers without a public API)
# ---------------------------------------------------------------------------

def _fetch_justetf(isin: str) -> pd.DataFrame | None:
    """Scrape the top-10 holdings table from the justETF profile page."""
    global _last_fetch_used_justetf
    _last_fetch_used_justetf = False

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
            _last_fetch_used_justetf = True
            return df.reset_index(drop=True)

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_justetf_tickers() -> set[str]:
    """Return the set of tickers that were resolved via justETF (top 10 only)."""
    return _justetf_used.copy()


def clear_justetf_tickers() -> None:
    """Clear the justETF tracking set. Call before a new analysis batch."""
    _justetf_used.clear()


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

    if provider == "invesco":
        result = _fetch_invesco(isin)
        if result is not None:
            cache.setdefault(yf_ticker, {}).update({"provider": "invesco", "isin": isin})
            _save_cache(cache)
            if _last_fetch_used_justetf:
                _justetf_used.add(yf_ticker)
        return result

    if provider == "xtrackers":
        result = _fetch_xtrackers(isin)
        if result is not None:
            cache.setdefault(yf_ticker, {}).update({"provider": "xtrackers", "isin": isin})
            _save_cache(cache)
            if _last_fetch_used_justetf:
                _justetf_used.add(yf_ticker)
        return result

    if provider == "amundi":
        print("  Using justETF fallback (top holdings only)")
        result = _fetch_justetf(isin)
        if result is not None:
            _justetf_used.add(yf_ticker)
        return result

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
            result = _fetch_justetf(isin)
            if result is not None:
                _justetf_used.add(yf_ticker)
            return result

    return _fetch_ishares(product_id)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== ETF Holdings Scraper ===\n")
    test_tickers = {
        "CSPX.L": "BlackRock",
        "VHVE.L": "Vanguard",
        "EQQQ.L": "Invesco",
        "XDWD.DE": "DWS Investment S.A. (ETF)",
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
