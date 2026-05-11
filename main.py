"""
Unified ETF Portfolio Analyser

Merges holdings from multiple ETFs into a single unified view,
showing your true exposure to individual companies across all funds.
"""

import re
import sys

import pandas as pd

try:
    import yfinance  # noqa: F401 — validated at import time
except ImportError:
    print("[!] Missing required library: yfinance")
    print(f"    Python interpreter: {sys.executable}")
    print(f"    Install with: {sys.executable} -m pip install yfinance")
    sys.exit(1)

from etf_holdings import fetch_holdings
from portfolio import PORTFOLIO, TARGET_CURRENCY, fetch_prices, get_fund_family


# ---------------------------------------------------------------------------
# Company name normalisation (for merging across ETFs)
# ---------------------------------------------------------------------------

_STRIP_SUFFIXES = re.compile(
    r"\b(?:INC|CORP|CORPORATION|LTD|LIMITED|PLC|COMPANY|AG|SA|NV|"
    r"GROUP|HOLDINGS|HOLDING|SE|CLASS\s+[A-C]|CL\s+[A-C]|CO(?!\w))\b"
)


def _normalise_name(name) -> str:
    """Reduce a company name to a short canonical form for deduplication."""
    if pd.isna(name):
        return ""
    text = re.sub(r"[^\w\s]", "", str(name).upper())
    text = _STRIP_SUFFIXES.sub("", text)
    words = text.split()
    return " ".join(words[:2])


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def _build_unified_portfolio(
    portfolio: dict[str, float],
    prices: dict[str, float | None],
) -> tuple[pd.DataFrame, float]:
    """
    Fetch holdings for each ETF and merge into a single DataFrame.

    Returns (combined_df, total_portfolio_value).
    """
    frames: list[pd.DataFrame] = []
    total_value = 0.0

    for ticker, units in portfolio.items():
        if units <= 0:
            continue

        price = prices.get(ticker)
        if price is None:
            print(f"\n  [!] No price for {ticker} — skipping")
            continue

        etf_value = units * price
        fund_family = get_fund_family(ticker)

        print(
            f"\n  {ticker} | Price: {price:.2f} {TARGET_CURRENCY} | "
            f"Units: {units} | Value: {etf_value:.2f} {TARGET_CURRENCY}"
        )

        df = fetch_holdings(yf_ticker=ticker, fund_family=fund_family)
        if df is None or df.empty:
            print("  [!] Failed to fetch holdings")
            continue

        # Convert weight (%) to portfolio value
        weight_sum = df["weight"].sum()
        if weight_sum > 10:
            df["value"] = (df["weight"] / 100.0) * etf_value
        else:
            df["value"] = df["weight"] * etf_value

        df["source"] = ticker
        df["merge_key"] = df["name"].apply(_normalise_name)

        total_value += etf_value
        frames.append(df)

    if not frames:
        return pd.DataFrame(), 0.0

    combined = pd.concat(frames, ignore_index=True)
    return combined, total_value


def _aggregate(combined: pd.DataFrame, total_value: float) -> pd.DataFrame:
    """Group by normalised name and compute unified weights."""
    # Prefer shorter ticker strings (real tickers over ISINs)
    combined["_ticker_len"] = combined["ticker"].astype(str).str.len()
    combined = combined.sort_values("_ticker_len")

    grouped = (
        combined.groupby("merge_key")
        .agg(ticker=("ticker", "first"), name=("name", "first"), value=("value", "sum"))
        .reset_index()
    )
    grouped["weight_%"] = (grouped["value"] / total_value) * 100.0

    # Per-source contribution pivot
    combined["contrib_%"] = (combined["value"] / total_value) * 100.0
    pivot = combined.pivot_table(
        index="merge_key", columns="source", values="contrib_%", aggfunc="sum", fill_value=0.0
    ).reset_index()

    return grouped.merge(pivot, on="merge_key")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_report(final: pd.DataFrame, total_value: float, sources: list[str]) -> list[str]:
    """Print a formatted top-20 table and return the lines for file output."""
    top20 = final.nlargest(20, "weight_%")

    COL_TICKER = min(max(12, top20["ticker"].astype(str).str.len().max()), 20)
    COL_NAME = min(max(14, top20["name"].astype(str).str.len().max()), 40)
    COL_SRC = min(max(9, max((len(s) for s in sources), default=0)), 14)

    def fmt_ticker(v):
        s = str(v).replace("nan", "N/A").strip()[:COL_TICKER]
        return s.ljust(COL_TICKER)

    def fmt_name(v):
        s = "" if pd.isna(v) else str(v).strip()[:COL_NAME]
        return s.ljust(COL_NAME)

    def fmt_weight(v):
        return f"{v:>6.2f}%".rjust(9)

    def fmt_value(v):
        return f"{v:>12.2f}"

    def fmt_src(v):
        return f"{v:>6.2f}%".rjust(COL_SRC) if v > 0.005 else "-".rjust(COL_SRC)

    header = (
        f"{'Ticker':<{COL_TICKER}} | {'Company':<{COL_NAME}} | {'Weight':>9} | "
        f"{'Value (' + TARGET_CURRENCY + ')':>12} | "
        + " | ".join(s[:COL_SRC].ljust(COL_SRC) for s in sources)
    )
    sep = "-" * len(header)

    lines: list[str] = []

    def emit(line=""):
        print(line)
        lines.append(line)

    emit()
    emit("=" * len(header))
    emit("TOP 20 HOLDINGS IN YOUR UNIFIED PORTFOLIO")
    emit("=" * len(header))
    emit(header)
    emit(sep)

    for _, row in top20.iterrows():
        src_cells = " | ".join(fmt_src(row.get(s, 0)) for s in sources)
        emit(
            f"{fmt_ticker(row['ticker'])} | {fmt_name(row['name'])} | "
            f"{fmt_weight(row['weight_%'])} | {fmt_value(row['value'])} | {src_cells}"
        )

    emit(sep)
    emit(f"Total portfolio value: {total_value:,.2f} {TARGET_CURRENCY}")
    emit(f"Unique holdings (merged): {len(final)}")
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not PORTFOLIO:
        print("[!] PORTFOLIO is empty. Add ETFs to portfolio.py")
        return

    tickers = list(PORTFOLIO.keys())
    prices = fetch_prices(tickers)

    print("\n--- UNIFIED ETF PORTFOLIO ANALYSIS ---")

    combined, total_value = _build_unified_portfolio(PORTFOLIO, prices)
    if combined.empty:
        print("\n[!] No holdings data available for analysis.")
        return

    sources = sorted(combined["source"].unique())
    final = _aggregate(combined, total_value)

    lines = _print_report(final, total_value, sources)


if __name__ == "__main__":
    main()
