"""
Unified ETF Portfolio Analyser

Merges holdings from multiple ETFs into a single unified view,
showing your true exposure to individual companies across all funds.
"""

import logging
import sys

import pandas as pd

try:
    import yfinance  # noqa: F401 — validated at import time
except ImportError:
    print("[!] Missing required library: yfinance")
    print(f"    Python interpreter: {sys.executable}")
    print(f"    Install with: {sys.executable} -m pip install yfinance")
    sys.exit(1)

from analysis import enrich_holdings, merge_holdings
from etf_holdings import HoldingsFetcher
from models import AnalysisResult
from portfolio import PORTFOLIO, TARGET_CURRENCY
from pricing import fetch_prices, get_fund_family

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def _build_unified_portfolio(
    portfolio: dict[str, float],
    prices: dict[str, float | None],
) -> AnalysisResult:
    """
    Fetch holdings for each ETF and enrich with values.

    Returns an AnalysisResult with the merged DataFrame and metadata.
    """
    frames: list[pd.DataFrame] = []
    total_value = 0.0
    failed_tickers: list[str] = []
    fetcher = HoldingsFetcher()

    for ticker, units in portfolio.items():
        if units <= 0:
            continue

        price = prices.get(ticker)
        if price is None:
            logger.warning("No price for %s — skipping", ticker)
            failed_tickers.append(ticker)
            continue

        etf_value = units * price
        fund_family = get_fund_family(ticker)

        logger.info(
            "%s | Price: %.2f %s | Units: %s | Value: %.2f %s",
            ticker, price, TARGET_CURRENCY, units, etf_value, TARGET_CURRENCY,
        )

        df = fetcher.fetch(yf_ticker=ticker, fund_family=fund_family)
        if df is None or df.empty:
            logger.warning("Failed to fetch holdings for %s", ticker)
            failed_tickers.append(ticker)
            continue

        enriched = enrich_holdings(df, etf_value, ticker)
        total_value += etf_value
        frames.append(enriched)

    if not frames:
        return AnalysisResult(
            df=pd.DataFrame(),
            total_value=0.0,
            failed_tickers=failed_tickers,
            justetf_tickers=sorted(fetcher.justetf_tickers),
        )

    final = merge_holdings(frames, total_value)
    sources = sorted(set(s for f in frames for s in f["source"].unique()))

    return AnalysisResult(
        df=final,
        total_value=total_value,
        sources=sources,
        failed_tickers=failed_tickers,
        justetf_tickers=sorted(fetcher.justetf_tickers),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_report(result: AnalysisResult) -> list[str]:
    """Print a formatted top-20 table and return the lines for file output."""
    top20 = result.df.nlargest(20, "weight_%")
    sources = result.sources

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
    emit(f"Total portfolio value: {result.total_value:,.2f} {TARGET_CURRENCY}")
    emit(f"Unique holdings (merged): {len(result.df)}")
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not PORTFOLIO:
        print("[!] PORTFOLIO is empty. Add ETFs to portfolio.py")
        return

    tickers = list(PORTFOLIO.keys())
    prices = fetch_prices(tickers)

    print("\n--- UNIFIED ETF PORTFOLIO ANALYSIS ---")

    result = _build_unified_portfolio(PORTFOLIO, prices)
    if result.is_empty:
        print("\n[!] No holdings data available for analysis.")
        return

    _print_report(result)


if __name__ == "__main__":
    main()
