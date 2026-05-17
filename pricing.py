"""
Price fetching and FX conversion.

Fetches latest closing prices from Yahoo Finance and converts to a target currency.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_prices(
    tickers: list[str], target_currency: str = "PLN"
) -> dict[str, float | None]:
    """
    Fetch latest closing prices for *tickers* converted to *target_currency*.

    Returns a dict mapping ticker → price in target currency (or None on failure).
    """
    if not tickers:
        return {}

    logger.info("Fetching prices for %d tickers...", len(tickers))

    # Download recent close prices
    try:
        close_data = yf.download(
            tickers, period="5d", auto_adjust=True, progress=False
        )["Close"]
    except Exception as exc:
        logger.error("Price download failed: %s", exc)
        return {}

    if isinstance(close_data, pd.Series):
        close_data = close_data.to_frame(name=tickers[0])

    last_prices = close_data.dropna(how="all").iloc[-1]

    # Determine currency for each ticker (parallelised)
    currencies: dict[str, str] = {}

    def _get_currency(ticker: str) -> tuple[str, str]:
        try:
            info = yf.Ticker(ticker).info
            return ticker, info.get("currency", target_currency).upper()
        except Exception as exc:
            logger.warning("Error fetching info for %s: %s", ticker, exc)
            return ticker, target_currency

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(_get_currency, t): t for t in tickers}
        for future in as_completed(futures):
            ticker, curr = future.result()
            currencies[ticker] = curr

    # Fetch FX rates for unique currencies (deduplicated, preserving ticker-list order)
    fx_rates: dict[str, float] = {target_currency: 1.0}
    seen_currencies: set[str] = set()
    unique_currencies: list[str] = []
    for t in tickers:
        curr = currencies.get(t, target_currency)
        if curr != target_currency and curr not in seen_currencies:
            seen_currencies.add(curr)
            unique_currencies.append(curr)

    for curr in unique_currencies:
        fx_ticker = f"{curr}{target_currency}=X"
        try:
            fx_data = yf.download(
                fx_ticker, period="5d", auto_adjust=True, progress=False, timeout=5
            )["Close"]
            fx_data = fx_data.dropna()
            if not fx_data.empty:
                fx_rates[curr] = float(fx_data.squeeze().iloc[-1])
                logger.info("FX %s/%s: %.4f", curr, target_currency, fx_rates[curr])
            else:
                logger.warning("Could not fetch %s/%s rate, using 1.0", curr, target_currency)
                fx_rates[curr] = 1.0
        except Exception as exc:
            logger.warning("FX fetch error for %s%s=X: %s, using 1.0", curr, target_currency, exc)
            fx_rates[curr] = 1.0

    # Build result
    result: dict[str, float | None] = {}
    for ticker in tickers:
        try:
            price = float(last_prices[ticker])
            curr = currencies.get(ticker, target_currency)
            fx = fx_rates.get(curr, 1.0)
            price_converted = round(price * fx, 2)
            result[ticker] = price_converted
            logger.info(
                "%s: %.2f %s x %.4f = %.2f %s",
                ticker, price, curr, fx, price_converted, target_currency,
            )
        except Exception:
            logger.warning("No price available for %s", ticker)
            result[ticker] = None

    return result


def get_fund_family(ticker: str) -> str:
    """Return the fund family string from yfinance (used for provider detection)."""
    try:
        return yf.Ticker(ticker).info.get("fundFamily", "")
    except Exception:
        return ""
