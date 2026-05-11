"""
Portfolio configuration and price fetching.

Edit the PORTFOLIO dict below to reflect your current ETF holdings.
Prices and FX rates are fetched live from Yahoo Finance.
"""

import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------------------
# PORTFOLIO — Yahoo Finance ticker → number of shares/units
# ---------------------------------------------------------------------------
PORTFOLIO: dict[str, float] = {
    "CSPX.L": 0.0,      # iShares Core S&P 500 UCITS
    "ISAC.L": 0.0,     # iShares MSCI ACWI UCITS
    "CNDX.L": 0.0,       # iShares NASDAQ 100 UCITS
    "VHVE.L": 0.0,          # Vanguard FTSE Developed World UCITS
    "SEC0.DE": 0.0,          # iShares MSCI Global Semiconductors UCITS
    "PRAM.DE": 0.0,         # Amundi Prime Emerging Markets UCITS
}

# Target currency for portfolio valuation
TARGET_CURRENCY = "PLN"


# ---------------------------------------------------------------------------
# Price fetching
# ---------------------------------------------------------------------------

def fetch_prices(tickers: list[str], target_currency: str = TARGET_CURRENCY) -> dict[str, float | None]:
    """
    Fetch latest closing prices for *tickers* converted to *target_currency*.

    Returns a dict mapping ticker → price in target currency (or None on failure).
    """
    if not tickers:
        return {}

    print(f"\n  Fetching prices from Yahoo Finance for {len(tickers)} tickers...")

    # Download recent close prices
    try:
        close_data = yf.download(tickers, period="5d", auto_adjust=True, progress=False)["Close"]
    except Exception as exc:
        print(f"  [!] Price download failed: {exc}")
        return {}

    if isinstance(close_data, pd.Series):
        close_data = close_data.to_frame(name=tickers[0])

    last_prices = close_data.dropna(how="all").iloc[-1]

    # Determine currency for each ticker and fetch FX rates
    currencies: dict[str, str] = {}
    fx_rates: dict[str, float] = {target_currency: 1.0}

    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            curr = info.get("currency", target_currency).upper()
            currencies[ticker] = curr

            if curr != target_currency and curr not in fx_rates:
                fx_ticker = f"{curr}{target_currency}=X"
                fx_data = yf.download(fx_ticker, period="5d", auto_adjust=True, progress=False)["Close"]
                fx_data = fx_data.dropna()
                if not fx_data.empty:
                    fx_rates[curr] = float(fx_data.squeeze().iloc[-1])
                    print(f"  FX {curr}/{target_currency}: {fx_rates[curr]:.4f}")
                else:
                    print(f"  [!] Could not fetch {curr}/{target_currency} rate, using 1.0")
                    fx_rates[curr] = 1.0
        except Exception as exc:
            print(f"  [!] Error for {ticker}: {exc}")
            currencies[ticker] = target_currency

    # Build result
    result: dict[str, float | None] = {}
    for ticker in tickers:
        try:
            price = float(last_prices[ticker])
            curr = currencies.get(ticker, target_currency)
            fx = fx_rates.get(curr, 1.0)
            price_converted = round(price * fx, 2)
            result[ticker] = price_converted
            print(f"  {ticker}: {price:.2f} {curr} x {fx:.4f} = {price_converted:.2f} {target_currency}")
        except Exception:
            print(f"  [!] No price available for {ticker}")
            result[ticker] = None

    return result


def get_fund_family(ticker: str) -> str:
    """Return the fund family string from yfinance (used for provider detection)."""
    try:
        return yf.Ticker(ticker).info.get("fundFamily", "")
    except Exception:
        return ""
