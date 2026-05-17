"""
Portfolio configuration.

Edit the PORTFOLIO dict below to reflect your current ETF holdings.
Runtime logic (price fetching, FX conversion) lives in pricing.py.
"""

# ---------------------------------------------------------------------------
# PORTFOLIO — Yahoo Finance ticker → number of shares/units
# ---------------------------------------------------------------------------
PORTFOLIO: dict[str, float] = {
    "CSPX.L": 0.0,      # iShares Core S&P 500 UCITS
    "ISAC.L": 0.0,      # iShares MSCI ACWI UCITS
    "CNDX.L": 0.0,      # iShares NASDAQ 100 UCITS
    "VHVE.L": 0.0,      # Vanguard FTSE Developed World UCITS
    "SEC0.DE": 0.0,     # iShares MSCI Global Semiconductors UCITS
    "PRAM.DE": 0.0,     # Amundi Prime Emerging Markets UCITS
}

# Target currency for portfolio valuation
TARGET_CURRENCY = "PLN"

# Supported currencies for target currency selection (web UI and validation)
SUPPORTED_CURRENCIES: list[str] = ["PLN", "EUR", "USD", "GBP", "CHF"]
