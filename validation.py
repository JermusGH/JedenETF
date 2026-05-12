"""
Validation utilities for the Unified ETF Portfolio tool.

Provides:
- validate_ticker: Ticker format validation (dot separator, whitespace, uppercase)
- is_valid_isin: ISIN format validation (12-char: 2 uppercase letters + 10 alphanumeric)
- validate_holdings_csv: CSV file validation for custom holdings uploads
"""

import re

import pandas as pd


# ---------------------------------------------------------------------------
# ISIN validation
# ---------------------------------------------------------------------------

_ISIN_PATTERN = re.compile(r"^[A-Z]{2}[A-Z0-9]{10}$")


def is_valid_isin(value: str) -> bool:
    """
    Validate that a string matches the 12-character ISIN format.

    A valid ISIN consists of exactly 2 uppercase letters followed by
    exactly 10 uppercase alphanumeric characters (total 12 characters).

    Parameters
    ----------
    value : str
        The string to validate.

    Returns
    -------
    bool
        True if the value matches the ISIN format, False otherwise.

    Validates: Requirements 3.1
    """
    if not isinstance(value, str):
        return False
    return bool(_ISIN_PATTERN.match(value))


# ---------------------------------------------------------------------------
# Ticker validation
# ---------------------------------------------------------------------------


def validate_ticker(ticker: str) -> tuple[str | None, str]:
    """
    Validate and clean a ticker symbol for the portfolio.

    Checks that the ticker contains a dot separator (exchange suffix),
    strips leading/trailing whitespace, and converts to uppercase.

    Parameters
    ----------
    ticker : str
        The raw ticker input from the user.

    Returns
    -------
    tuple[str | None, str]
        A tuple of (cleaned_ticker, error_message).
        - On success: (cleaned_ticker, "")
        - On failure: (None, error_message)

    Validates: Requirements 1.2, 1.7
    """
    if not ticker or not ticker.strip():
        return None, "Ticker cannot be empty."

    cleaned = ticker.strip().upper()

    if "." not in cleaned:
        return None, (
            "Ticker is missing an exchange suffix (e.g. `.L`, `.DE`, `.AS`). "
            "Please use the full Yahoo Finance ticker format."
        )

    return cleaned, ""


# ---------------------------------------------------------------------------
# CSV validation
# ---------------------------------------------------------------------------


def validate_holdings_csv(file) -> tuple[pd.DataFrame | None, str]:
    """
    Validate an uploaded CSV file for custom holdings.

    Expected format: CSV with columns 'ticker', 'name', 'weight'.
    - ticker: stock ticker (string)
    - name: company name (string, non-empty)
    - weight: percentage weight (numeric, non-negative, e.g. 8.5 means 8.5%)

    Rows with weight equal to zero are filtered out.

    Parameters
    ----------
    file : file-like object
        The uploaded CSV file to validate.

    Returns
    -------
    tuple[pd.DataFrame | None, str]
        A tuple of (DataFrame, error_message).
        - On success: (DataFrame with columns ticker/name/weight, "")
        - On failure: (None, error_message)

    Validates: Requirements 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9
    """
    # Requirement 7.4: If file cannot be parsed as CSV → error
    try:
        df = pd.read_csv(file)
    except Exception as exc:
        return None, f"Could not read CSV file: {exc}"

    # Requirement 7.3: Validate required columns present (case-insensitive, whitespace-trimmed)
    required_cols = {"ticker", "name", "weight"}
    actual_cols = {c.strip().lower() for c in df.columns}
    missing = required_cols - actual_cols
    if missing:
        return None, f"Missing required columns: {', '.join(sorted(missing))}. Expected: ticker, name, weight"

    # Normalise column names
    df.columns = [c.strip().lower() for c in df.columns]
    df = df[["ticker", "name", "weight"]].copy()

    # Validate data types
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["name"] = df["name"].astype(str).str.strip()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")

    # Requirement 7.5: Non-numeric weight values → error with affected row numbers
    if df["weight"].isna().any():
        bad_rows = df[df["weight"].isna()].index.tolist()
        return None, f"Non-numeric values in 'weight' column at rows: {bad_rows}"

    # Requirement 7.6: Negative weight values → error
    if (df["weight"] < 0).any():
        return None, "Negative values found in 'weight' column."

    # Requirement 7.7: Empty name values → error
    if df["name"].eq("").any() or df["name"].eq("nan").any():
        return None, "Empty values found in 'name' column."

    # Requirement 7.8: Filter out rows with weight equal to zero
    df = df[df["weight"] > 0].reset_index(drop=True)

    # Requirement 7.9: If all rows have weight zero → error
    if df.empty:
        return None, "No holdings with weight > 0 found in the file."

    return df, ""
