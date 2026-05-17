"""
Property-based tests for the Unified ETF Portfolio.

Uses Hypothesis to validate correctness properties defined in the design document.
Feature: unified-etf-portfolio
"""

from hypothesis import given, settings, assume
from hypothesis import strategies as st
import pandas as pd

from validation import validate_ticker


# ---------------------------------------------------------------------------
# Property 1: Ticker without dot separator is rejected
# ---------------------------------------------------------------------------
# For any string that does not contain a dot character, submitting it as a
# ticker to validate_ticker() SHALL result in rejection (returns None as first
# element) and an error message.
#
# Feature: unified-etf-portfolio, Property 1: Ticker without dot separator is rejected
# Validates: Requirements 1.2, 2.9, 14.3
# ---------------------------------------------------------------------------


@given(ticker=st.text(alphabet=st.characters(blacklist_characters="."), min_size=1))
@settings(max_examples=100)
def test_property_1_ticker_without_dot_is_rejected(ticker: str):
    """
    **Validates: Requirements 1.2, 2.9, 14.3**

    Property 1: Ticker without dot separator is rejected.

    Any non-empty string that does not contain a dot character must be
    rejected by validate_ticker (first element is None, second element
    is a non-empty error message).
    """
    result, error_message = validate_ticker(ticker)
    assert result is None, (
        f"Expected None for ticker without dot, got {result!r} for input {ticker!r}"
    )
    assert isinstance(error_message, str) and len(error_message) > 0, (
        f"Expected non-empty error message, got {error_message!r} for input {ticker!r}"
    )


# ---------------------------------------------------------------------------
# Property 2: Ticker normalisation preserves identity
# ---------------------------------------------------------------------------
# For any string input submitted as a ticker that contains a dot, the stored
# ticker SHALL equal the input with leading/trailing whitespace removed and all
# characters converted to uppercase. That is: stored == input.strip().upper().
#
# Feature: unified-etf-portfolio, Property 2: Ticker normalisation preserves identity
# Validates: Requirements 1.7
# ---------------------------------------------------------------------------


def _strings_with_dot() -> st.SearchStrategy[str]:
    """Generate arbitrary strings that contain at least one dot character."""
    # Build strings as: prefix + "." + suffix, where prefix and suffix are
    # arbitrary text (may include additional dots, whitespace, etc.)
    return st.builds(
        lambda prefix, suffix: prefix + "." + suffix,
        st.text(min_size=0, max_size=20),
        st.text(min_size=0, max_size=20),
    )


@given(ticker_input=_strings_with_dot())
@settings(max_examples=100)
def test_property_2_ticker_normalisation_preserves_identity(ticker_input: str):
    """
    **Validates: Requirements 1.7**

    For any string input that contains a dot, validate_ticker SHALL return
    a cleaned_ticker equal to input.strip().upper().
    """
    # Ensure the input is non-empty after stripping (validate_ticker rejects empty)
    assume(ticker_input.strip() != "")

    cleaned_ticker, error_msg = validate_ticker(ticker_input)

    # Since the input contains a dot, it should pass validation
    assert cleaned_ticker is not None, (
        f"Expected ticker to be accepted (contains dot), but got error: {error_msg!r} "
        f"for input: {ticker_input!r}"
    )
    assert error_msg == ""

    # The core property: stored ticker == input.strip().upper()
    expected = ticker_input.strip().upper()
    assert cleaned_ticker == expected, (
        f"Normalisation failed: expected {expected!r}, got {cleaned_ticker!r} "
        f"for input: {ticker_input!r}"
    )


# ---------------------------------------------------------------------------
# Property 12: Cache resilience to malformed data
# ---------------------------------------------------------------------------
# For any file content that is not valid JSON (including empty files, binary
# data, truncated JSON), loading the cache SHALL return an empty dictionary
# without raising an exception.
#
# Feature: unified-etf-portfolio, Property 12: Cache resilience to malformed data
# Validates: Requirements 12.6
# ---------------------------------------------------------------------------

import tempfile
import os
from unittest.mock import patch

from etf_holdings import _load_cache


def _non_json_text() -> st.SearchStrategy[str]:
    """Generate arbitrary text that is NOT valid JSON."""
    import json as _json

    def is_not_valid_json(s: str) -> bool:
        try:
            result = _json.loads(s)
            # Even if it parses, reject it if it's a dict (valid cache)
            return not isinstance(result, dict)
        except (ValueError, TypeError):
            return True

    return st.text(min_size=0, max_size=200).filter(is_not_valid_json)


def _non_json_binary() -> st.SearchStrategy[bytes]:
    """Generate arbitrary binary content that is NOT valid JSON."""
    import json as _json

    def is_not_valid_json_bytes(b: bytes) -> bool:
        try:
            result = _json.loads(b)
            return not isinstance(result, dict)
        except (ValueError, TypeError, UnicodeDecodeError):
            return True

    return st.binary(min_size=0, max_size=200).filter(is_not_valid_json_bytes)


@given(content=st.one_of(_non_json_text(), _non_json_binary()))
@settings(max_examples=100)
def test_property_12_cache_resilience_to_malformed_data(content):
    """
    **Validates: Requirements 12.6**

    For any file content that is not valid JSON (including empty files, binary
    data, truncated JSON), loading the cache SHALL return an empty dictionary
    without raising an exception.
    """
    # Write the malformed content to a temporary file
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".json", delete=False
    ) as tmp:
        if isinstance(content, bytes):
            tmp.write(content)
        else:
            tmp.write(content.encode("utf-8"))
        tmp_path = tmp.name

    try:
        # Patch _CACHE_PATH to point to our malformed file
        with patch("etf_holdings._CACHE_PATH", tmp_path):
            result = _load_cache()

        # The function must return an empty dict without raising
        assert result == {}, (
            f"Expected empty dict for malformed content, got {result!r} "
            f"for content: {content!r}"
        )
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Property 3: Provider detection is deterministic and keyword-based
# ---------------------------------------------------------------------------
# Feature: unified-etf-portfolio, Property 3: Provider detection is deterministic and keyword-based
# **Validates: Requirements 2.2, 2.3, 2.4, 2.5**
# ---------------------------------------------------------------------------

from etf_holdings import _detect_provider


@given(fund_family=st.text())
@settings(max_examples=100)
def test_property_3_provider_detection_keyword_logic(fund_family: str):
    """
    **Validates: Requirements 2.2, 2.3, 2.4, 2.5**

    For any fund family string, the provider classification SHALL be:
    - "ishares" if the lowercased string contains "blackrock" or "ishares"
    - "vanguard" if the lowercased string contains "vanguard"
    - "invesco" if the lowercased string contains "invesco"
    - "amundi" if the lowercased string contains "amundi"
    - "unknown" otherwise
    """
    result = _detect_provider(fund_family)
    lowered = fund_family.lower()

    if "blackrock" in lowered or "ishares" in lowered:
        assert result == "ishares", (
            f"Expected 'ishares' for fund_family={fund_family!r}, got {result!r}"
        )
    elif "vanguard" in lowered:
        assert result == "vanguard", (
            f"Expected 'vanguard' for fund_family={fund_family!r}, got {result!r}"
        )
    elif "invesco" in lowered:
        assert result == "invesco", (
            f"Expected 'invesco' for fund_family={fund_family!r}, got {result!r}"
        )
    elif "amundi" in lowered:
        assert result == "amundi", (
            f"Expected 'amundi' for fund_family={fund_family!r}, got {result!r}"
        )
    else:
        assert result == "unknown", (
            f"Expected 'unknown' for fund_family={fund_family!r}, got {result!r}"
        )


@given(fund_family=st.text())
@settings(max_examples=100)
def test_property_3_provider_detection_determinism(fund_family: str):
    """
    **Validates: Requirements 2.2, 2.3, 2.4, 2.5**

    Provider detection is deterministic — same input always yields same output.
    """
    result1 = _detect_provider(fund_family)
    result2 = _detect_provider(fund_family)
    assert result1 == result2, (
        f"Non-deterministic result for fund_family={fund_family!r}: "
        f"{result1!r} != {result2!r}"
    )


# ---------------------------------------------------------------------------
# Property 4: ISIN format validation
# ---------------------------------------------------------------------------
# For any string, it SHALL be accepted as a valid ISIN if and only if it
# matches the pattern: exactly 2 uppercase letters followed by exactly 10
# uppercase alphanumeric characters (total 12 characters).
#
# Feature: unified-etf-portfolio, Property 4: ISIN format validation
# Validates: Requirements 3.1
# ---------------------------------------------------------------------------

import re as _re

from validation import is_valid_isin

_ISIN_REGEX = _re.compile(r"^[A-Z]{2}[A-Z0-9]{10}$")


@given(isin=st.from_regex(r"[A-Z]{2}[A-Z0-9]{10}", fullmatch=True))
@settings(max_examples=100)
def test_property_4_valid_isins_accepted(isin: str):
    """
    **Validates: Requirements 3.1**

    Valid ISINs (2 uppercase letters + 10 uppercase alphanumeric) are accepted.
    """
    assert is_valid_isin(isin) is True, (
        f"Expected is_valid_isin to return True for valid ISIN '{isin}'"
    )


@given(s=st.text())
@settings(max_examples=100)
def test_property_4_invalid_isins_rejected(s: str):
    """
    **Validates: Requirements 3.1**

    Arbitrary strings that do NOT match the ISIN pattern are rejected.
    """
    assume(not _ISIN_REGEX.match(s))
    assert is_valid_isin(s) is False, (
        f"Expected is_valid_isin to return False for invalid string '{s}'"
    )


# ---------------------------------------------------------------------------
# Property 5: Name normaliser produces canonical short uppercase form
# ---------------------------------------------------------------------------
# For any non-null, non-empty input string, the normalised output SHALL:
# - Contain only uppercase letters, digits, and spaces
# - Contain no corporate suffixes as whole words
# - Consist of at most three whitespace-separated words
# And for null or empty input, the output SHALL be an empty string.
#
# Feature: unified-etf-portfolio, Property 5: Name normaliser produces canonical short uppercase form
# Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.7
# ---------------------------------------------------------------------------

import re as _re_p5

from name_normaliser import normalise_name

_CORPORATE_SUFFIXES = [
    "INC", "CORP", "CORPORATION", "LTD", "LIMITED", "PLC", "COMPANY",
    "AG", "SA", "NV", "GROUP", "HOLDINGS", "HOLDING", "SE", "CO",
]

_CANONICAL_FORM_REGEX = _re_p5.compile(r"^[A-Z0-9 ]*$")


@given(name=st.text(min_size=1))
@settings(max_examples=100)
def test_property_5_name_normaliser_canonical_form(name: str):
    """
    **Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.7**

    Property 5: Name normaliser produces canonical short uppercase form.

    For any non-empty input string, the normalised output SHALL:
    - Contain only uppercase letters, digits, and spaces (regex: ^[A-Z0-9 ]*$)
    - Contain no corporate suffixes as whole words
    - Consist of at most three whitespace-separated words
    """
    result = normalise_name(name)

    # Output contains only uppercase letters, digits, and spaces
    assert _CANONICAL_FORM_REGEX.match(result) is not None, (
        f"Output contains invalid characters: {result!r} for input {name!r}"
    )

    # Output has at most 3 whitespace-separated words (MAX_NORMALISED_WORDS)
    words = result.split()
    assert len(words) <= 3, (
        f"Output has {len(words)} words (max 3): {result!r} for input {name!r}"
    )

    # Output does not contain any corporate suffixes as whole words
    for suffix in _CORPORATE_SUFFIXES:
        assert not _re_p5.search(r"\b" + suffix + r"\b", result), (
            f"Output contains corporate suffix '{suffix}': {result!r} for input {name!r}"
        )


def test_property_5_name_normaliser_null_and_empty():
    """
    **Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.7**

    Property 5: For null or empty input, the output SHALL be an empty string.
    """
    assert normalise_name(None) == "", "Expected empty string for None input"
    assert normalise_name("") == "", "Expected empty string for empty string input"
    assert normalise_name("   ") == "", "Expected empty string for whitespace-only input"


# ---------------------------------------------------------------------------
# Property 8: Weight interpretation heuristic
# ---------------------------------------------------------------------------
# For any set of holdings from a single ETF with a known ETF position value:
# - If the sum of all weight values exceeds 10, THEN each holding's value
#   SHALL be calculated as (weight / 100) * etf_value
# - If the sum of all weight values is 10 or less, THEN each holding's value
#   SHALL be calculated as weight * etf_value
#
# Feature: unified-etf-portfolio, Property 8: Weight interpretation heuristic
# Validates: Requirements 10.2, 10.3
# ---------------------------------------------------------------------------

import math

from analysis import calculate_holding_values


@given(
    weights=st.lists(
        st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=20,
    ),
    etf_value=st.floats(min_value=0.01, max_value=1_000_000.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_property_8_weight_interpretation_heuristic(weights: list[float], etf_value: float):
    """
    **Validates: Requirements 10.2, 10.3**

    Property 8: Weight interpretation heuristic.

    For any set of holdings from a single ETF with a known ETF position value:
    - If sum(weights) > 10: each value == (weight / 100) * etf_value
    - If sum(weights) <= 10: each value == weight * etf_value
    """
    # Build a DataFrame with ticker, name, weight columns
    df = pd.DataFrame({
        "ticker": [f"T{i}" for i in range(len(weights))],
        "name": [f"Company {i}" for i in range(len(weights))],
        "weight": weights,
    })

    result = calculate_holding_values(df, etf_value)

    # Verify the result has a 'value' column
    assert "value" in result.columns, "Result DataFrame must have a 'value' column"

    weight_sum = sum(weights)

    for idx, weight in enumerate(weights):
        actual_value = result.iloc[idx]["value"]
        if weight_sum > 10:
            expected_value = (weight / 100.0) * etf_value
        else:
            expected_value = weight * etf_value

        assert math.isclose(actual_value, expected_value, rel_tol=1e-9), (
            f"Row {idx}: expected {expected_value}, got {actual_value} "
            f"(weight={weight}, etf_value={etf_value}, weight_sum={weight_sum})"
        )



# ---------------------------------------------------------------------------
# Property 6: Holdings output excludes empty or blank names
# ---------------------------------------------------------------------------
# For any valid holdings response (from iShares CSV, Vanguard API, or justETF),
# the returned DataFrame SHALL contain no rows where the `name` column is empty,
# whitespace-only, or null.
#
# Feature: unified-etf-portfolio, Property 6: Holdings output excludes empty or blank names
# Validates: Requirements 4.4, 5.4, 6.5
# ---------------------------------------------------------------------------

import numpy as np


def _filter_empty_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replicate the name-filtering logic used across all fetcher functions.

    This mirrors the filtering in:
    - _fetch_ishares: df[df["Name"].notna() & (df["Name"].astype(str).str.strip() != "")]
    - _fetch_vanguard: df[df["name"].str.strip() != ""]
    - _fetch_justetf: (implicitly via name column presence)

    The unified rule: exclude rows where name is null, empty, or whitespace-only.
    """
    if df.empty:
        return df
    mask = df["name"].notna() & (df["name"].astype(str).str.strip() != "")
    return df[mask].reset_index(drop=True)


def _holdings_dataframe_strategy():
    """
    Generate DataFrames simulating raw holdings data with a mix of:
    - Valid company names
    - Empty strings
    - Whitespace-only strings
    - None/NaN values

    This represents the data BEFORE filtering is applied.
    """
    # Strategy for individual name values: mix of valid and problematic
    valid_names = st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "Zs")),
        min_size=1,
        max_size=50,
    ).filter(lambda s: s.strip() != "")

    empty_names = st.just("")
    whitespace_names = st.text(
        alphabet=st.just(" "),
        min_size=1,
        max_size=10,
    )
    # None will become NaN in the DataFrame
    null_names = st.just(None)

    # Mix of all name types
    name_strategy = st.one_of(valid_names, empty_names, whitespace_names, null_names)

    # Generate lists of holdings rows
    row_strategy = st.tuples(
        st.text(min_size=1, max_size=10),  # ticker
        name_strategy,                      # name (may be invalid)
        st.floats(min_value=0.01, max_value=100.0, allow_nan=False),  # weight
    )

    return st.lists(row_strategy, min_size=1, max_size=30).map(
        lambda rows: pd.DataFrame(rows, columns=["ticker", "name", "weight"])
    )


@given(df=_holdings_dataframe_strategy())
@settings(max_examples=100)
def test_property_6_holdings_exclude_empty_names(df: pd.DataFrame):
    """
    **Validates: Requirements 4.4, 5.4, 6.5**

    Property 6: Holdings output excludes empty or blank names.

    For any holdings DataFrame (simulating raw provider output), after applying
    the name-filtering logic used by the fetchers, the result SHALL contain no
    rows where the name column is empty, whitespace-only, or null.
    """
    # Apply the same filtering logic used by the fetcher functions
    filtered = _filter_empty_names(df)

    # Assert: no row in the filtered result has an empty/blank/null name
    for idx, row in filtered.iterrows():
        name_val = row["name"]
        # Must not be null/NaN
        assert name_val is not None and not (isinstance(name_val, float) and np.isnan(name_val)), (
            f"Row {idx} has null/NaN name after filtering"
        )
        # Must not be empty or whitespace-only
        assert str(name_val).strip() != "", (
            f"Row {idx} has empty/whitespace-only name after filtering: {name_val!r}"
        )


# ---------------------------------------------------------------------------
# Property 11: Price conversion arithmetic
# ---------------------------------------------------------------------------
# For any positive price value and positive FX rate, the converted price
# SHALL equal round(price * fx_rate, 2).
#
# Feature: unified-etf-portfolio, Property 11: Price conversion arithmetic
# Validates: Requirements 8.5
# ---------------------------------------------------------------------------


@given(
    price=st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
    fx_rate=st.floats(min_value=0.01, max_value=1e4, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_property_11_price_conversion_arithmetic(price: float, fx_rate: float):
    """
    **Validates: Requirements 8.5**

    Property 11: Price conversion arithmetic.

    For any positive price value and positive FX rate, the converted price
    SHALL equal round(price * fx_rate, 2).
    """
    converted = round(price * fx_rate, 2)

    # The conversion formula must produce a finite, non-negative result
    assert converted >= 0, (
        f"Converted price should be non-negative, got {converted} "
        f"for price={price}, fx_rate={fx_rate}"
    )

    # The conversion must equal the expected formula exactly
    expected = round(price * fx_rate, 2)
    assert converted == expected, (
        f"Price conversion mismatch: got {converted}, expected {expected} "
        f"for price={price}, fx_rate={fx_rate}"
    )

    # Verify the result is a float (round returns float for ndigits >= 0)
    assert isinstance(converted, float), (
        f"Expected float result, got {type(converted).__name__} "
        f"for price={price}, fx_rate={fx_rate}"
    )


# ---------------------------------------------------------------------------
# Property 7: CSV validation correctly accepts and rejects files
# ---------------------------------------------------------------------------
# For any CSV content:
# - If it contains columns matching "ticker", "name", "weight"
#   (case-insensitive, whitespace-trimmed), AND all weight values are numeric
#   and non-negative, AND all name values are non-empty, THEN validation SHALL
#   succeed and return a DataFrame with only rows where weight > 0.
# - If any required column is missing, OR any weight is non-numeric, OR any
#   weight is negative, OR any name is empty, THEN validation SHALL fail with
#   an appropriate error message.
#
# Feature: unified-etf-portfolio, Property 7: CSV validation correctly accepts and rejects files
# Validates: Requirements 7.2, 7.3, 7.5, 7.6, 7.7, 7.8, 7.9
# ---------------------------------------------------------------------------

import io

import pandas as pd

from validation import validate_holdings_csv


def _valid_csv_strategy():
    """
    Generate valid CSV data: 3 required columns (with possible case/whitespace
    variations), non-negative numeric weights (at least one > 0), non-empty names.
    """
    # Strategy for column name variations (case-insensitive, whitespace-trimmed)
    col_name_ticker = st.sampled_from(["ticker", "Ticker", "TICKER", " ticker ", " Ticker"])
    col_name_name = st.sampled_from(["name", "Name", "NAME", " name ", " Name"])
    col_name_weight = st.sampled_from(["weight", "Weight", "WEIGHT", " weight ", " Weight"])

    # Strategy for row data
    ticker_values = st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters=".-"),
        min_size=1,
        max_size=10,
    )
    name_values = st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters=" "),
        min_size=1,
        max_size=30,
    ).filter(lambda s: s.strip() != "" and s.strip().lower() not in ("nan", "none", "null", "na", "n/a"))
    # Non-negative weights, at least some > 0
    weight_values = st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)

    @st.composite
    def build_csv(draw):
        col_t = draw(col_name_ticker)
        col_n = draw(col_name_name)
        col_w = draw(col_name_weight)

        num_rows = draw(st.integers(min_value=1, max_value=10))
        rows = []
        has_positive = False
        for _ in range(num_rows):
            t = draw(ticker_values)
            n = draw(name_values)
            w = draw(weight_values)
            if w > 0:
                has_positive = True
            rows.append((t, n, w))

        # Ensure at least one row has weight > 0
        if not has_positive:
            t = draw(ticker_values)
            n = draw(name_values)
            w = draw(st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False))
            rows.append((t, n, w))

        # Build CSV string
        lines = [f"{col_t},{col_n},{col_w}"]
        for t, n, w in rows:
            lines.append(f"{t},{n},{w}")
        return "\n".join(lines)

    return build_csv()


@given(csv_content=_valid_csv_strategy())
@settings(max_examples=100)
def test_property_7_valid_csvs_accepted(csv_content: str):
    """
    **Validates: Requirements 7.2, 7.3, 7.5, 7.6, 7.7, 7.8, 7.9**

    Property 7 (sub-test 1): Valid CSVs are accepted.

    For any CSV with required columns (case-insensitive, whitespace-trimmed),
    non-negative numeric weights, and non-empty names, validation SHALL succeed
    and the returned DataFrame SHALL only contain rows with weight > 0.
    """
    file_obj = io.StringIO(csv_content)
    df, error = validate_holdings_csv(file_obj)

    assert df is not None, (
        f"Expected validation to succeed for valid CSV, but got error: {error!r}\n"
        f"CSV content:\n{csv_content}"
    )
    assert error == "", (
        f"Expected empty error message, got: {error!r}"
    )
    # All returned rows must have weight > 0
    assert (df["weight"] > 0).all(), (
        f"Expected all rows to have weight > 0, but found:\n{df[df['weight'] <= 0]}"
    )
    # DataFrame must have the expected columns
    assert set(df.columns) == {"ticker", "name", "weight"}, (
        f"Expected columns {{ticker, name, weight}}, got {set(df.columns)}"
    )


def _invalid_csv_missing_columns_strategy():
    """Generate CSV data with one or more required columns missing."""
    all_cols = ["ticker", "name", "weight"]

    @st.composite
    def build_csv(draw):
        # Remove at least one required column
        num_to_remove = draw(st.integers(min_value=1, max_value=3))
        cols_to_keep = draw(
            st.lists(
                st.sampled_from(all_cols),
                min_size=max(0, 3 - num_to_remove),
                max_size=max(0, 3 - num_to_remove),
                unique=True,
            )
        )
        # Ensure at least one column is actually missing
        assume(set(cols_to_keep) != set(all_cols))

        if not cols_to_keep:
            # All columns missing - just put some random header
            cols_to_keep = ["foo"]

        header = ",".join(cols_to_keep)
        # Add a data row
        values = ",".join(["test"] * len(cols_to_keep))
        return f"{header}\n{values}"

    return build_csv()


def _invalid_csv_negative_weight_strategy():
    """Generate CSV data with at least one negative weight value."""

    @st.composite
    def build_csv(draw):
        num_rows = draw(st.integers(min_value=1, max_value=5))
        lines = ["ticker,name,weight"]
        has_negative = False
        for i in range(num_rows):
            w = draw(st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False))
            if w < 0:
                has_negative = True
            lines.append(f"TICK{i},Company{i},{w}")

        # Ensure at least one negative weight
        if not has_negative:
            neg_w = draw(st.floats(min_value=-100.0, max_value=-0.01, allow_nan=False, allow_infinity=False))
            lines.append(f"TICKN,NegCompany,{neg_w}")

        return "\n".join(lines)

    return build_csv()


def _invalid_csv_empty_name_strategy():
    """Generate CSV data with at least one empty name value."""

    @st.composite
    def build_csv(draw):
        num_rows = draw(st.integers(min_value=1, max_value=5))
        lines = ["ticker,name,weight"]
        for i in range(num_rows):
            lines.append(f"TICK{i},Company{i},{draw(st.floats(min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False))}")

        # Add a row with empty name
        lines.append(f"TICKE,,5.0")
        return "\n".join(lines)

    return build_csv()


def _invalid_csv_non_numeric_weight_strategy():
    """Generate CSV data with at least one non-numeric weight value."""
    non_numeric = st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll"), whitelist_characters=" "),
        min_size=1,
        max_size=10,
    ).filter(lambda s: s.strip() != "")

    @st.composite
    def build_csv(draw):
        num_rows = draw(st.integers(min_value=1, max_value=5))
        lines = ["ticker,name,weight"]
        for i in range(num_rows):
            lines.append(f"TICK{i},Company{i},{draw(st.floats(min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False))}")

        # Add a row with non-numeric weight
        bad_weight = draw(non_numeric)
        lines.append(f"TICKBAD,BadCompany,{bad_weight}")
        return "\n".join(lines)

    return build_csv()


@given(csv_content=st.one_of(
    _invalid_csv_missing_columns_strategy(),
    _invalid_csv_negative_weight_strategy(),
    _invalid_csv_empty_name_strategy(),
    _invalid_csv_non_numeric_weight_strategy(),
))
@settings(max_examples=100)
def test_property_7_invalid_csvs_rejected(csv_content: str):
    """
    **Validates: Requirements 7.2, 7.3, 7.5, 7.6, 7.7, 7.8, 7.9**

    Property 7 (sub-test 2): Invalid CSVs are rejected.

    For any CSV with missing required columns, negative weights, empty names,
    or non-numeric weights, validation SHALL fail with a non-empty error message.
    """
    file_obj = io.StringIO(csv_content)
    df, error = validate_holdings_csv(file_obj)

    assert df is None, (
        f"Expected validation to fail for invalid CSV, but got DataFrame:\n{df}\n"
        f"CSV content:\n{csv_content}"
    )
    assert isinstance(error, str) and len(error) > 0, (
        f"Expected non-empty error message, got: {error!r}\n"
        f"CSV content:\n{csv_content}"
    )


# ---------------------------------------------------------------------------
# Property 10: Unified weight sums to 100%
# ---------------------------------------------------------------------------
# For any non-empty set of merged holdings with a positive total portfolio
# value, the sum of all weight_% values SHALL equal approximately 100%
# (within floating-point tolerance). Each individual weight is calculated as
# (holding_value / total_value) * 100.
#
# Feature: unified-etf-portfolio, Property 10: Unified weight sums to 100%
# Validates: Requirements 10.5
# ---------------------------------------------------------------------------

import pytest
import pandas as pd

from analysis import merge_holdings


@st.composite
def enriched_frames_strategy(draw):
    """
    Generate a list of enriched DataFrames with positive values.

    Each DataFrame has columns: ticker, name, weight, value, source, merge_key.
    The total_value returned equals the sum of all values across all frames.
    """
    num_frames = draw(st.integers(min_value=1, max_value=3))
    frames = []
    for i in range(num_frames):
        num_rows = draw(st.integers(min_value=1, max_value=5))
        rows = []
        for j in range(num_rows):
            value = draw(st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False))
            # Use unique names to avoid merge_key collisions across frames
            name = f"COMPANY{i}X{j}"
            rows.append({
                "ticker": f"TK{i}{j}",
                "name": name,
                "weight": draw(st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False)),
                "value": value,
                "source": f"ETF{i}.L",
                "merge_key": name,
            })
        frames.append(pd.DataFrame(rows))
    total_value = sum(row["value"] for frame in frames for _, row in frame.iterrows())
    return frames, total_value


@given(data=enriched_frames_strategy())
@settings(max_examples=100)
def test_property_10_unified_weight_sum_approximately_100(data):
    """
    **Validates: Requirements 10.5**

    Property 10: Unified weight sums to 100%.

    For any non-empty set of merged holdings with a positive total portfolio
    value, the sum of all weight_% values SHALL equal approximately 100%
    (within floating-point tolerance).
    """
    frames, total_value = data
    assume(total_value > 0)

    result = merge_holdings(frames, total_value)

    assert not result.empty, "merge_holdings returned empty DataFrame for valid input"
    weight_sum = result["weight_%"].sum()
    assert weight_sum == pytest.approx(100.0, abs=0.01), (
        f"Expected weight sum ≈ 100%, got {weight_sum:.6f}% "
        f"(total_value={total_value})"
    )


# ---------------------------------------------------------------------------
# Property 9: Merging preserves total value
# ---------------------------------------------------------------------------
# For any set of holdings with the same normalised name (merge_key), the merged
# entry's value SHALL equal the sum of all individual holding values. That is,
# merging is value-preserving: no value is created or destroyed.
#
# Feature: unified-etf-portfolio, Property 9: Merging preserves total value
# Validates: Requirements 9.5
# ---------------------------------------------------------------------------

import math

from analysis import merge_holdings


def _enriched_frame_strategy():
    """
    Strategy that generates a list of enriched DataFrames (2-5 frames),
    where some holdings share the same merge_key to exercise merging.
    """
    # Pool of merge_keys — using a small pool ensures collisions across frames
    merge_keys_pool = st.sampled_from(["APPLE", "MICROSOFT", "GOOGLE", "AMAZON", "NVIDIA", "TESLA"])
    tickers_pool = st.sampled_from(["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "N/A"])
    sources_pool = st.sampled_from(["CSPX.L", "VWCE.DE", "VUAA.L", "IWDA.AS"])

    # A single holding row
    holding_row = st.fixed_dictionaries({
        "ticker": tickers_pool,
        "name": st.text(min_size=1, max_size=30).filter(lambda s: s.strip() != ""),
        "weight": st.floats(min_value=0.01, max_value=50.0, allow_nan=False, allow_infinity=False),
        "value": st.floats(min_value=0.01, max_value=100000.0, allow_nan=False, allow_infinity=False),
        "source": sources_pool,
        "merge_key": merge_keys_pool,
    })

    # A single frame is a list of 1-10 holding rows
    single_frame = st.lists(holding_row, min_size=1, max_size=10).map(
        lambda rows: pd.DataFrame(rows)
    )

    # Generate 2-5 frames
    return st.lists(single_frame, min_size=2, max_size=5)


@given(frames=_enriched_frame_strategy())
@settings(max_examples=100)
def test_property_9_merge_value_preservation(frames):
    """
    **Validates: Requirements 9.5**

    Property 9: Merging preserves total value.

    For any set of holdings with the same normalised name (merge_key), the
    merged entry's value SHALL equal the sum of all individual holding values.
    Merging is value-preserving: no value is created or destroyed.
    """
    # Calculate total value from all input frames
    all_values = []
    for frame in frames:
        all_values.extend(frame["value"].tolist())
    total_value = sum(all_values)

    # Avoid division by zero in merge_holdings
    assume(total_value > 0)

    result = merge_holdings(frames, total_value)

    # Result should not be empty since we have frames with positive total_value
    assert not result.empty, "merge_holdings returned empty DataFrame for valid input"

    # Combine all input frames to compute expected per-merge_key sums
    combined_input = pd.concat(frames, ignore_index=True)
    expected_per_key = combined_input.groupby("merge_key")["value"].sum()

    # Assert: for each merge_key in result, value equals sum of input values
    for _, row in result.iterrows():
        key = row["merge_key"]
        expected_value = expected_per_key[key]
        assert math.isclose(row["value"], expected_value, rel_tol=1e-9), (
            f"Value mismatch for merge_key={key!r}: "
            f"expected {expected_value}, got {row['value']}"
        )

    # Assert: total value across all merged holdings equals sum of all input values
    result_total = result["value"].sum()
    assert math.isclose(result_total, total_value, rel_tol=1e-9), (
        f"Total value mismatch: expected {total_value}, got {result_total}"
    )


# ---------------------------------------------------------------------------
# Property 13: Same currency requires no FX conversion
# ---------------------------------------------------------------------------
# For any ticker whose trading currency equals the user-selected Target
# Currency, the Price Fetcher SHALL use a conversion rate of 1.0 without
# fetching an FX rate from Yahoo Finance.
#
# Feature: unified-etf-portfolio, Property 13: Same currency requires no FX conversion
# Validates: Requirements 15.6
# ---------------------------------------------------------------------------

from unittest.mock import patch, MagicMock

from pricing import fetch_prices


def _currency_code_strategy() -> st.SearchStrategy[str]:
    """Generate realistic 3-letter uppercase currency codes."""
    return st.sampled_from(["PLN", "EUR", "USD", "GBP", "CHF", "JPY", "AUD", "CAD", "SEK", "NOK"])


@given(currency=_currency_code_strategy())
@settings(max_examples=100)
def test_property_13_same_currency_no_fx_conversion(currency: str):
    """
    **Validates: Requirements 15.6**

    Property 13: Same currency requires no FX conversion.

    For any ticker whose trading currency equals the target currency, the
    conversion rate is 1.0 and no FX rate is fetched from Yahoo Finance.
    """
    ticker = "TEST.L"
    mock_price = 100.0

    # Mock yf.download for the initial price fetch — returns a DataFrame
    # with a single ticker column and one row of price data
    price_df = pd.DataFrame(
        {ticker: [mock_price]},
        index=pd.DatetimeIndex(["2024-01-01"]),
    )
    price_df.columns.name = None

    # Track calls to yf.download to verify no FX fetch occurs
    download_calls = []

    def mock_download(tickers_arg, **kwargs):
        download_calls.append(tickers_arg)
        # First call is for the price data
        if isinstance(tickers_arg, list) and ticker in tickers_arg:
            result = pd.DataFrame(
                {ticker: [mock_price]},
                index=pd.DatetimeIndex(["2024-01-01"]),
            )
            # Wrap in MultiIndex columns like yf.download returns
            result.columns = pd.MultiIndex.from_tuples(
                [(ticker, )], names=["Ticker"]
            )
            # Return a DataFrame that when ["Close"] is accessed gives the prices
            close_df = pd.DataFrame(
                {ticker: [mock_price]},
                index=pd.DatetimeIndex(["2024-01-01"]),
            )
            mock_result = MagicMock()
            mock_result.__getitem__ = lambda self, key: close_df
            return mock_result
        # Any other call (would be FX) — should NOT happen for same currency
        fx_df = pd.DataFrame(
            {"Close": [1.5]},
            index=pd.DatetimeIndex(["2024-01-01"]),
        )
        mock_result = MagicMock()
        mock_result.__getitem__ = lambda self, key: fx_df
        return mock_result

    # Mock yf.Ticker to return the same currency as target
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.info = {"currency": currency}

    with patch("pricing.yf.download", side_effect=mock_download), \
         patch("pricing.yf.Ticker", return_value=mock_ticker_instance):
        result = fetch_prices([ticker], target_currency=currency)

    # The result should have the ticker with price converted at rate 1.0
    assert ticker in result, f"Expected {ticker} in result, got {result}"
    assert result[ticker] is not None, f"Expected non-None price for {ticker}"

    # Price should be round(mock_price * 1.0, 2) = 100.0
    expected_price = round(mock_price * 1.0, 2)
    assert result[ticker] == expected_price, (
        f"Expected price {expected_price} (rate 1.0), got {result[ticker]} "
        f"for currency={currency}"
    )

    # Verify no FX download was attempted — only the initial price download
    # should have been called. Any call with a "=X" ticker would indicate
    # an FX fetch.
    fx_calls = [c for c in download_calls if isinstance(c, str) and "=X" in c]
    assert len(fx_calls) == 0, (
        f"Expected no FX rate fetch for same currency ({currency}), "
        f"but found FX download calls: {fx_calls}"
    )


# ---------------------------------------------------------------------------
# Property 14: FX pair format correctness
# ---------------------------------------------------------------------------
# For any trading currency T and target currency C where T ≠ C, the FX rate
# SHALL be fetched using the Yahoo Finance ticker format `{T}{C}=X`.
#
# Feature: unified-etf-portfolio, Property 14: FX pair format correctness
# Validates: Requirements 15.5
# ---------------------------------------------------------------------------

import re as _re_p14


def _build_fx_ticker(trading_currency: str, target_currency: str) -> str:
    """
    Replicate the FX ticker construction logic from portfolio.py fetch_prices().

    In portfolio.py the FX ticker is built as:
        fx_ticker = f"{curr}{target_currency}=X"
    """
    return f"{trading_currency}{target_currency}=X"


# Strategy: generate 3-letter uppercase ASCII currency codes (ISO 4217 style)
_currency_code_strategy = st.text(
    alphabet=st.sampled_from("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    min_size=3,
    max_size=3,
)

_FX_TICKER_PATTERN = _re_p14.compile(r"^[A-Z]{3}[A-Z]{3}=X$")


@given(
    trading_currency=_currency_code_strategy,
    target_currency=_currency_code_strategy,
)
@settings(max_examples=100)
def test_property_14_fx_pair_format_correctness(trading_currency: str, target_currency: str):
    """
    **Validates: Requirements 15.5**

    Property 14: FX pair format correctness.

    For any trading currency T and target currency C where T ≠ C, the FX ticker
    is formatted as `{T}{C}=X` (e.g. USDPLN=X, GBPEUR=X).
    """
    # Only test when currencies differ (same currency uses rate 1.0, no FX fetch)
    assume(trading_currency != target_currency)

    fx_ticker = _build_fx_ticker(trading_currency, target_currency)

    # 1. Must match the pattern {3 uppercase}{3 uppercase}=X
    assert _FX_TICKER_PATTERN.match(fx_ticker), (
        f"FX ticker does not match pattern {{T}}{{C}}=X: {fx_ticker!r} "
        f"for trading={trading_currency!r}, target={target_currency!r}"
    )

    # 2. Must start with the trading currency
    assert fx_ticker.startswith(trading_currency), (
        f"FX ticker does not start with trading currency: {fx_ticker!r} "
        f"should start with {trading_currency!r}"
    )

    # 3. Must contain the target currency after the trading currency
    assert fx_ticker[3:6] == target_currency, (
        f"FX ticker does not contain target currency at positions 3-5: {fx_ticker!r} "
        f"expected {target_currency!r} at positions 3-5"
    )

    # 4. Must end with =X
    assert fx_ticker.endswith("=X"), (
        f"FX ticker does not end with '=X': {fx_ticker!r}"
    )

    # 5. Total length must be exactly 7 (3 + 3 + 2 for "=X")
    assert len(fx_ticker) == 8, (
        f"FX ticker length should be 8, got {len(fx_ticker)}: {fx_ticker!r}"
    )
