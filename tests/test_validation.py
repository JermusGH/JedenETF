"""
Unit tests for the validation module.

Tests cover:
- validate_ticker: dot separator check, whitespace trimming, uppercase conversion
- is_valid_isin: 12-character ISIN format validation
- validate_holdings_csv: CSV column validation, weight/name checks, zero-weight filtering
"""

import io

import pandas as pd
import pytest

from validation import is_valid_isin, validate_holdings_csv, validate_ticker


# ---------------------------------------------------------------------------
# validate_ticker tests
# ---------------------------------------------------------------------------


class TestValidateTicker:
    """Tests for validate_ticker function."""

    def test_valid_ticker_with_dot(self):
        cleaned, error = validate_ticker("CSPX.L")
        assert cleaned == "CSPX.L"
        assert error == ""

    def test_valid_ticker_lowercase_converted_to_uppercase(self):
        cleaned, error = validate_ticker("cspx.l")
        assert cleaned == "CSPX.L"
        assert error == ""

    def test_valid_ticker_with_whitespace_trimmed(self):
        cleaned, error = validate_ticker("  VWCE.DE  ")
        assert cleaned == "VWCE.DE"
        assert error == ""

    def test_valid_ticker_mixed_case_and_whitespace(self):
        cleaned, error = validate_ticker("  vwce.de  ")
        assert cleaned == "VWCE.DE"
        assert error == ""

    def test_ticker_without_dot_rejected(self):
        cleaned, error = validate_ticker("CSPX")
        assert cleaned is None
        assert "exchange suffix" in error.lower() or "dot" in error.lower() or "." in error

    def test_empty_ticker_rejected(self):
        cleaned, error = validate_ticker("")
        assert cleaned is None
        assert error != ""

    def test_whitespace_only_ticker_rejected(self):
        cleaned, error = validate_ticker("   ")
        assert cleaned is None
        assert error != ""

    def test_ticker_with_multiple_dots(self):
        cleaned, error = validate_ticker("BRK.B.L")
        assert cleaned == "BRK.B.L"
        assert error == ""


# ---------------------------------------------------------------------------
# is_valid_isin tests
# ---------------------------------------------------------------------------


class TestIsValidIsin:
    """Tests for is_valid_isin function."""

    def test_valid_isin_ie(self):
        assert is_valid_isin("IE00B5BMR087") is True

    def test_valid_isin_lu(self):
        assert is_valid_isin("LU1681043599") is True

    def test_valid_isin_us(self):
        assert is_valid_isin("US0378331005") is True

    def test_too_short(self):
        assert is_valid_isin("IE00B5BMR08") is False

    def test_too_long(self):
        assert is_valid_isin("IE00B5BMR0877") is False

    def test_lowercase_letters_rejected(self):
        assert is_valid_isin("ie00B5BMR087") is False

    def test_first_two_not_letters(self):
        assert is_valid_isin("12345678901A") is False

    def test_empty_string(self):
        assert is_valid_isin("") is False

    def test_non_string_input(self):
        assert is_valid_isin(123456789012) is False  # type: ignore

    def test_special_characters_rejected(self):
        assert is_valid_isin("IE00B5BMR0-7") is False

    def test_all_uppercase_letters_valid(self):
        assert is_valid_isin("ABCDEFGHIJKL") is True

    def test_all_digits_after_prefix(self):
        assert is_valid_isin("IE0000000000") is True


# ---------------------------------------------------------------------------
# validate_holdings_csv tests
# ---------------------------------------------------------------------------


class TestValidateHoldingsCsv:
    """Tests for validate_holdings_csv function."""

    def _make_csv(self, content: str) -> io.StringIO:
        return io.StringIO(content)

    def test_valid_csv(self):
        csv = self._make_csv("ticker,name,weight\nAAPL,Apple Inc,8.5\nMSFT,Microsoft,7.2\n")
        df, error = validate_holdings_csv(csv)
        assert error == ""
        assert df is not None
        assert len(df) == 2
        assert list(df.columns) == ["ticker", "name", "weight"]

    def test_missing_column(self):
        csv = self._make_csv("ticker,weight\nAAPL,8.5\n")
        df, error = validate_holdings_csv(csv)
        assert df is None
        assert "name" in error.lower()

    def test_case_insensitive_columns(self):
        csv = self._make_csv("Ticker,Name,Weight\nAAPL,Apple,8.5\n")
        df, error = validate_holdings_csv(csv)
        assert error == ""
        assert df is not None

    def test_whitespace_trimmed_columns(self):
        csv = self._make_csv(" ticker , name , weight \nAAPL,Apple,8.5\n")
        df, error = validate_holdings_csv(csv)
        assert error == ""
        assert df is not None

    def test_non_numeric_weight_error(self):
        csv = self._make_csv("ticker,name,weight\nAAPL,Apple,abc\nMSFT,Microsoft,7.2\n")
        df, error = validate_holdings_csv(csv)
        assert df is None
        assert "non-numeric" in error.lower() or "row" in error.lower()

    def test_negative_weight_error(self):
        csv = self._make_csv("ticker,name,weight\nAAPL,Apple,-5.0\n")
        df, error = validate_holdings_csv(csv)
        assert df is None
        assert "negative" in error.lower()

    def test_empty_name_error(self):
        csv = self._make_csv("ticker,name,weight\nAAPL,,8.5\n")
        df, error = validate_holdings_csv(csv)
        assert df is None
        assert "empty" in error.lower() or "name" in error.lower()

    def test_zero_weight_filtered_out(self):
        csv = self._make_csv("ticker,name,weight\nAAPL,Apple,8.5\nMSFT,Microsoft,0.0\n")
        df, error = validate_holdings_csv(csv)
        assert error == ""
        assert df is not None
        assert len(df) == 1
        assert df.iloc[0]["ticker"] == "AAPL"

    def test_all_zero_weights_error(self):
        csv = self._make_csv("ticker,name,weight\nAAPL,Apple,0.0\nMSFT,Microsoft,0.0\n")
        df, error = validate_holdings_csv(csv)
        assert df is None
        assert "weight > 0" in error.lower() or "no holdings" in error.lower()

    def test_invalid_csv_format(self):
        csv = self._make_csv("this is not a csv\nwith proper structure")
        df, error = validate_holdings_csv(csv)
        # Should either fail to parse or fail column validation
        assert df is None
        assert error != ""
