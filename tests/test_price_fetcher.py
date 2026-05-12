"""
Integration tests for price fetching and FX conversion.

Tests the fetch_prices function from portfolio.py with mocked Yahoo Finance responses.

Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7
"""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from portfolio import fetch_prices


def _make_price_download(data: dict[str, list[float]], dates=None):
    """
    Helper to create a mock return value for yf.download(...).

    The code does: yf.download(tickers, ...)["Close"]
    - For single ticker: ["Close"] returns a Series
    - For multiple tickers: ["Close"] returns a DataFrame with ticker columns

    This helper returns a dict-like object where ["Close"] gives the right result.
    """
    if dates is None:
        dates = pd.date_range("2024-01-15", periods=len(next(iter(data.values()))), freq="B")

    if len(data) == 1:
        # Single ticker: return DataFrame where ["Close"] gives a Series
        ticker = list(data.keys())[0]
        series = pd.Series(data[ticker], index=dates, name=ticker)
        return pd.DataFrame({"Close": series})
    else:
        # Multiple tickers: return DataFrame with MultiIndex columns
        # where ["Close"] gives a DataFrame with ticker columns
        close_df = pd.DataFrame(data, index=dates)
        arrays = [["Close"] * len(data), list(data.keys())]
        tuples = list(zip(*arrays))
        multi_idx = pd.MultiIndex.from_tuples(tuples, names=["Price", "Ticker"])
        result = pd.DataFrame(close_df.values, index=dates, columns=multi_idx)
        return result


def _make_fx_download(rates: list[float], dates=None):
    """
    Helper to create a mock return value for yf.download(fx_ticker, ...).

    The code does: yf.download(fx_ticker, ...)["Close"]
    Returns a DataFrame where ["Close"] gives a Series.
    """
    if dates is None:
        dates = pd.date_range("2024-01-15", periods=len(rates), freq="B")
    series = pd.Series(rates, index=dates)
    return pd.DataFrame({"Close": series})


class TestPriceFetchWithFXConversion:
    """Test successful price fetch with FX conversion (USD → PLN)."""

    @patch("portfolio.yf.Ticker")
    @patch("portfolio.yf.download")
    def test_usd_ticker_converted_to_pln(self, mock_download, mock_ticker):
        """
        Validates: Requirements 8.1, 8.2, 8.4, 8.5

        A ticker trading in USD should have its price multiplied by the USD/PLN FX rate.
        """
        mock_download.side_effect = [
            _make_price_download({"CSPX.L": [500.0, 510.0, 520.0]}),  # price
            _make_fx_download([4.05, 4.06, 4.07]),                     # USDPLN=X
        ]

        # Setup: ticker info returns USD currency
        ticker_instance = MagicMock()
        ticker_instance.info = {"currency": "USD"}
        mock_ticker.return_value = ticker_instance

        result = fetch_prices(["CSPX.L"], target_currency="PLN")

        # Price = 520.0 (last close) * 4.07 (last FX rate) = 2116.40
        assert result["CSPX.L"] == round(520.0 * 4.07, 2)


class TestTickerAlreadyInPLN:
    """Test ticker already trading in PLN — no FX conversion needed."""

    @patch("portfolio.yf.Ticker")
    @patch("portfolio.yf.download")
    def test_pln_ticker_no_conversion(self, mock_download, mock_ticker):
        """
        Validates: Requirements 8.2, 8.5

        A ticker already in PLN should not have FX conversion applied (rate = 1.0).
        """
        mock_download.side_effect = [
            _make_price_download({"CDR.WA": [300.0, 305.0, 310.0]}),  # price
        ]

        ticker_instance = MagicMock()
        ticker_instance.info = {"currency": "PLN"}
        mock_ticker.return_value = ticker_instance

        result = fetch_prices(["CDR.WA"], target_currency="PLN")

        # Price = 310.0 * 1.0 = 310.0 (no conversion)
        assert result["CDR.WA"] == 310.0


class TestCurrencyCannotBeDetermined:
    """Test currency can't be determined — assumes PLN (rate 1.0)."""

    @patch("portfolio.yf.Ticker")
    @patch("portfolio.yf.download")
    def test_missing_currency_assumes_pln(self, mock_download, mock_ticker):
        """
        Validates: Requirements 8.3

        When currency field is missing from ticker info, assume PLN.
        """
        mock_download.side_effect = [
            _make_price_download({"UNKNOWN.L": [100.0, 105.0, 110.0]}),  # price
        ]

        # No "currency" key in info — should default to target_currency (PLN)
        ticker_instance = MagicMock()
        ticker_instance.info = {}
        mock_ticker.return_value = ticker_instance

        result = fetch_prices(["UNKNOWN.L"], target_currency="PLN")

        # Price = 110.0 * 1.0 = 110.0 (assumed PLN, no conversion)
        assert result["UNKNOWN.L"] == 110.0


class TestFXRateFetchFails:
    """Test FX rate fetch fails — uses rate 1.0."""

    @patch("portfolio.yf.Ticker")
    @patch("portfolio.yf.download")
    def test_fx_rate_failure_uses_one(self, mock_download, mock_ticker):
        """
        Validates: Requirements 8.6

        When FX rate download returns empty data, fallback to rate 1.0.
        """
        # FX download returns empty — no rates available
        empty_fx = pd.DataFrame({"Close": pd.Series([], dtype=float)})

        mock_download.side_effect = [
            _make_price_download({"VHVE.L": [80.0, 82.0, 85.0]}),  # price
            empty_fx,                                                 # FX (empty)
        ]

        ticker_instance = MagicMock()
        ticker_instance.info = {"currency": "GBP"}
        mock_ticker.return_value = ticker_instance

        result = fetch_prices(["VHVE.L"], target_currency="PLN")

        # Price = 85.0 * 1.0 = 85.0 (FX fallback)
        assert result["VHVE.L"] == 85.0


class TestTickerPriceCannotBeRetrieved:
    """Test ticker price can't be retrieved — returns None."""

    @patch("portfolio.yf.Ticker")
    @patch("portfolio.yf.download")
    def test_missing_price_returns_none(self, mock_download, mock_ticker):
        """
        Validates: Requirements 8.7

        When a ticker's price cannot be converted to float (e.g., KeyError because
        the ticker is missing from the download data), it should return None.
        """
        # Download returns data for GOOD.L only — FAKE.L is not in the columns.
        # When the code tries last_prices["FAKE.L"], it raises KeyError → None.
        dates = pd.date_range("2024-01-15", periods=3, freq="B")
        # Simulate: yf.download(["GOOD.L", "FAKE.L"])["Close"] returns a DataFrame
        # but FAKE.L column is missing (Yahoo couldn't find it)
        close_df = pd.DataFrame(
            {"GOOD.L": [100.0, 105.0, 110.0]},
            index=dates,
        )
        # Build MultiIndex structure for the download mock
        arrays = [["Close"], ["GOOD.L"]]
        tuples = list(zip(*arrays))
        multi_idx = pd.MultiIndex.from_tuples(tuples, names=["Price", "Ticker"])
        download_result = pd.DataFrame(close_df.values, index=dates, columns=multi_idx)

        mock_download.side_effect = [
            download_result,                            # price download
            _make_fx_download([4.05, 4.06, 4.07]),     # USDPLN=X
        ]

        ticker_instance = MagicMock()
        ticker_instance.info = {"currency": "USD"}
        mock_ticker.return_value = ticker_instance

        result = fetch_prices(["GOOD.L", "FAKE.L"], target_currency="PLN")

        # GOOD.L should have a valid converted price
        assert result["GOOD.L"] == round(110.0 * 4.07, 2)
        # FAKE.L is missing from download data — should return None
        assert result["FAKE.L"] is None


class TestMultipleTickersDifferentCurrencies:
    """Test multiple tickers with different currencies — each converted correctly."""

    @patch("portfolio.yf.Ticker")
    @patch("portfolio.yf.download")
    def test_multiple_tickers_correct_conversion(self, mock_download, mock_ticker):
        """
        Validates: Requirements 8.1, 8.2, 8.4, 8.5

        Multiple tickers in different currencies should each be converted
        using their respective FX rates.
        """
        tickers = ["CSPX.L", "SEC0.DE"]

        mock_download.side_effect = [
            _make_price_download({
                "CSPX.L": [500.0, 510.0, 520.0],
                "SEC0.DE": [30.0, 31.0, 32.0],
            }),
            _make_fx_download([4.05, 4.06, 4.07]),  # USDPLN=X
            _make_fx_download([4.30, 4.31, 4.32]),  # EURPLN=X
        ]

        # Ticker info: CSPX.L in USD, SEC0.DE in EUR
        def ticker_side_effect(ticker):
            instance = MagicMock()
            if ticker == "CSPX.L":
                instance.info = {"currency": "USD"}
            elif ticker == "SEC0.DE":
                instance.info = {"currency": "EUR"}
            return instance

        mock_ticker.side_effect = ticker_side_effect

        result = fetch_prices(tickers, target_currency="PLN")

        # CSPX.L: 520.0 * 4.07 = 2116.40
        assert result["CSPX.L"] == round(520.0 * 4.07, 2)
        # SEC0.DE: 32.0 * 4.32 = 138.24
        assert result["SEC0.DE"] == round(32.0 * 4.32, 2)
