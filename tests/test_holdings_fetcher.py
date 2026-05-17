"""
Integration tests for iShares CSV fetching with mocked HTTP responses.

Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6
"""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from etf_holdings import _fetch_ishares, _discover_ishares_product_id, fetch_holdings, _default_fetcher
from models import FetchResult


# ---------------------------------------------------------------------------
# Realistic iShares CSV data fixtures
# ---------------------------------------------------------------------------

REALISTIC_ISHARES_CSV = """\
iShares Core S&P 500 UCITS ETF
Fund Holdings as of,"Jun 20, 2024"
Inception Date,"May 19, 2010"

Ticker,Name,Sector,Asset Class,Market Value,Weight (%),Notional Value,Shares,Price,Location,Exchange,Currency
AAPL,Apple Inc,Information Technology,Equity,"1,234,567",8.24,1234567,5000,247.50,United States,NASDAQ,USD
MSFT,Microsoft Corp,Information Technology,Equity,"1,100,000",7.15,1100000,2800,392.86,United States,NASDAQ,USD
NVDA,NVIDIA Corp,Information Technology,Equity,"900,000",5.89,900000,1000,900.00,United States,NASDAQ,USD
AMZN,Amazon.com Inc,Consumer Discretionary,Equity,"800,000",4.50,800000,4500,177.78,United States,NASDAQ,USD
GOOGL,Alphabet Inc Class A,Communication,Equity,"700,000",3.80,700000,4000,175.00,United States,NASDAQ,USD
"""

ISHARES_CSV_WITH_EMPTY_NAMES = """\
iShares MSCI World UCITS ETF
Fund Holdings as of,"Jun 20, 2024"

Ticker,Name,Sector,Asset Class,Market Value,Weight (%),Notional Value,Shares,Price,Location,Exchange,Currency
AAPL,Apple Inc,Information Technology,Equity,"1,234,567",8.24,1234567,5000,247.50,United States,NASDAQ,USD
MSFT,,Information Technology,Equity,"1,100,000",7.15,1100000,2800,392.86,United States,NASDAQ,USD
NVDA,NVIDIA Corp,Information Technology,Equity,"900,000",5.89,900000,1000,900.00,United States,NASDAQ,USD
CASH,   ,Cash,Cash,"50,000",0.30,50000,50000,1.00,United States,,USD
AMZN,Amazon.com Inc,Consumer Discretionary,Equity,"800,000",4.50,800000,4500,177.78,United States,NASDAQ,USD
"""

ISHARES_CSV_WITH_NON_NUMERIC_WEIGHTS = """\
iShares ETF
Fund Holdings as of,"Jun 20, 2024"

Ticker,Name,Sector,Asset Class,Market Value,Weight (%),Notional Value,Shares,Price,Location,Exchange,Currency
AAPL,Apple Inc,Information Technology,Equity,"1,234,567",8.24,1234567,5000,247.50,United States,NASDAQ,USD
MSFT,Microsoft Corp,Information Technology,Equity,"1,100,000",N/A,1100000,2800,392.86,United States,NASDAQ,USD
NVDA,NVIDIA Corp,Information Technology,Equity,"900,000",-,900000,1000,900.00,United States,NASDAQ,USD
AMZN,Amazon.com Inc,Consumer Discretionary,Equity,"800,000",4.50,800000,4500,177.78,United States,NASDAQ,USD
"""

ISHARES_CSV_NO_HEADER = """\
iShares Core S&P 500 UCITS ETF
Fund Holdings as of,"Jun 20, 2024"
Inception Date,"May 19, 2010"

Some random data without proper headers
AAPL,Apple Inc,Information Technology,Equity,1234567,8.24
MSFT,Microsoft Corp,Information Technology,Equity,1100000,7.15
"""


# ---------------------------------------------------------------------------
# Test 1: Successful CSV download with realistic data
# ---------------------------------------------------------------------------

class TestFetchIsharesSuccess:
    """Test successful iShares CSV parsing with realistic data."""

    @patch("etf_holdings.requests.get")
    def test_parses_realistic_csv_correctly(self, mock_get):
        """Verify correct parsing of a realistic iShares CSV response."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.text = REALISTIC_ISHARES_CSV
        mock_get.return_value = mock_response

        fetch_result = _fetch_ishares("253743")
        result = fetch_result.holdings

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["ticker", "name", "weight"]
        assert len(result) == 5

    @patch("etf_holdings.requests.get")
    def test_ticker_values_parsed_correctly(self, mock_get):
        """Verify ticker column values are correct."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.text = REALISTIC_ISHARES_CSV
        mock_get.return_value = mock_response

        result = _fetch_ishares("253743").holdings

        assert result["ticker"].tolist() == ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]

    @patch("etf_holdings.requests.get")
    def test_name_values_parsed_correctly(self, mock_get):
        """Verify name column values are correct."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.text = REALISTIC_ISHARES_CSV
        mock_get.return_value = mock_response

        result = _fetch_ishares("253743").holdings

        assert result["name"].tolist() == [
            "Apple Inc",
            "Microsoft Corp",
            "NVIDIA Corp",
            "Amazon.com Inc",
            "Alphabet Inc Class A",
        ]

    @patch("etf_holdings.requests.get")
    def test_weight_values_parsed_as_percentages(self, mock_get):
        """Verify weight values are parsed as float percentages."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.text = REALISTIC_ISHARES_CSV
        mock_get.return_value = mock_response

        result = _fetch_ishares("253743").holdings

        expected_weights = [8.24, 7.15, 5.89, 4.50, 3.80]
        assert result["weight"].tolist() == pytest.approx(expected_weights)

    @patch("etf_holdings.requests.get")
    def test_header_detection_skips_metadata_lines(self, mock_get):
        """Verify that metadata lines before the header are skipped."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.text = REALISTIC_ISHARES_CSV
        mock_get.return_value = mock_response

        result = _fetch_ishares("253743").holdings

        # Should not contain metadata like "iShares Core S&P 500" as a row
        assert "iShares Core S&P 500 UCITS ETF" not in result["name"].tolist()


# ---------------------------------------------------------------------------
# Test 2: CSV with empty names — verify they're filtered out
# ---------------------------------------------------------------------------

class TestFetchIsharesEmptyNames:
    """Test that rows with empty or whitespace-only names are excluded."""

    @patch("etf_holdings.requests.get")
    def test_empty_names_filtered_out(self, mock_get):
        """Verify rows with empty name values are excluded."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.text = ISHARES_CSV_WITH_EMPTY_NAMES
        mock_get.return_value = mock_response

        result = _fetch_ishares("253743").holdings

        assert result is not None
        # MSFT has empty name "", CASH has whitespace-only name "   "
        # Only AAPL, NVDA, AMZN should remain
        assert len(result) == 3
        assert "Apple Inc" in result["name"].tolist()
        assert "NVIDIA Corp" in result["name"].tolist()
        assert "Amazon.com Inc" in result["name"].tolist()

    @patch("etf_holdings.requests.get")
    def test_whitespace_only_names_filtered_out(self, mock_get):
        """Verify rows with whitespace-only names are excluded."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.text = ISHARES_CSV_WITH_EMPTY_NAMES
        mock_get.return_value = mock_response

        result = _fetch_ishares("253743").holdings

        # No name should be empty or whitespace-only
        for name in result["name"].tolist():
            assert str(name).strip() != ""


# ---------------------------------------------------------------------------
# Test 3: CSV with non-numeric weights — verify they become 0.0
# ---------------------------------------------------------------------------

class TestFetchIsharesNonNumericWeights:
    """Test that non-numeric weight values are converted to 0.0."""

    @patch("etf_holdings.requests.get")
    def test_non_numeric_weights_become_zero(self, mock_get):
        """Verify non-numeric weight values (N/A, -) are converted to 0.0."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.text = ISHARES_CSV_WITH_NON_NUMERIC_WEIGHTS
        mock_get.return_value = mock_response

        result = _fetch_ishares("253743").holdings

        assert result is not None
        # AAPL=8.24, MSFT=0.0 (N/A), NVDA=0.0 (-), AMZN=4.50
        weights = result.set_index("ticker")["weight"]
        assert weights["AAPL"] == pytest.approx(8.24)
        assert weights["MSFT"] == pytest.approx(0.0)
        assert weights["NVDA"] == pytest.approx(0.0)
        assert weights["AMZN"] == pytest.approx(4.50)


# ---------------------------------------------------------------------------
# Test 4: CSV download failure (HTTP error) — verify returns None
# ---------------------------------------------------------------------------

class TestFetchIsharesHttpError:
    """Test that HTTP errors result in None being returned."""

    @patch("etf_holdings.requests.get")
    def test_http_error_returns_none(self, mock_get):
        """Verify that an HTTP error causes _fetch_ishares to return None."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("HTTP 500 Server Error")
        mock_get.return_value = mock_response

        result = _fetch_ishares("253743")

        assert result.holdings is None

    @patch("etf_holdings.requests.get")
    def test_connection_error_returns_none(self, mock_get):
        """Verify that a connection error causes _fetch_ishares to return None."""
        mock_get.side_effect = Exception("Connection refused")

        result = _fetch_ishares("253743")

        assert result.holdings is None

    @patch("etf_holdings.requests.get")
    def test_timeout_returns_none(self, mock_get):
        """Verify that a timeout causes _fetch_ishares to return None."""
        mock_get.side_effect = TimeoutError("Request timed out")

        result = _fetch_ishares("253743")

        assert result.holdings is None


# ---------------------------------------------------------------------------
# Test 5: CSV without proper header row — verify returns None
# ---------------------------------------------------------------------------

class TestFetchIsharesNoHeader:
    """Test that CSV without a proper header row returns None."""

    @patch("etf_holdings.requests.get")
    def test_missing_header_returns_none(self, mock_get):
        """Verify that a CSV without Ticker/Name header returns None."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.text = ISHARES_CSV_NO_HEADER
        mock_get.return_value = mock_response

        result = _fetch_ishares("253743")

        assert result.holdings is None

    @patch("etf_holdings.requests.get")
    def test_empty_response_returns_none(self, mock_get):
        """Verify that an empty response body returns None."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.text = ""
        mock_get.return_value = mock_response

        result = _fetch_ishares("253743")

        assert result.holdings is None


# ---------------------------------------------------------------------------
# Test 6: Fallback chain when product ID discovery fails
# ---------------------------------------------------------------------------

class TestFallbackChain:
    """Test the fallback chain: iShares → Vanguard → justETF."""

    def setup_method(self):
        """Clear the module-level singleton state between tests."""
        _default_fetcher.clear()

    @patch("etf_holdings._save_cache")
    @patch("etf_holdings._load_cache")
    @patch("etf_holdings._fetch_justetf")
    @patch("etf_holdings._fetch_vanguard")
    @patch("etf_holdings._discover_ishares_product_id")
    @patch("etf_holdings._discover_isin")
    def test_fallback_to_vanguard_when_product_id_not_found(
        self, mock_isin, mock_product_id, mock_vanguard, mock_justetf, mock_cache, mock_save
    ):
        """When iShares product ID discovery fails, try Vanguard API."""
        mock_cache.return_value = {}
        mock_isin.return_value = "IE00B5BMR087"
        mock_product_id.return_value = None  # Product ID discovery fails

        vanguard_df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT"],
            "name": ["Apple Inc", "Microsoft Corp"],
            "weight": [8.0, 7.0],
        })
        mock_vanguard.return_value = FetchResult(holdings=vanguard_df, used_justetf=False)

        result = fetch_holdings("CSPX.L", fund_family="BlackRock")

        assert result is not None
        mock_vanguard.assert_called_once_with("IE00B5BMR087")
        # justETF should NOT be called since Vanguard succeeded
        mock_justetf.assert_not_called()

    @patch("etf_holdings._save_cache")
    @patch("etf_holdings._load_cache")
    @patch("etf_holdings._fetch_justetf")
    @patch("etf_holdings._fetch_vanguard")
    @patch("etf_holdings._discover_ishares_product_id")
    @patch("etf_holdings._discover_isin")
    def test_fallback_to_justetf_when_vanguard_also_fails(
        self, mock_isin, mock_product_id, mock_vanguard, mock_justetf, mock_cache, mock_save
    ):
        """When both iShares and Vanguard fail, fall back to justETF."""
        mock_cache.return_value = {}
        mock_isin.return_value = "IE00B5BMR087"
        mock_product_id.return_value = None  # Product ID discovery fails
        mock_vanguard.return_value = FetchResult(holdings=None, used_justetf=False)

        justetf_df = pd.DataFrame({
            "ticker": ["N/A", "N/A"],
            "name": ["Apple Inc", "Microsoft Corp"],
            "weight": [8.0, 7.0],
        })
        mock_justetf.return_value = FetchResult(holdings=justetf_df, used_justetf=True)

        result = fetch_holdings("CSPX.L", fund_family="BlackRock")

        assert result is not None
        mock_vanguard.assert_called_once_with("IE00B5BMR087")
        mock_justetf.assert_called_once_with("IE00B5BMR087")

    @patch("etf_holdings._save_cache")
    @patch("etf_holdings._load_cache")
    @patch("etf_holdings._fetch_justetf")
    @patch("etf_holdings._fetch_vanguard")
    @patch("etf_holdings._discover_ishares_product_id")
    @patch("etf_holdings._discover_isin")
    def test_returns_none_when_all_sources_fail(
        self, mock_isin, mock_product_id, mock_vanguard, mock_justetf, mock_cache, mock_save
    ):
        """When all sources fail, return None."""
        mock_cache.return_value = {}
        mock_isin.return_value = "IE00B5BMR087"
        mock_product_id.return_value = None
        mock_vanguard.return_value = FetchResult(holdings=None, used_justetf=False)
        mock_justetf.return_value = FetchResult(holdings=None, used_justetf=False)

        result = fetch_holdings("CSPX.L", fund_family="BlackRock")

        assert result is None

    @patch("etf_holdings._save_cache")
    @patch("etf_holdings._load_cache")
    @patch("etf_holdings._fetch_ishares")
    @patch("etf_holdings._discover_ishares_product_id")
    @patch("etf_holdings._discover_isin")
    def test_uses_ishares_when_product_id_found(
        self, mock_isin, mock_product_id, mock_ishares, mock_cache, mock_save
    ):
        """When product ID is discovered, use iShares CSV directly."""
        mock_cache.return_value = {}
        mock_isin.return_value = "IE00B5BMR087"
        mock_product_id.return_value = "253743"

        ishares_df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT"],
            "name": ["Apple Inc", "Microsoft Corp"],
            "weight": [8.24, 7.15],
        })
        mock_ishares.return_value = FetchResult(holdings=ishares_df, used_justetf=False)

        result = fetch_holdings("CSPX.L", fund_family="BlackRock")

        assert result is not None
        mock_ishares.assert_called_once_with("253743")

    @patch("etf_holdings._save_cache")
    @patch("etf_holdings._load_cache")
    @patch("etf_holdings._discover_isin")
    def test_returns_none_when_isin_discovery_fails(
        self, mock_isin, mock_cache, mock_save
    ):
        """When ISIN cannot be discovered, return None immediately."""
        mock_cache.return_value = {}
        mock_isin.return_value = None

        result = fetch_holdings("UNKNOWN.L", fund_family="BlackRock")

        assert result is None

    @patch("etf_holdings._save_cache")
    @patch("etf_holdings._load_cache")
    @patch("etf_holdings._fetch_vanguard")
    @patch("etf_holdings._discover_ishares_product_id")
    @patch("etf_holdings._discover_isin")
    def test_caches_provider_on_vanguard_fallback(
        self, mock_isin, mock_product_id, mock_vanguard, mock_cache, mock_save
    ):
        """When Vanguard fallback succeeds, cache the provider as vanguard."""
        mock_cache.return_value = {}
        mock_isin.return_value = "IE00B5BMR087"
        mock_product_id.return_value = None

        vanguard_df = pd.DataFrame({
            "ticker": ["AAPL"],
            "name": ["Apple Inc"],
            "weight": [8.0],
        })
        mock_vanguard.return_value = FetchResult(holdings=vanguard_df, used_justetf=False)

        fetch_holdings("CSPX.L", fund_family="BlackRock")

        # Verify cache was saved with vanguard provider
        mock_save.assert_called()
        saved_cache = mock_save.call_args[0][0]
        assert saved_cache["CSPX.L"]["provider"] == "vanguard"
