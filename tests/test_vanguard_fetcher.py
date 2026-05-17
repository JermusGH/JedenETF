"""
Integration tests for Vanguard GraphQL holdings fetching.

Tests verify:
- Successful GraphQL response parsing with realistic data
- Holdings missing ticker get "N/A" substitution
- Holdings with empty/blank names are filtered out
- Error responses from API return None
- Unexpected response structure (missing fields) returns None
- Empty holdings list returns None

Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5
"""

from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from etf_holdings import _fetch_vanguard


# ---------------------------------------------------------------------------
# Realistic Vanguard GraphQL response fixtures
# ---------------------------------------------------------------------------

def _make_graphql_response(items, include_errors=False):
    """Build a realistic Vanguard GraphQL JSON response."""
    response = {
        "data": {
            "borHoldings": [
                {
                    "holdings": {
                        "totalHoldings": len(items),
                        "items": items,
                    }
                }
            ]
        }
    }
    if include_errors:
        response["errors"] = [{"message": "Something went wrong"}]
    return response


REALISTIC_HOLDINGS_ITEMS = [
    {"ticker": "AAPL", "issuerName": "Apple Inc", "marketValuePercentage": 6.52},
    {"ticker": "MSFT", "issuerName": "Microsoft Corp", "marketValuePercentage": 5.89},
    {"ticker": "AMZN", "issuerName": "Amazon.com Inc", "marketValuePercentage": 3.21},
    {"ticker": "NVDA", "issuerName": "NVIDIA Corp", "marketValuePercentage": 2.98},
    {"ticker": "GOOGL", "issuerName": "Alphabet Inc Class A", "marketValuePercentage": 2.15},
    {"ticker": "META", "issuerName": "Meta Platforms Inc", "marketValuePercentage": 1.87},
    {"ticker": "TSLA", "issuerName": "Tesla Inc", "marketValuePercentage": 1.54},
    {"ticker": "BRK.B", "issuerName": "Berkshire Hathaway Inc", "marketValuePercentage": 1.32},
    {"ticker": "JPM", "issuerName": "JPMorgan Chase & Co", "marketValuePercentage": 1.21},
    {"ticker": "V", "issuerName": "Visa Inc", "marketValuePercentage": 1.05},
]


# ---------------------------------------------------------------------------
# Test: Successful GraphQL response with realistic data
# ---------------------------------------------------------------------------


class TestVanguardSuccessfulParsing:
    """Test successful parsing of Vanguard GraphQL responses (Req 5.1, 5.2, 5.3)."""

    def test_parses_realistic_response_correctly(self):
        """Verify correct parsing of a realistic Vanguard GraphQL response."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = _make_graphql_response(REALISTIC_HOLDINGS_ITEMS)

        with patch("etf_holdings.requests.post", return_value=mock_response) as mock_post:
            fetch_result = _fetch_vanguard("IE00BK5BQT80")

        result = fetch_result.holdings
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["ticker", "name", "weight"]
        assert len(result) == 10

        # Verify first row data
        assert result.iloc[0]["ticker"] == "AAPL"
        assert result.iloc[0]["name"] == "Apple Inc"
        assert result.iloc[0]["weight"] == 6.52

        # Verify last row data
        assert result.iloc[9]["ticker"] == "V"
        assert result.iloc[9]["name"] == "Visa Inc"
        assert result.iloc[9]["weight"] == 1.05

    def test_request_uses_correct_endpoint_and_headers(self):
        """Verify the request is made to the correct Vanguard GraphQL endpoint."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = _make_graphql_response(REALISTIC_HOLDINGS_ITEMS)

        with patch("etf_holdings.requests.post", return_value=mock_response) as mock_post:
            _fetch_vanguard("IE00BK5BQT80")

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        # Verify endpoint
        assert call_kwargs[0][0] == "https://www.nl.vanguard/gpx/graphql"
        # Verify timeout is 30 seconds (Req 5.1)
        assert call_kwargs[1]["timeout"] == 30
        # Verify query requests 1500 items (Req 5.2)
        query = call_kwargs[1]["json"]["query"]
        assert "1500" in query
        # Verify ISIN is in the query
        assert "IE00BK5BQT80" in query

    def test_all_weights_are_floats(self):
        """Verify all weight values are numeric floats."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = _make_graphql_response(REALISTIC_HOLDINGS_ITEMS)

        with patch("etf_holdings.requests.post", return_value=mock_response):
            fetch_result = _fetch_vanguard("IE00BK5BQT80")

        result = fetch_result.holdings
        assert result is not None
        assert result["weight"].dtype == float


# ---------------------------------------------------------------------------
# Test: Holdings missing ticker get "N/A" substitution
# ---------------------------------------------------------------------------


class TestVanguardMissingTicker:
    """Test N/A substitution for holdings without ticker (Req 5.3)."""

    def test_missing_ticker_gets_na(self):
        """Holdings with no ticker value should get 'N/A' substituted."""
        items = [
            {"ticker": "AAPL", "issuerName": "Apple Inc", "marketValuePercentage": 6.52},
            {"ticker": None, "issuerName": "Some Bond Fund", "marketValuePercentage": 2.10},
            {"issuerName": "Government Bond", "marketValuePercentage": 1.50},
        ]
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = _make_graphql_response(items)

        with patch("etf_holdings.requests.post", return_value=mock_response):
            result = _fetch_vanguard("IE00BK5BQT80").holdings

        assert result is not None
        assert len(result) == 3
        assert result.iloc[0]["ticker"] == "AAPL"
        assert result.iloc[1]["ticker"] == "N/A"
        assert result.iloc[2]["ticker"] == "N/A"

    def test_empty_string_ticker_gets_na(self):
        """Holdings with empty string ticker should get 'N/A' substituted."""
        items = [
            {"ticker": "", "issuerName": "Cash Reserve", "marketValuePercentage": 0.50},
            {"ticker": "MSFT", "issuerName": "Microsoft Corp", "marketValuePercentage": 5.89},
        ]
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = _make_graphql_response(items)

        with patch("etf_holdings.requests.post", return_value=mock_response):
            result = _fetch_vanguard("IE00BK5BQT80").holdings

        assert result is not None
        assert result.iloc[0]["ticker"] == "N/A"
        assert result.iloc[1]["ticker"] == "MSFT"


# ---------------------------------------------------------------------------
# Test: Holdings with empty/blank names are filtered out
# ---------------------------------------------------------------------------


class TestVanguardEmptyNames:
    """Test filtering of holdings with empty or blank names (Req 5.4)."""

    def test_empty_name_is_excluded(self):
        """Holdings with empty string name should be filtered out."""
        items = [
            {"ticker": "AAPL", "issuerName": "Apple Inc", "marketValuePercentage": 6.52},
            {"ticker": "CASH", "issuerName": "", "marketValuePercentage": 0.30},
            {"ticker": "MSFT", "issuerName": "Microsoft Corp", "marketValuePercentage": 5.89},
        ]
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = _make_graphql_response(items)

        with patch("etf_holdings.requests.post", return_value=mock_response):
            result = _fetch_vanguard("IE00BK5BQT80").holdings

        assert result is not None
        assert len(result) == 2
        assert "Apple Inc" in result["name"].values
        assert "Microsoft Corp" in result["name"].values
        assert "" not in result["name"].values

    def test_whitespace_only_name_is_excluded(self):
        """Holdings with whitespace-only name should be filtered out."""
        items = [
            {"ticker": "AAPL", "issuerName": "Apple Inc", "marketValuePercentage": 6.52},
            {"ticker": "X", "issuerName": "   ", "marketValuePercentage": 0.10},
            {"ticker": "Y", "issuerName": "  \t  ", "marketValuePercentage": 0.05},
        ]
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = _make_graphql_response(items)

        with patch("etf_holdings.requests.post", return_value=mock_response):
            result = _fetch_vanguard("IE00BK5BQT80").holdings

        assert result is not None
        assert len(result) == 1
        assert result.iloc[0]["name"] == "Apple Inc"

    def test_all_empty_names_returns_none(self):
        """If all holdings have empty names, result after filtering is None."""
        items = [
            {"ticker": "X", "issuerName": "", "marketValuePercentage": 1.0},
            {"ticker": "Y", "issuerName": "   ", "marketValuePercentage": 2.0},
        ]
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = _make_graphql_response(items)

        with patch("etf_holdings.requests.post", return_value=mock_response):
            result = _fetch_vanguard("IE00BK5BQT80")

        # After filtering all empty names, the DataFrame is empty → holdings is None
        assert result.holdings is None


# ---------------------------------------------------------------------------
# Test: Error response from API returns None
# ---------------------------------------------------------------------------


class TestVanguardErrorResponse:
    """Test error handling for Vanguard API errors (Req 5.5)."""

    def test_graphql_error_response_returns_none(self):
        """API returning errors field should result in None."""
        error_response = {
            "errors": [{"message": "ISIN not found", "extensions": {"code": "NOT_FOUND"}}]
        }
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = error_response

        with patch("etf_holdings.requests.post", return_value=mock_response):
            result = _fetch_vanguard("IE00INVALID00")

        assert result.holdings is None

    def test_http_error_returns_none(self):
        """HTTP error (e.g. 500) should result in None."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("500 Server Error")

        with patch("etf_holdings.requests.post", return_value=mock_response):
            result = _fetch_vanguard("IE00BK5BQT80")

        assert result.holdings is None

    def test_connection_timeout_returns_none(self):
        """Connection timeout should result in None."""
        with patch("etf_holdings.requests.post", side_effect=Exception("Connection timed out")):
            result = _fetch_vanguard("IE00BK5BQT80")

        assert result.holdings is None

    def test_network_error_returns_none(self):
        """Network error should result in None."""
        with patch("etf_holdings.requests.post", side_effect=ConnectionError("Network unreachable")):
            result = _fetch_vanguard("IE00BK5BQT80")

        assert result.holdings is None


# ---------------------------------------------------------------------------
# Test: Unexpected response structure returns None
# ---------------------------------------------------------------------------


class TestVanguardUnexpectedStructure:
    """Test handling of unexpected response structures (Req 5.5)."""

    def test_missing_data_key_returns_none(self):
        """Response without 'data' key should return None."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"something": "else"}

        with patch("etf_holdings.requests.post", return_value=mock_response):
            result = _fetch_vanguard("IE00BK5BQT80")

        assert result.holdings is None

    def test_missing_borholdings_key_returns_none(self):
        """Response without 'borHoldings' in data should return None."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"data": {"otherField": []}}

        with patch("etf_holdings.requests.post", return_value=mock_response):
            result = _fetch_vanguard("IE00BK5BQT80")

        assert result.holdings is None

    def test_empty_borholdings_array_returns_none(self):
        """Empty borHoldings array should return None."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"data": {"borHoldings": []}}

        with patch("etf_holdings.requests.post", return_value=mock_response):
            result = _fetch_vanguard("IE00BK5BQT80")

        assert result.holdings is None

    def test_missing_holdings_key_returns_none(self):
        """borHoldings entry without 'holdings' key should return None."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"data": {"borHoldings": [{"other": "data"}]}}

        with patch("etf_holdings.requests.post", return_value=mock_response):
            result = _fetch_vanguard("IE00BK5BQT80")

        assert result.holdings is None

    def test_null_data_returns_none(self):
        """Null data field should return None."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"data": None}

        with patch("etf_holdings.requests.post", return_value=mock_response):
            result = _fetch_vanguard("IE00BK5BQT80")

        assert result.holdings is None


# ---------------------------------------------------------------------------
# Test: Empty holdings list returns None
# ---------------------------------------------------------------------------


class TestVanguardEmptyHoldings:
    """Test handling of empty holdings list (Req 5.5)."""

    def test_empty_items_list_returns_none(self):
        """Empty items list should return None."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = _make_graphql_response([])

        with patch("etf_holdings.requests.post", return_value=mock_response):
            result = _fetch_vanguard("IE00BK5BQT80")

        assert result.holdings is None

    def test_null_items_returns_none(self):
        """Null items field should return None."""
        response = {
            "data": {
                "borHoldings": [
                    {"holdings": {"totalHoldings": 0, "items": None}}
                ]
            }
        }
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = response

        with patch("etf_holdings.requests.post", return_value=mock_response):
            result = _fetch_vanguard("IE00BK5BQT80")

        assert result.holdings is None
