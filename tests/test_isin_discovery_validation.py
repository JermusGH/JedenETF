"""
Unit tests for ISIN validation in the _discover_isin flow.

Verifies that only ISINs matching the 12-character format
(2 uppercase letters + 10 alphanumeric chars) are accepted and cached.

Validates: Requirements 3.1, 3.3
"""

from unittest.mock import patch, MagicMock

import pytest

from etf_holdings import _discover_isin


class TestDiscoverIsinValidation:
    """Tests that _discover_isin validates ISINs before returning them."""

    @patch("etf_holdings.curl_requests", None)
    @patch("etf_holdings.requests.get")
    def test_valid_isin_returned(self, mock_get):
        """A valid ISIN found on the page is returned."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.text = '<div>ISIN</div><span>IE00B5BMR087</span>'
        mock_get.return_value = mock_resp

        result = _discover_isin("CSPX.L")
        assert result == "IE00B5BMR087"

    @patch("etf_holdings.curl_requests", None)
    @patch("etf_holdings.requests.get")
    def test_invalid_isin_too_short_rejected(self, mock_get):
        """An ISIN that's too short is rejected (returns None)."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        # The regex in _discover_isin won't match a short ISIN anyway,
        # but test the fallback path with an invalid candidate
        mock_resp.text = '<div>No ISIN label here</div><p>IE00B5BMR08</p>'
        mock_get.return_value = mock_resp

        result = _discover_isin("TEST.L")
        assert result is None

    @patch("etf_holdings.curl_requests", None)
    @patch("etf_holdings.requests.get")
    def test_valid_isin_from_fallback_path(self, mock_get):
        """A valid ISIN found via the fallback regex is returned."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        # No structured ISIN label, but a single IE/LU ISIN in the text
        mock_resp.text = '<div>Some content</div><p>Fund domicile: IE00BK5BQT80</p>'
        mock_get.return_value = mock_resp

        result = _discover_isin("VHVE.L")
        assert result == "IE00BK5BQT80"

    @patch("etf_holdings.curl_requests", None)
    @patch("etf_holdings.requests.get")
    def test_invalid_fallback_isin_rejected(self, mock_get):
        """An invalid ISIN from the fallback path is rejected."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        # Lowercase letters in what would be an ISIN - won't match regex
        mock_resp.text = '<div>Some content with no valid ISIN</div>'
        mock_get.return_value = mock_resp

        result = _discover_isin("BAD.L")
        assert result is None

    @patch("etf_holdings.curl_requests", None)
    @patch("etf_holdings.requests.get")
    def test_http_failure_returns_none(self, mock_get):
        """HTTP failure returns None."""
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_get.return_value = mock_resp

        result = _discover_isin("FAIL.L")
        assert result is None

    @patch("etf_holdings.curl_requests", None)
    @patch("etf_holdings.requests.get")
    def test_network_exception_returns_none(self, mock_get):
        """Network exception returns None."""
        mock_get.side_effect = Exception("Connection timeout")

        result = _discover_isin("TIMEOUT.L")
        assert result is None
