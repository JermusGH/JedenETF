"""
Integration tests for justETF scraping via _fetch_justetf().

Tests verify:
- Successful scraping with realistic HTML containing a valid holdings table
- Performance tables (containing YTD/Volatility keywords) are excluded
- Pages with no matching table return None
- Holdings with zero weight are excluded
- HTTP failures return None
- Pages with multiple tables correctly identify the holdings table

Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
"""

from unittest.mock import patch, MagicMock

import pytest
import requests

from etf_holdings import _fetch_justetf


# ---------------------------------------------------------------------------
# HTML fixtures
# ---------------------------------------------------------------------------


def _build_html_with_table(rows: list[tuple[str, str]], extra_html: str = "") -> str:
    """Build a minimal HTML page with a two-column table from (name, weight) rows."""
    table_rows = "\n".join(
        f"<tr><td>{name}</td><td>{weight}</td></tr>" for name, weight in rows
    )
    return f"""
    <html><body>
    {extra_html}
    <table>
    {table_rows}
    </table>
    </body></html>
    """


REALISTIC_HOLDINGS_HTML = _build_html_with_table([
    ("Apple Inc", "8.24%"),
    ("Microsoft Corp", "7.15%"),
    ("Amazon.com Inc", "5.02%"),
    ("NVIDIA Corp", "4.88%"),
    ("Alphabet Inc", "3.91%"),
    ("Meta Platforms Inc", "2.75%"),
    ("Tesla Inc", "2.10%"),
    ("Berkshire Hathaway", "1.95%"),
    ("JPMorgan Chase", "1.80%"),
    ("Johnson & Johnson", "1.65%"),
])

PERFORMANCE_TABLE_HTML = _build_html_with_table([
    ("YTD", "12.50%"),
    ("1 month", "2.30%"),
    ("3 months", "5.10%"),
    ("6 months", "8.20%"),
    ("1 year", "15.40%"),
    ("3 years", "42.10%"),
    ("5 years", "68.30%"),
    ("Volatility 1yr", "14.20%"),
    ("Max drawdown", "-8.50%"),
    ("Sharpe Ratio", "1.20%"),
])

HOLDINGS_WITH_ZERO_WEIGHT_HTML = _build_html_with_table([
    ("Apple Inc", "8.24%"),
    ("Microsoft Corp", "7.15%"),
    ("Amazon.com Inc", "5.02%"),
    ("NVIDIA Corp", "4.88%"),
    ("Alphabet Inc", "3.91%"),
    ("Cash", "0.00%"),
    ("Other", "0.00%"),
    ("Residual", "-0.50%"),
])

NO_MATCHING_TABLE_HTML = """
<html><body>
<table>
<tr><td>Fund Name</td><td>ISIN</td><td>TER</td></tr>
<tr><td>iShares Core S&P 500</td><td>IE00B5BMR087</td><td>0.07%</td></tr>
</table>
</body></html>
"""

MULTIPLE_TABLES_HTML = f"""
<html><body>
<table>
<tr><td>Fund Name</td><td>ISIN</td><td>TER</td></tr>
<tr><td>iShares Core S&P 500</td><td>IE00B5BMR087</td><td>0.07%</td></tr>
</table>

{PERFORMANCE_TABLE_HTML.split("<table>")[1].split("</table>")[0].join(["<table>", "</table>"])}

<table>
<tr><td>Apple Inc</td><td>8.24%</td></tr>
<tr><td>Microsoft Corp</td><td>7.15%</td></tr>
<tr><td>Amazon.com Inc</td><td>5.02%</td></tr>
<tr><td>NVIDIA Corp</td><td>4.88%</td></tr>
<tr><td>Alphabet Inc</td><td>3.91%</td></tr>
<tr><td>Meta Platforms Inc</td><td>2.75%</td></tr>
<tr><td>Tesla Inc</td><td>2.10%</td></tr>
</table>
</body></html>
"""


# ---------------------------------------------------------------------------
# Helper to create a mock response
# ---------------------------------------------------------------------------


def _mock_response(text: str, status_code: int = 200) -> MagicMock:
    """Create a mock requests.Response with given text and status."""
    mock = MagicMock()
    mock.ok = status_code == 200
    mock.status_code = status_code
    mock.text = text
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFetchJustETFSuccessfulScraping:
    """Test successful scraping with realistic HTML (Req 6.1, 6.2, 6.4)."""

    @patch("etf_holdings.requests.get")
    def test_parses_holdings_table_correctly(self, mock_get):
        """Verify correct parsing of a realistic holdings table."""
        mock_get.return_value = _mock_response(REALISTIC_HOLDINGS_HTML)

        result = _fetch_justetf("IE00B5BMR087")

        assert result is not None
        assert len(result) == 10
        assert list(result.columns) == ["ticker", "name", "weight"]

    @patch("etf_holdings.requests.get")
    def test_all_tickers_are_na(self, mock_get):
        """Verify ticker is always 'N/A' since justETF doesn't provide tickers (Req 6.4)."""
        mock_get.return_value = _mock_response(REALISTIC_HOLDINGS_HTML)

        result = _fetch_justetf("IE00B5BMR087")

        assert result is not None
        assert (result["ticker"] == "N/A").all()

    @patch("etf_holdings.requests.get")
    def test_weights_parsed_as_floats(self, mock_get):
        """Verify weight strings like '8.24%' are parsed to float 8.24."""
        mock_get.return_value = _mock_response(REALISTIC_HOLDINGS_HTML)

        result = _fetch_justetf("IE00B5BMR087")

        assert result is not None
        assert result["weight"].iloc[0] == pytest.approx(8.24)
        assert result["weight"].iloc[1] == pytest.approx(7.15)
        assert result["weight"].iloc[4] == pytest.approx(3.91)

    @patch("etf_holdings.requests.get")
    def test_names_preserved_from_html(self, mock_get):
        """Verify company names are preserved from the HTML table."""
        mock_get.return_value = _mock_response(REALISTIC_HOLDINGS_HTML)

        result = _fetch_justetf("IE00B5BMR087")

        assert result is not None
        assert result["name"].iloc[0] == "Apple Inc"
        assert result["name"].iloc[1] == "Microsoft Corp"


class TestFetchJustETFPerformanceTableExclusion:
    """Test that performance tables are excluded (Req 6.3)."""

    @patch("etf_holdings.requests.get")
    def test_excludes_table_with_ytd_keyword(self, mock_get):
        """A table containing 'YTD' in the first column should be skipped."""
        mock_get.return_value = _mock_response(PERFORMANCE_TABLE_HTML)

        result = _fetch_justetf("IE00B5BMR087")

        # Performance table is excluded, no other valid table exists
        assert result is None

    @patch("etf_holdings.requests.get")
    def test_excludes_table_with_volatility_keyword(self, mock_get):
        """A table containing 'Volatility' in the first column should be skipped."""
        html = _build_html_with_table([
            ("US Equities", "45.20%"),
            ("EU Equities", "25.10%"),
            ("Volatility 1yr", "14.20%"),
            ("Asia Pacific", "12.50%"),
            ("Emerging Markets", "8.30%"),
            ("Fixed Income", "4.70%"),
        ])
        mock_get.return_value = _mock_response(html)

        result = _fetch_justetf("IE00B5BMR087")

        assert result is None

    @patch("etf_holdings.requests.get")
    def test_excludes_table_with_drawdown_keyword(self, mock_get):
        """A table containing 'drawdown' in the first column should be skipped."""
        html = _build_html_with_table([
            ("Return 2020", "15.20%"),
            ("Return 2021", "22.10%"),
            ("Return 2022", "-12.50%"),
            ("Return 2023", "18.30%"),
            ("Max drawdown", "-18.50%"),
            ("Recovery time", "6.00%"),
        ])
        mock_get.return_value = _mock_response(html)

        result = _fetch_justetf("IE00B5BMR087")

        assert result is None

    @patch("etf_holdings.requests.get")
    def test_excludes_table_with_month_keyword(self, mock_get):
        """A table containing 'month' in the first column should be skipped."""
        html = _build_html_with_table([
            ("1 month", "2.30%"),
            ("3 months", "5.10%"),
            ("6 months", "8.20%"),
            ("9 months", "11.00%"),
            ("12 months", "15.40%"),
            ("18 months", "20.10%"),
        ])
        mock_get.return_value = _mock_response(html)

        result = _fetch_justetf("IE00B5BMR087")

        assert result is None

    @patch("etf_holdings.requests.get")
    def test_excludes_table_with_year_keyword(self, mock_get):
        """A table containing 'year' in the first column should be skipped."""
        html = _build_html_with_table([
            ("1 year", "15.40%"),
            ("2 years", "28.10%"),
            ("3 years", "42.10%"),
            ("5 years", "68.30%"),
            ("7 years", "95.20%"),
            ("10 years", "150.00%"),
        ])
        mock_get.return_value = _mock_response(html)

        result = _fetch_justetf("IE00B5BMR087")

        assert result is None


class TestFetchJustETFNoMatchingTable:
    """Test pages with no matching table return None (Req 6.7)."""

    @patch("etf_holdings.requests.get")
    def test_returns_none_for_no_two_column_table(self, mock_get):
        """A page with only a 3-column table should return None."""
        mock_get.return_value = _mock_response(NO_MATCHING_TABLE_HTML)

        result = _fetch_justetf("IE00B5BMR087")

        assert result is None

    @patch("etf_holdings.requests.get")
    def test_returns_none_for_table_with_fewer_than_5_rows(self, mock_get):
        """A two-column table with fewer than 5 rows should not match."""
        html = _build_html_with_table([
            ("Apple Inc", "8.24%"),
            ("Microsoft Corp", "7.15%"),
            ("Amazon.com Inc", "5.02%"),
        ])
        mock_get.return_value = _mock_response(html)

        result = _fetch_justetf("IE00B5BMR087")

        assert result is None

    @patch("etf_holdings.requests.get")
    def test_returns_none_for_table_without_enough_percent_cells(self, mock_get):
        """A table where fewer than 5 cells contain '%' should not match."""
        html = _build_html_with_table([
            ("Apple Inc", "8.24%"),
            ("Microsoft Corp", "7.15%"),
            ("Amazon.com Inc", "5.02%"),
            ("NVIDIA Corp", "4.88%"),
            ("Alphabet Inc", "N/A"),
            ("Meta Platforms", "Unknown"),
            ("Tesla Inc", "Pending"),
        ])
        mock_get.return_value = _mock_response(html)

        result = _fetch_justetf("IE00B5BMR087")

        assert result is None

    @patch("etf_holdings.requests.get")
    def test_returns_none_for_empty_page(self, mock_get):
        """An empty HTML page with no tables should return None."""
        mock_get.return_value = _mock_response("<html><body></body></html>")

        result = _fetch_justetf("IE00B5BMR087")

        assert result is None


class TestFetchJustETFZeroWeightExclusion:
    """Test that holdings with zero or negative weight are excluded (Req 6.5)."""

    @patch("etf_holdings.requests.get")
    def test_excludes_zero_weight_holdings(self, mock_get):
        """Holdings with weight 0.00% should be excluded from results."""
        mock_get.return_value = _mock_response(HOLDINGS_WITH_ZERO_WEIGHT_HTML)

        result = _fetch_justetf("IE00B5BMR087")

        assert result is not None
        # Only 5 rows have weight > 0 (Apple, Microsoft, Amazon, NVIDIA, Alphabet)
        assert len(result) == 5
        assert (result["weight"] > 0).all()

    @patch("etf_holdings.requests.get")
    def test_excludes_negative_weight_holdings(self, mock_get):
        """Holdings with negative weight should be excluded from results."""
        mock_get.return_value = _mock_response(HOLDINGS_WITH_ZERO_WEIGHT_HTML)

        result = _fetch_justetf("IE00B5BMR087")

        assert result is not None
        # The row with -0.50% should be excluded
        assert not (result["name"] == "Residual").any()


class TestFetchJustETFHTTPFailure:
    """Test HTTP failure handling (Req 6.7)."""

    @patch("etf_holdings.requests.get")
    def test_returns_none_on_http_error_status(self, mock_get):
        """Non-200 HTTP status should return None."""
        mock_get.return_value = _mock_response("", status_code=404)

        result = _fetch_justetf("IE00B5BMR087")

        assert result is None

    @patch("etf_holdings.requests.get")
    def test_returns_none_on_500_error(self, mock_get):
        """Server error (500) should return None."""
        mock_get.return_value = _mock_response("Internal Server Error", status_code=500)

        result = _fetch_justetf("IE00B5BMR087")

        assert result is None

    @patch("etf_holdings.requests.get")
    def test_returns_none_on_connection_error(self, mock_get):
        """Connection errors should return None."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        result = _fetch_justetf("IE00B5BMR087")

        assert result is None

    @patch("etf_holdings.requests.get")
    def test_returns_none_on_timeout(self, mock_get):
        """Timeout errors should return None."""
        mock_get.side_effect = requests.exceptions.Timeout("Request timed out")

        result = _fetch_justetf("IE00B5BMR087")

        assert result is None


class TestFetchJustETFMultipleTables:
    """Test correct table identification with multiple tables (Req 6.2, 6.3)."""

    @patch("etf_holdings.requests.get")
    def test_identifies_correct_table_among_multiple(self, mock_get):
        """When page has multiple tables, the valid holdings table is identified."""
        mock_get.return_value = _mock_response(MULTIPLE_TABLES_HTML)

        result = _fetch_justetf("IE00B5BMR087")

        assert result is not None
        # Should find the holdings table (7 rows with valid holdings)
        assert len(result) == 7
        assert result["name"].iloc[0] == "Apple Inc"
        assert result["weight"].iloc[0] == pytest.approx(8.24)

    @patch("etf_holdings.requests.get")
    def test_skips_performance_table_selects_holdings(self, mock_get):
        """Performance table is skipped, holdings table is selected."""
        # Build HTML with performance table first, then holdings table
        perf_rows = [
            ("YTD", "12.50%"),
            ("1 month", "2.30%"),
            ("3 months", "5.10%"),
            ("6 months", "8.20%"),
            ("1 year", "15.40%"),
            ("3 years", "42.10%"),
        ]
        holdings_rows = [
            ("Apple Inc", "8.24%"),
            ("Microsoft Corp", "7.15%"),
            ("Amazon.com Inc", "5.02%"),
            ("NVIDIA Corp", "4.88%"),
            ("Alphabet Inc", "3.91%"),
            ("Meta Platforms Inc", "2.75%"),
        ]
        perf_table = "\n".join(
            f"<tr><td>{n}</td><td>{w}</td></tr>" for n, w in perf_rows
        )
        holdings_table = "\n".join(
            f"<tr><td>{n}</td><td>{w}</td></tr>" for n, w in holdings_rows
        )
        html = f"""
        <html><body>
        <table>{perf_table}</table>
        <table>{holdings_table}</table>
        </body></html>
        """
        mock_get.return_value = _mock_response(html)

        result = _fetch_justetf("IE00B5BMR087")

        assert result is not None
        assert len(result) == 6
        assert result["name"].iloc[0] == "Apple Inc"
        assert (result["ticker"] == "N/A").all()
