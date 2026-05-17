"""
Unit tests for the CLI analyser output format (main.py).

Tests cover:
- Top-20 sorting by weight descending (Req 13.2)
- Column alignment and header formatting
- Contribution formatting: "-" for contributions ≤ 0.01% (Req 13.3)
- Total value and unique holdings count printed after table (Req 13.3)
- Empty portfolio error message (Req 13.4)
- No holdings available error message (Req 13.5)
"""

import io
import sys
from unittest.mock import patch

import pandas as pd
import pytest

from main import _print_report, main
from models import AnalysisResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_final_df(n: int = 25, sources: list[str] | None = None) -> pd.DataFrame:
    """
    Create a synthetic merged DataFrame with *n* holdings.

    Each holding has a unique ticker, name, value, weight_%, and per-source
    contribution columns.
    """
    if sources is None:
        sources = ["ETF_A", "ETF_B"]

    rows = []
    for i in range(n):
        row = {
            "ticker": f"TKR{i:03d}",
            "name": f"Company {i:03d}",
            "weight_%": float(n - i),  # descending: n, n-1, ..., 1
            "value": (n - i) * 100.0,
        }
        for j, src in enumerate(sources):
            # Alternate: first source gets most contribution, second gets tiny
            if j == 0:
                row[src] = float(n - i) * 0.6
            else:
                row[src] = float(n - i) * 0.4
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# _print_report tests
# ---------------------------------------------------------------------------


class TestPrintReportSorting:
    """Verify output is sorted by weight descending (Req 13.2)."""

    def test_top20_sorted_by_weight_descending(self, capsys):
        """The printed table should list holdings from highest to lowest weight."""
        sources = ["ETF_A", "ETF_B"]
        df = _make_final_df(25, sources)
        # Shuffle the DataFrame so sorting is actually tested
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        _print_report(AnalysisResult(df=df, total_value=10000.0, sources=sources))

        captured = capsys.readouterr().out
        lines = captured.strip().split("\n")

        # Find data lines: lines after the dash separator that contain "|"
        # and are NOT the header (header contains "Ticker")
        data_lines = []
        past_dash_sep = False
        for line in lines:
            if line and all(c == "-" for c in line):
                past_dash_sep = True
                continue
            if past_dash_sep and "|" in line and "Ticker" not in line:
                # Stop at the final dash separator (after data rows)
                if line and all(c == "-" for c in line.strip()):
                    break
                data_lines.append(line)

        # Should have exactly 20 data rows
        assert len(data_lines) == 20

        # Extract weights from data lines
        weights = []
        for line in data_lines:
            parts = [p.strip() for p in line.split("|")]
            # Weight is the 3rd column (index 2)
            weight_str = parts[2].replace("%", "").strip()
            weights.append(float(weight_str))

        # Verify descending order
        assert weights == sorted(weights, reverse=True)

    def test_only_top20_printed_when_more_holdings_exist(self, capsys):
        """When more than 20 holdings exist, only top 20 are printed."""
        sources = ["ETF_A"]
        df = _make_final_df(30, sources)

        _print_report(AnalysisResult(df=df, total_value=10000.0, sources=sources))

        captured = capsys.readouterr().out
        lines = captured.strip().split("\n")

        # Count data lines with ticker pattern
        data_lines = [l for l in lines if "TKR" in l and "|" in l]
        assert len(data_lines) == 20


class TestPrintReportColumnAlignment:
    """Verify the table has proper column headers and alignment."""

    def test_header_contains_expected_columns(self, capsys):
        """Header row should contain Ticker, Company, Weight, Value columns."""
        sources = ["ETF_A"]
        df = _make_final_df(5, sources)

        _print_report(AnalysisResult(df=df, total_value=5000.0, sources=sources))

        captured = capsys.readouterr().out
        lines = captured.strip().split("\n")

        # Find the header line (contains "Ticker" and "Company")
        header_line = None
        for line in lines:
            if "Ticker" in line and "Company" in line:
                header_line = line
                break

        assert header_line is not None
        assert "Ticker" in header_line
        assert "Company" in header_line
        assert "Weight" in header_line
        assert "Value" in header_line

    def test_header_contains_source_columns(self, capsys):
        """Header should include per-ETF source columns."""
        sources = ["CSPX.L", "VWCE.DE"]
        df = _make_final_df(5, sources)

        _print_report(AnalysisResult(df=df, total_value=5000.0, sources=sources))

        captured = capsys.readouterr().out

        assert "CSPX.L" in captured
        assert "VWCE.DE" in captured

    def test_separator_lines_present(self, capsys):
        """Output should contain separator lines (dashes or equals)."""
        sources = ["ETF_A"]
        df = _make_final_df(5, sources)

        _print_report(AnalysisResult(df=df, total_value=5000.0, sources=sources))

        captured = capsys.readouterr().out
        lines = captured.strip().split("\n")

        # Should have lines made of "=" and lines made of "-"
        eq_lines = [l for l in lines if l and all(c == "=" for c in l)]
        dash_lines = [l for l in lines if l and all(c == "-" for c in l)]

        assert len(eq_lines) >= 2  # top and bottom of title
        assert len(dash_lines) >= 1  # separator after header


class TestPrintReportContributionFormatting:
    """Verify that contributions ≤ 0.01% show "-" (Req 13.3)."""

    def test_tiny_contribution_shows_dash(self, capsys):
        """Contributions of 0.005 or less should display as '-'."""
        sources = ["ETF_A", "ETF_B"]
        df = pd.DataFrame([
            {
                "ticker": "AAPL",
                "name": "Apple Inc",
                "weight_%": 5.0,
                "value": 500.0,
                "ETF_A": 4.99,
                "ETF_B": 0.005,  # ≤ 0.01% → should show "-"
            },
        ])

        _print_report(AnalysisResult(df=df, total_value=10000.0, sources=sources))

        captured = capsys.readouterr().out
        lines = [l for l in captured.split("\n") if "AAPL" in l]
        assert len(lines) == 1

        # The line should contain a "-" for the tiny contribution
        parts = lines[0].split("|")
        # Last source column (ETF_B) should be "-"
        etf_b_cell = parts[-1].strip()
        assert etf_b_cell == "-"

    def test_normal_contribution_shows_percentage(self, capsys):
        """Contributions > 0.005 should display as formatted percentage."""
        sources = ["ETF_A"]
        df = pd.DataFrame([
            {
                "ticker": "MSFT",
                "name": "Microsoft Corp",
                "weight_%": 3.5,
                "value": 350.0,
                "ETF_A": 3.5,
            },
        ])

        _print_report(AnalysisResult(df=df, total_value=10000.0, sources=sources))

        captured = capsys.readouterr().out
        lines = [l for l in captured.split("\n") if "MSFT" in l]
        assert len(lines) == 1

        # Should contain "3.50%" somewhere in the source column
        assert "3.50%" in lines[0]

    def test_zero_contribution_shows_dash(self, capsys):
        """Contributions of exactly 0.0 should display as '-'."""
        sources = ["ETF_A", "ETF_B"]
        df = pd.DataFrame([
            {
                "ticker": "GOOG",
                "name": "Alphabet Inc",
                "weight_%": 2.0,
                "value": 200.0,
                "ETF_A": 2.0,
                "ETF_B": 0.0,
            },
        ])

        _print_report(AnalysisResult(df=df, total_value=10000.0, sources=sources))

        captured = capsys.readouterr().out
        lines = [l for l in captured.split("\n") if "GOOG" in l]
        parts = lines[0].split("|")
        etf_b_cell = parts[-1].strip()
        assert etf_b_cell == "-"


class TestPrintReportTotalAndCount:
    """Verify total value and unique holdings count are printed (Req 13.3)."""

    def test_total_value_printed(self, capsys):
        """Output should contain the total portfolio value formatted."""
        sources = ["ETF_A"]
        df = _make_final_df(5, sources)

        _print_report(AnalysisResult(df=df, total_value=12345.67, sources=sources))

        captured = capsys.readouterr().out
        assert "12,345.67" in captured
        assert "PLN" in captured

    def test_unique_holdings_count_printed(self, capsys):
        """Output should contain the count of unique holdings."""
        sources = ["ETF_A"]
        df = _make_final_df(8, sources)

        _print_report(AnalysisResult(df=df, total_value=5000.0, sources=sources))

        captured = capsys.readouterr().out
        # The full DataFrame has 8 holdings
        assert "8" in captured
        assert "Unique holdings" in captured


# ---------------------------------------------------------------------------
# main() tests — empty portfolio and no holdings
# ---------------------------------------------------------------------------


class TestMainEmptyPortfolio:
    """When PORTFOLIO is empty, verify error message is printed (Req 13.4)."""

    @patch("main.PORTFOLIO", {})
    def test_empty_portfolio_prints_error(self, capsys):
        """Empty PORTFOLIO should print an error and return without table."""
        main()

        captured = capsys.readouterr().out
        assert "PORTFOLIO is empty" in captured

    @patch("main.PORTFOLIO", {})
    def test_empty_portfolio_no_table_printed(self, capsys):
        """Empty PORTFOLIO should not produce any table output."""
        main()

        captured = capsys.readouterr().out
        assert "TOP 20" not in captured
        assert "Ticker" not in captured


class TestMainNoHoldings:
    """When all ETFs fail to return holdings, verify error message (Req 13.5)."""

    @patch("main.PORTFOLIO", {"CSPX.L": 0.0, "VWCE.DE": 5.0})
    @patch("main.fetch_prices")
    @patch("main.get_fund_family")
    @patch("main.HoldingsFetcher")
    def test_no_holdings_prints_error(
        self, mock_fetcher_cls, mock_family, mock_prices, capsys
    ):
        """When fetch_holdings returns None for all ETFs, error is printed."""
        mock_prices.return_value = {"CSPX.L": 0.0, "VWCE.DE": 50.0}
        mock_family.return_value = "ishares"
        mock_fetcher_cls.return_value.fetch.return_value = None
        mock_fetcher_cls.return_value.justetf_tickers = set()

        main()

        captured = capsys.readouterr().out
        assert "No holdings data available" in captured

    @patch("main.PORTFOLIO", {"CSPX.L": 0.0})
    @patch("main.fetch_prices")
    @patch("main.get_fund_family")
    @patch("main.HoldingsFetcher")
    def test_no_holdings_no_table_printed(
        self, mock_fetcher_cls, mock_family, mock_prices, capsys
    ):
        """When no holdings are available, no table should be printed."""
        mock_prices.return_value = {"CSPX.L": 0.0}
        mock_family.return_value = "vanguard"
        mock_fetcher_cls.return_value.fetch.return_value = pd.DataFrame()
        mock_fetcher_cls.return_value.justetf_tickers = set()

        main()

        captured = capsys.readouterr().out
        assert "TOP 20" not in captured
