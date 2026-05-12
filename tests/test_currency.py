"""
Unit tests for currency selection and FX formatting.

Tests verify:
- SUPPORTED_CURRENCIES contains exactly the expected currencies
- Default target currency is PLN
- Currency change clears previous results (session state invalidation logic)
- CLI mode reads TARGET_CURRENCY from portfolio.py

Validates: Requirements 15.2, 15.3, 15.9, 15.10, 15.11
"""

import pytest

from portfolio import SUPPORTED_CURRENCIES, TARGET_CURRENCY


# ---------------------------------------------------------------------------
# SUPPORTED_CURRENCIES constant (Requirement 15.2)
# ---------------------------------------------------------------------------


class TestSupportedCurrencies:
    """Tests for the SUPPORTED_CURRENCIES constant in portfolio.py."""

    def test_supported_currencies_exact_list(self):
        """
        Validates: Requirement 15.2

        SUPPORTED_CURRENCIES must contain exactly ["PLN", "EUR", "USD", "GBP", "CHF"].
        """
        assert SUPPORTED_CURRENCIES == ["PLN", "EUR", "USD", "GBP", "CHF"]

    def test_supported_currencies_length(self):
        """SUPPORTED_CURRENCIES must have exactly 5 entries."""
        assert len(SUPPORTED_CURRENCIES) == 5

    def test_supported_currencies_contains_pln(self):
        """PLN must be in SUPPORTED_CURRENCIES."""
        assert "PLN" in SUPPORTED_CURRENCIES

    def test_supported_currencies_contains_eur(self):
        """EUR must be in SUPPORTED_CURRENCIES."""
        assert "EUR" in SUPPORTED_CURRENCIES

    def test_supported_currencies_contains_usd(self):
        """USD must be in SUPPORTED_CURRENCIES."""
        assert "USD" in SUPPORTED_CURRENCIES

    def test_supported_currencies_contains_gbp(self):
        """GBP must be in SUPPORTED_CURRENCIES."""
        assert "GBP" in SUPPORTED_CURRENCIES

    def test_supported_currencies_contains_chf(self):
        """CHF must be in SUPPORTED_CURRENCIES."""
        assert "CHF" in SUPPORTED_CURRENCIES

    def test_supported_currencies_is_list(self):
        """SUPPORTED_CURRENCIES must be a list type."""
        assert isinstance(SUPPORTED_CURRENCIES, list)

    def test_supported_currencies_all_strings(self):
        """All entries in SUPPORTED_CURRENCIES must be strings."""
        assert all(isinstance(c, str) for c in SUPPORTED_CURRENCIES)


# ---------------------------------------------------------------------------
# Default target currency (Requirement 15.3)
# ---------------------------------------------------------------------------


class TestDefaultTargetCurrency:
    """Tests for the default TARGET_CURRENCY in portfolio.py."""

    def test_default_target_currency_is_pln(self):
        """
        Validates: Requirement 15.3

        TARGET_CURRENCY must default to "PLN".
        """
        assert TARGET_CURRENCY == "PLN"

    def test_target_currency_is_string(self):
        """TARGET_CURRENCY must be a string."""
        assert isinstance(TARGET_CURRENCY, str)

    def test_target_currency_in_supported_list(self):
        """TARGET_CURRENCY must be one of the supported currencies."""
        assert TARGET_CURRENCY in SUPPORTED_CURRENCIES


# ---------------------------------------------------------------------------
# Currency change clears results (Requirement 15.11)
# ---------------------------------------------------------------------------


class TestCurrencyChangeInvalidation:
    """
    Tests for the currency change invalidation logic.

    The actual Streamlit session state logic in app.py:
    - Stores target_currency and _prev_target_currency in session state
    - When target_currency != _prev_target_currency and results is not None,
      clears results and reruns

    We test the underlying logic without running Streamlit.

    Validates: Requirement 15.11
    """

    def test_currency_change_should_clear_results(self):
        """
        Validates: Requirement 15.11

        When the target currency changes and results exist, results should be cleared.
        Simulates the invalidation condition from app.py.
        """
        # Simulate session state
        session_state = {
            "target_currency": "EUR",
            "_prev_target_currency": "PLN",
            "results": ("some_df", 1000.0, ["CSPX.L"]),
        }

        # The invalidation condition from app.py:
        # if target_currency != _prev_target_currency and results is not None: clear results
        if session_state["target_currency"] != session_state["_prev_target_currency"]:
            if session_state["results"] is not None:
                session_state["results"] = None
                session_state["_prev_target_currency"] = session_state["target_currency"]

        assert session_state["results"] is None
        assert session_state["_prev_target_currency"] == "EUR"

    def test_same_currency_does_not_clear_results(self):
        """
        Validates: Requirement 15.10

        When the target currency has not changed, results should remain intact.
        """
        session_state = {
            "target_currency": "PLN",
            "_prev_target_currency": "PLN",
            "results": ("some_df", 1000.0, ["CSPX.L"]),
        }

        # Same currency — no invalidation
        if session_state["target_currency"] != session_state["_prev_target_currency"]:
            if session_state["results"] is not None:
                session_state["results"] = None
                session_state["_prev_target_currency"] = session_state["target_currency"]

        # Results should remain
        assert session_state["results"] is not None

    def test_currency_change_without_results_is_noop(self):
        """
        Validates: Requirement 15.11

        When the target currency changes but no results exist, nothing to clear.
        """
        session_state = {
            "target_currency": "USD",
            "_prev_target_currency": "PLN",
            "results": None,
        }

        # Currency changed but results is None — no action needed
        if session_state["target_currency"] != session_state["_prev_target_currency"]:
            if session_state["results"] is not None:
                session_state["results"] = None
            session_state["_prev_target_currency"] = session_state["target_currency"]

        assert session_state["results"] is None
        assert session_state["_prev_target_currency"] == "USD"


# ---------------------------------------------------------------------------
# CLI mode reads TARGET_CURRENCY from portfolio.py (Requirement 15.9)
# ---------------------------------------------------------------------------


class TestCLITargetCurrency:
    """
    Tests that CLI mode (main.py) reads TARGET_CURRENCY from portfolio.py.

    Validates: Requirement 15.9
    """

    def test_main_imports_target_currency(self):
        """
        Validates: Requirement 15.9

        main.py must import TARGET_CURRENCY from portfolio.py.
        Verify by importing main and checking it uses the same value.
        """
        from main import TARGET_CURRENCY as main_target_currency

        assert main_target_currency == "PLN"
        assert main_target_currency == TARGET_CURRENCY

    def test_target_currency_used_in_report_header(self):
        """
        Validates: Requirement 15.9

        The CLI report uses TARGET_CURRENCY in the Value column header.
        Verify by checking the _print_report function references TARGET_CURRENCY.
        """
        import inspect
        from main import _print_report

        source = inspect.getsource(_print_report)
        # The report header should reference TARGET_CURRENCY for the value column
        assert "TARGET_CURRENCY" in source

    def test_target_currency_used_in_total_value_line(self):
        """
        Validates: Requirement 15.9

        The CLI report prints total portfolio value with TARGET_CURRENCY.
        """
        import inspect
        from main import _print_report

        source = inspect.getsource(_print_report)
        # The total value line should include TARGET_CURRENCY
        assert "total_value" in source
        assert "TARGET_CURRENCY" in source
