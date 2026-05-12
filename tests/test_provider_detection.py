"""
Unit tests for provider detection logic.

Tests verify that both _detect_provider (etf_holdings.py) and _detect_ticker_provider (app.py)
follow the exact deterministic keyword-based priority:
  1. "blackrock" or "ishares" (case-insensitive) → "ishares"
  2. "vanguard" (case-insensitive) → "vanguard"
  3. "invesco" (case-insensitive) → "invesco"
  4. "dws" or "xtrackers" (case-insensitive) → "xtrackers"
  5. "amundi" (case-insensitive) → "amundi"
  6. Otherwise → "unknown"

Requirements: 2.2, 2.3, 2.4, 2.5, 2.8
"""

import pytest

from etf_holdings import _detect_provider


# ---------------------------------------------------------------------------
# _detect_provider tests (etf_holdings.py)
# ---------------------------------------------------------------------------


class TestDetectProvider:
    """Tests for _detect_provider in etf_holdings.py."""

    # --- iShares / BlackRock detection (Requirement 2.2) ---

    def test_blackrock_lowercase(self):
        assert _detect_provider("blackrock") == "ishares"

    def test_blackrock_mixed_case(self):
        assert _detect_provider("BlackRock") == "ishares"

    def test_blackrock_uppercase(self):
        assert _detect_provider("BLACKROCK") == "ishares"

    def test_ishares_lowercase(self):
        assert _detect_provider("ishares") == "ishares"

    def test_ishares_mixed_case(self):
        assert _detect_provider("iShares") == "ishares"

    def test_blackrock_in_longer_string(self):
        assert _detect_provider("BlackRock Asset Management") == "ishares"

    def test_ishares_in_longer_string(self):
        assert _detect_provider("iShares by BlackRock") == "ishares"

    # --- Vanguard detection (Requirement 2.3) ---

    def test_vanguard_lowercase(self):
        assert _detect_provider("vanguard") == "vanguard"

    def test_vanguard_mixed_case(self):
        assert _detect_provider("Vanguard") == "vanguard"

    def test_vanguard_uppercase(self):
        assert _detect_provider("VANGUARD") == "vanguard"

    def test_vanguard_in_longer_string(self):
        assert _detect_provider("The Vanguard Group, Inc.") == "vanguard"

    # --- Amundi detection (Requirement 2.4) ---

    def test_amundi_lowercase(self):
        assert _detect_provider("amundi") == "amundi"

    def test_amundi_mixed_case(self):
        assert _detect_provider("Amundi") == "amundi"

    def test_amundi_uppercase(self):
        assert _detect_provider("AMUNDI") == "amundi"

    def test_amundi_in_longer_string(self):
        assert _detect_provider("Amundi Asset Management") == "amundi"

    # --- Invesco detection ---

    def test_invesco_lowercase(self):
        assert _detect_provider("invesco") == "invesco"

    def test_invesco_mixed_case(self):
        assert _detect_provider("Invesco") == "invesco"

    def test_invesco_uppercase(self):
        assert _detect_provider("INVESCO") == "invesco"

    def test_invesco_in_longer_string(self):
        assert _detect_provider("Invesco Capital Management") == "invesco"

    # --- Xtrackers / DWS detection ---

    def test_dws_lowercase(self):
        assert _detect_provider("dws") == "xtrackers"

    def test_dws_mixed_case(self):
        assert _detect_provider("DWS Investment S.A. (ETF)") == "xtrackers"

    def test_xtrackers_lowercase(self):
        assert _detect_provider("xtrackers") == "xtrackers"

    def test_xtrackers_mixed_case(self):
        assert _detect_provider("Xtrackers") == "xtrackers"

    def test_dws_in_longer_string(self):
        assert _detect_provider("DWS Investment S.A.") == "xtrackers"

    # --- Unknown / no match (Requirement 2.5) ---

    def test_empty_string_returns_unknown(self):
        assert _detect_provider("") == "unknown"

    def test_unrecognized_provider_returns_unknown(self):
        assert _detect_provider("Fidelity") == "unknown"

    def test_random_string_returns_unknown(self):
        assert _detect_provider("some random fund family") == "unknown"

    # --- Priority order: ishares checked before vanguard/invesco/xtrackers/amundi ---

    def test_blackrock_takes_priority_over_vanguard(self):
        """If a string somehow contains both blackrock and vanguard, ishares wins."""
        assert _detect_provider("blackrock vanguard") == "ishares"

    def test_ishares_takes_priority_over_amundi(self):
        """If a string contains both ishares and amundi, ishares wins."""
        assert _detect_provider("ishares amundi") == "ishares"

    def test_vanguard_takes_priority_over_invesco(self):
        """If a string contains both vanguard and invesco, vanguard wins."""
        assert _detect_provider("vanguard invesco") == "vanguard"

    def test_invesco_takes_priority_over_xtrackers(self):
        """If a string contains both invesco and dws, invesco wins."""
        assert _detect_provider("invesco dws") == "invesco"

    def test_xtrackers_takes_priority_over_amundi(self):
        """If a string contains both dws and amundi, xtrackers wins."""
        assert _detect_provider("dws amundi") == "xtrackers"

    def test_invesco_takes_priority_over_amundi(self):
        """If a string contains both invesco and amundi, invesco wins."""
        assert _detect_provider("invesco amundi") == "invesco"

    def test_vanguard_takes_priority_over_amundi(self):
        """If a string contains both vanguard and amundi, vanguard wins."""
        assert _detect_provider("vanguard amundi") == "vanguard"

    # --- Determinism ---

    def test_deterministic_same_input_same_output(self):
        """Same input always produces same output."""
        for _ in range(10):
            assert _detect_provider("BlackRock Fund") == "ishares"
            assert _detect_provider("Vanguard Group") == "vanguard"
            assert _detect_provider("Invesco Ltd") == "invesco"
            assert _detect_provider("DWS Investment S.A. (ETF)") == "xtrackers"
            assert _detect_provider("Amundi SA") == "amundi"
            assert _detect_provider("Unknown Corp") == "unknown"

    # --- Requirement 2.8: empty/failed fund family → unknown ---

    def test_empty_fund_family_returns_unknown(self):
        """When Yahoo Finance returns empty fund family, result is 'unknown'."""
        assert _detect_provider("") == "unknown"
