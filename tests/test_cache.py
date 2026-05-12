"""
Unit tests for cache loading and saving resilience.

Tests verify:
- _load_cache() returns {} for malformed JSON, empty files, and unreadable files
- _load_cache() never raises exceptions
- _save_cache() continues silently on write failure
- _save_cache() never raises exceptions

Validates: Requirements 12.6, 12.7
"""

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from etf_holdings import _load_cache, _save_cache, _CACHE_PATH


# ---------------------------------------------------------------------------
# _load_cache tests
# ---------------------------------------------------------------------------


class TestLoadCache:
    """Tests for _load_cache resilience (Requirement 12.6)."""

    def test_returns_empty_dict_when_file_does_not_exist(self, tmp_path):
        fake_path = str(tmp_path / "nonexistent.json")
        with patch("etf_holdings._CACHE_PATH", fake_path):
            result = _load_cache()
        assert result == {}

    def test_returns_empty_dict_for_empty_file(self, tmp_path):
        cache_file = tmp_path / ".etf_cache.json"
        cache_file.write_text("", encoding="utf-8")
        with patch("etf_holdings._CACHE_PATH", str(cache_file)):
            result = _load_cache()
        assert result == {}

    def test_returns_empty_dict_for_malformed_json(self, tmp_path):
        cache_file = tmp_path / ".etf_cache.json"
        cache_file.write_text("{invalid json content!!!", encoding="utf-8")
        with patch("etf_holdings._CACHE_PATH", str(cache_file)):
            result = _load_cache()
        assert result == {}

    def test_returns_empty_dict_for_partial_json(self, tmp_path):
        cache_file = tmp_path / ".etf_cache.json"
        cache_file.write_text('{"CSPX.L": {"isin": "IE00B5BMR087"', encoding="utf-8")
        with patch("etf_holdings._CACHE_PATH", str(cache_file)):
            result = _load_cache()
        assert result == {}

    def test_returns_empty_dict_for_json_array(self, tmp_path):
        """JSON that parses but is not a dict should return {}."""
        cache_file = tmp_path / ".etf_cache.json"
        cache_file.write_text('[1, 2, 3]', encoding="utf-8")
        with patch("etf_holdings._CACHE_PATH", str(cache_file)):
            result = _load_cache()
        assert result == {}

    def test_returns_empty_dict_for_json_string(self, tmp_path):
        """JSON that parses to a string should return {}."""
        cache_file = tmp_path / ".etf_cache.json"
        cache_file.write_text('"just a string"', encoding="utf-8")
        with patch("etf_holdings._CACHE_PATH", str(cache_file)):
            result = _load_cache()
        assert result == {}

    def test_returns_valid_cache_data(self, tmp_path):
        cache_file = tmp_path / ".etf_cache.json"
        data = {"CSPX.L": {"isin": "IE00B5BMR087", "provider": "ishares"}}
        cache_file.write_text(json.dumps(data), encoding="utf-8")
        with patch("etf_holdings._CACHE_PATH", str(cache_file)):
            result = _load_cache()
        assert result == data

    def test_returns_empty_dict_on_permission_error(self, tmp_path):
        """Simulate an unreadable file via mocked open raising PermissionError."""
        cache_file = tmp_path / ".etf_cache.json"
        cache_file.write_text("{}", encoding="utf-8")
        with patch("etf_holdings._CACHE_PATH", str(cache_file)):
            with patch("builtins.open", side_effect=PermissionError("Access denied")):
                result = _load_cache()
        assert result == {}

    def test_returns_empty_dict_on_os_error(self, tmp_path):
        """Simulate an OS-level read error."""
        cache_file = tmp_path / ".etf_cache.json"
        cache_file.write_text("{}", encoding="utf-8")
        with patch("etf_holdings._CACHE_PATH", str(cache_file)):
            with patch("builtins.open", side_effect=OSError("Disk error")):
                result = _load_cache()
        assert result == {}

    def test_never_raises_exception(self, tmp_path):
        """_load_cache should never raise, regardless of file content."""
        cache_file = tmp_path / ".etf_cache.json"
        # Write binary garbage
        cache_file.write_bytes(b"\x00\x01\x02\xff\xfe")
        with patch("etf_holdings._CACHE_PATH", str(cache_file)):
            # Should not raise
            result = _load_cache()
        assert result == {}


# ---------------------------------------------------------------------------
# _save_cache tests
# ---------------------------------------------------------------------------


class TestSaveCache:
    """Tests for _save_cache resilience (Requirement 12.7)."""

    def test_saves_valid_cache(self, tmp_path):
        cache_file = tmp_path / ".etf_cache.json"
        data = {"CSPX.L": {"isin": "IE00B5BMR087"}}
        with patch("etf_holdings._CACHE_PATH", str(cache_file)):
            _save_cache(data)
        assert json.loads(cache_file.read_text(encoding="utf-8")) == data

    def test_silent_on_permission_error(self, tmp_path):
        """_save_cache should not raise on write permission errors."""
        data = {"CSPX.L": {"isin": "IE00B5BMR087"}}
        with patch("etf_holdings._CACHE_PATH", str(tmp_path / "cache.json")):
            with patch("builtins.open", side_effect=PermissionError("Read-only")):
                # Should not raise
                _save_cache(data)

    def test_silent_on_os_error(self, tmp_path):
        """_save_cache should not raise on OS errors (disk full, etc.)."""
        data = {"CSPX.L": {"isin": "IE00B5BMR087"}}
        with patch("etf_holdings._CACHE_PATH", str(tmp_path / "cache.json")):
            with patch("builtins.open", side_effect=OSError("Disk full")):
                # Should not raise
                _save_cache(data)

    def test_silent_on_readonly_directory(self, tmp_path):
        """_save_cache should not raise when directory is not writable."""
        data = {"CSPX.L": {"isin": "IE00B5BMR087"}}
        fake_path = str(tmp_path / "nonexistent_dir" / "cache.json")
        # Don't create the directory - open will fail with FileNotFoundError
        with patch("etf_holdings._CACHE_PATH", fake_path):
            # Should not raise
            _save_cache(data)

    def test_never_raises_exception(self):
        """_save_cache should never raise, regardless of the error type."""
        data = {"CSPX.L": {"isin": "IE00B5BMR087"}}
        with patch("builtins.open", side_effect=Exception("Unexpected error")):
            with patch("etf_holdings._CACHE_PATH", "/impossible/path/cache.json"):
                # Should not raise
                _save_cache(data)
