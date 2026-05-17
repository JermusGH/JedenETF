"""
Data models for the Unified ETF Portfolio tool.

Provides structured types for passing analysis results between modules,
replacing fragile tuple unpacking.
"""

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class AnalysisResult:
    """Result of a portfolio analysis run."""

    df: pd.DataFrame
    total_value: float
    sources: list[str] = field(default_factory=list)
    failed_tickers: list[str] = field(default_factory=list)
    justetf_tickers: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return self.df.empty or self.total_value == 0


@dataclass
class FetchResult:
    """Result of a single ETF holdings fetch."""

    holdings: pd.DataFrame | None
    used_justetf: bool = False
