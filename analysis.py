"""
Analysis Engine — shared merging and weight calculation logic.

Provides reusable functions for:
- Weight interpretation heuristic (percentage vs decimal fraction)
- Merging holdings by normalised name with deduplication
- Calculating unified weights as percentage of total portfolio value
- Building per-ETF contribution pivot
"""

import pandas as pd

from name_normaliser import normalise_name

# If the sum of all weight values for a single ETF exceeds this threshold,
# weights are interpreted as percentages (e.g. 8.24 means 8.24%).
# Otherwise they are treated as decimal fractions (e.g. 0.0824 means 8.24%).
# The value 10 was chosen because no real ETF has fewer than 10 holdings
# summing to less than 10% total, while decimal fractions always sum to ≤ 1.
_PERCENTAGE_WEIGHT_THRESHOLD = 10.0


def calculate_holding_values(df: pd.DataFrame, etf_value: float) -> pd.DataFrame:
    """
    Apply the weight interpretation heuristic and calculate holding values.

    If weights sum > _PERCENTAGE_WEIGHT_THRESHOLD (10), treat as percentages:
        value = (weight / 100) * etf_value.
    If weights sum <= threshold, treat as decimal fractions:
        value = weight * etf_value.

    Parameters
    ----------
    df : pd.DataFrame
        Holdings DataFrame with at least columns: ticker, name, weight.
    etf_value : float
        Total value of the ETF position in target currency.

    Returns
    -------
    pd.DataFrame
        Copy of input with an added 'value' column.
    """
    result = df.copy()
    weight_sum = result["weight"].sum()
    if weight_sum > _PERCENTAGE_WEIGHT_THRESHOLD:
        result["value"] = (result["weight"] / 100.0) * etf_value
    else:
        result["value"] = result["weight"] * etf_value
    return result


def merge_holdings(frames: list[pd.DataFrame], total_value: float) -> pd.DataFrame:
    """
    Merge multiple enriched holdings DataFrames into a unified portfolio view.

    Performs name-based deduplication using normalise_name, sums values for
    holdings sharing the same normalised name, calculates unified weights,
    and builds a per-ETF contribution pivot.

    Parameters
    ----------
    frames : list[pd.DataFrame]
        List of DataFrames, each with columns: ticker, name, weight, value, source, merge_key.
    total_value : float
        Total portfolio value across all ETFs (used for weight calculation).

    Returns
    -------
    pd.DataFrame
        Unified DataFrame with columns: merge_key, ticker, name, value, weight_%,
        plus one column per ETF source showing contribution percentage.
        Returns empty DataFrame if frames is empty or total_value is zero.
    """
    if not frames or total_value == 0:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Prefer shorter ticker strings (real tickers over ISINs), excluding "N/A"
    def _ticker_sort_key(t: str) -> tuple[int, int]:
        s = str(t)
        # Push "N/A" to the end
        if s.upper() == "N/A":
            return (1, len(s))
        return (0, len(s))

    combined["_ticker_sort"] = combined["ticker"].apply(lambda t: _ticker_sort_key(t))
    combined = combined.sort_values("_ticker_sort")

    grouped = (
        combined.groupby("merge_key")
        .agg(ticker=("ticker", "first"), name=("name", "first"), value=("value", "sum"))
        .reset_index()
    )
    grouped["weight_%"] = (grouped["value"] / total_value) * 100.0

    # Per-source contribution pivot
    combined["contrib_%"] = (combined["value"] / total_value) * 100.0
    pivot = combined.pivot_table(
        index="merge_key", columns="source", values="contrib_%", aggfunc="sum", fill_value=0.0
    ).reset_index()

    final = grouped.merge(pivot, on="merge_key")

    # Clean up temporary column
    if "_ticker_sort" in final.columns:
        final = final.drop(columns=["_ticker_sort"])

    return final


def enrich_holdings(
    df: pd.DataFrame, etf_value: float, source_ticker: str
) -> pd.DataFrame:
    """
    Convenience function: calculate values, add source and merge_key columns.

    Combines calculate_holding_values with adding the source ticker and
    normalised merge key — the standard enrichment before merging.

    Parameters
    ----------
    df : pd.DataFrame
        Raw holdings DataFrame with columns: ticker, name, weight.
    etf_value : float
        Total value of the ETF position in target currency.
    source_ticker : str
        The ETF ticker this holdings data came from.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with columns: ticker, name, weight, value, source, merge_key.
    """
    result = calculate_holding_values(df, etf_value)
    result["source"] = source_ticker
    result["merge_key"] = result["name"].apply(normalise_name)
    return result
