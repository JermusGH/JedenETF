"""
Unified ETF Portfolio Dashboard — Streamlit App

Run with:  streamlit run app.py
"""

import re

import pandas as pd
import streamlit as st

from etf_holdings import fetch_holdings
from portfolio import TARGET_CURRENCY, fetch_prices, get_fund_family

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Unified ETF Portfolio", page_icon="📊", layout="wide")

# ---------------------------------------------------------------------------
# Company name normalisation
# ---------------------------------------------------------------------------

_STRIP_SUFFIXES = re.compile(
    r"\b(?:INC|CORP|CORPORATION|LTD|LIMITED|PLC|COMPANY|AG|SA|NV|"
    r"GROUP|HOLDINGS|HOLDING|SE|CLASS\s+[A-C]|CL\s+[A-C]|CO(?!\w))\b"
)


def _normalise_name(name) -> str:
    if pd.isna(name):
        return ""
    text = re.sub(r"[^\w\s]", "", str(name).upper())
    text = _STRIP_SUFFIXES.sub("", text)
    words = text.split()
    return " ".join(words[:2])


# ---------------------------------------------------------------------------
# Analysis logic
# ---------------------------------------------------------------------------

def run_analysis(portfolio: dict[str, float]) -> tuple[pd.DataFrame, float, list[str]]:
    """Fetch prices and holdings, merge into unified DataFrame."""
    tickers = list(portfolio.keys())
    prices = fetch_prices(tickers)

    frames: list[pd.DataFrame] = []
    total_value = 0.0

    progress = st.progress(0, text="Loading holdings...")
    for i, (ticker, units) in enumerate(portfolio.items()):
        progress.progress((i + 1) / len(portfolio), text=f"Loading {ticker}...")

        if units <= 0:
            continue
        price = prices.get(ticker)
        if price is None:
            continue

        etf_value = units * price
        fund_family = get_fund_family(ticker)
        df = fetch_holdings(yf_ticker=ticker, fund_family=fund_family)

        if df is None or df.empty:
            continue

        weight_sum = df["weight"].sum()
        if weight_sum > 10:
            df["value"] = (df["weight"] / 100.0) * etf_value
        else:
            df["value"] = df["weight"] * etf_value

        df["source"] = ticker
        df["merge_key"] = df["name"].apply(_normalise_name)
        total_value += etf_value
        frames.append(df)

    progress.empty()

    if not frames:
        return pd.DataFrame(), 0.0, []

    combined = pd.concat(frames, ignore_index=True)
    sources = sorted(combined["source"].unique().tolist())

    combined["_ticker_len"] = combined["ticker"].astype(str).str.len()
    combined = combined.sort_values("_ticker_len")

    grouped = (
        combined.groupby("merge_key")
        .agg(ticker=("ticker", "first"), name=("name", "first"), value=("value", "sum"))
        .reset_index()
    )
    grouped["weight_%"] = (grouped["value"] / total_value) * 100.0

    combined["contrib_%"] = (combined["value"] / total_value) * 100.0
    pivot = combined.pivot_table(
        index="merge_key", columns="source", values="contrib_%", aggfunc="sum", fill_value=0.0
    ).reset_index()

    final = grouped.merge(pivot, on="merge_key").sort_values("weight_%", ascending=False)
    return final, total_value, sources


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "portfolio" not in st.session_state:
    st.session_state.portfolio = {}

if "results" not in st.session_state:
    st.session_state.results = None


# ---------------------------------------------------------------------------
# Welcome screen / Portfolio editor
# ---------------------------------------------------------------------------

def show_portfolio_editor():
    st.title("� Unified ETF Portfolio")
    st.markdown(
        "See your true exposure across multiple ETFs. "
        "Add your holdings below, then click **Analyse** to merge them."
    )

    st.markdown("---")
    st.subheader("Your ETFs")

    # Show existing entries
    to_remove = []
    for ticker, units in st.session_state.portfolio.items():
        col1, col2, col3 = st.columns([3, 2, 1])
        col1.text_input("Ticker", value=ticker, disabled=True, key=f"disp_{ticker}", label_visibility="collapsed")
        new_val = col2.number_input(
            "Units", value=float(units), min_value=0.0, step=0.1,
            key=f"val_{ticker}", label_visibility="collapsed"
        )
        st.session_state.portfolio[ticker] = new_val
        if col3.button("✕", key=f"rm_{ticker}"):
            to_remove.append(ticker)

    for t in to_remove:
        del st.session_state.portfolio[t]
        st.rerun()

    # Add new ETF
    st.markdown("---")
    st.subheader("Add ETF")
    col_a, col_b, col_c = st.columns([3, 2, 1])
    new_ticker = col_a.text_input("Yahoo Finance ticker", placeholder="e.g. CSPX.L, VWCE.DE", key="new_ticker")
    new_units = col_b.number_input("Number of shares", value=0.0, min_value=0.0, step=0.1, key="new_units")
    if col_c.button("Add", type="primary"):
        if new_ticker and new_units > 0:
            st.session_state.portfolio[new_ticker.strip().upper()] = new_units
            st.rerun()
        else:
            st.warning("Enter a ticker and units > 0")

    # Analyse button
    st.markdown("---")
    active = {k: v for k, v in st.session_state.portfolio.items() if v > 0}

    if active:
        st.info(f"**{len(active)} ETFs** ready for analysis")
        if st.button("🚀 Analyse Portfolio", type="primary", use_container_width=True):
            with st.spinner("Fetching data..."):
                final_df, total_value, sources = run_analysis(active)
            st.session_state.results = (final_df, total_value, sources)
            st.rerun()
    else:
        st.caption("Add at least one ETF to get started.")


# ---------------------------------------------------------------------------
# Results screen
# ---------------------------------------------------------------------------

def show_results():
    final_df, total_value, sources = st.session_state.results

    if final_df.empty:
        st.error("Could not fetch holdings for any ETF. Check tickers and internet connection.")
        if st.button("← Back to editor"):
            st.session_state.results = None
            st.rerun()
        return

    # Header
    st.title("📊 Portfolio Analysis")
    if st.button("← Edit Portfolio"):
        st.session_state.results = None
        st.rerun()

    # Metrics
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Value", f"{total_value:,.0f} {TARGET_CURRENCY}")
    col2.metric("Unique Holdings", f"{len(final_df):,}")
    col3.metric("ETFs Loaded", f"{len(sources)}")

    # Top holdings table
    st.markdown("---")
    st.subheader("Top 20 Holdings")

    top20 = final_df.head(20).copy()
    display_df = top20[["ticker", "name", "weight_%", "value"]].copy()
    display_df.columns = ["Ticker", "Company", "Weight (%)", f"Value ({TARGET_CURRENCY})"]
    display_df = display_df.reset_index(drop=True)
    display_df.index += 1
    st.dataframe(
        display_df.style.format({"Weight (%)": "{:.2f}%", f"Value ({TARGET_CURRENCY})": "{:,.2f}"}),
        use_container_width=True,
    )

    # Charts
    st.markdown("---")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Weight Distribution (Top 15)")
        chart_data = final_df.head(15)[["name", "weight_%"]].set_index("name")
        st.bar_chart(chart_data, horizontal=True)

    with chart_col2:
        st.subheader("ETF Contribution")
        if sources:
            source_totals = final_df[sources].sum().reset_index()
            source_totals.columns = ["ETF", "Weight (%)"]
            source_totals = source_totals.set_index("ETF")
            st.bar_chart(source_totals)

    # Per-ETF breakdown
    st.markdown("---")
    st.subheader("Per-ETF Contribution (Top 20)")
    contrib_df = top20[["name"] + sources].copy().set_index("name")
    contrib_display = contrib_df.map(lambda x: f"{x:.2f}%" if x > 0.01 else "—")
    st.dataframe(contrib_display, use_container_width=True)

    # Full data
    with st.expander(f"View all {len(final_df)} holdings"):
        st.dataframe(
            final_df[["ticker", "name", "weight_%", "value"]].style.format(
                {"weight_%": "{:.3f}%", "value": "{:,.2f}"}
            ),
            use_container_width=True,
            height=600,
        )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if st.session_state.results is not None:
    show_results()
else:
    show_portfolio_editor()
