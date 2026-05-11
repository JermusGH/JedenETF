"""
Unified ETF Portfolio Dashboard — Streamlit App

Run with:  streamlit run app.py
"""

import re
import urllib.parse

import pandas as pd
import streamlit as st

from etf_holdings import fetch_holdings
from portfolio import TARGET_CURRENCY, fetch_prices, get_fund_family

# ---------------------------------------------------------------------------
# Provider detection helpers
# ---------------------------------------------------------------------------

_SUPPORTED_PROVIDERS = {"ishares", "vanguard", "amundi"}

_GITHUB_REPO = "JermusGH/JedenETF"


def _github_issue_url(title: str, body: str = "") -> str:
    """Build a GitHub new-issue URL with pre-filled title and body."""
    params = urllib.parse.urlencode({"title": title, "body": body})
    return f"https://github.com/{_GITHUB_REPO}/issues/new?{params}"


def _detect_ticker_provider(ticker: str) -> str:
    """Detect the provider for a ticker. Returns 'ishares', 'vanguard', 'amundi', or 'unknown'."""
    try:
        family = get_fund_family(ticker).lower()
        if "blackrock" in family or "ishares" in family:
            return "ishares"
        if "vanguard" in family:
            return "vanguard"
        if "amundi" in family:
            return "amundi"
    except Exception:
        pass
    return "unknown"


def _validate_holdings_csv(file) -> tuple[pd.DataFrame | None, str]:
    """
    Validate an uploaded CSV file for custom holdings.

    Expected format: CSV with columns 'ticker', 'name', 'weight'.
    - ticker: stock ticker (string)
    - name: company name (string)
    - weight: percentage weight (numeric, e.g. 8.5 means 8.5%)

    Returns (DataFrame, "") on success or (None, error_message) on failure.
    """
    try:
        df = pd.read_csv(file)
    except Exception as exc:
        return None, f"Could not read CSV file: {exc}"

    required_cols = {"ticker", "name", "weight"}
    actual_cols = {c.strip().lower() for c in df.columns}
    missing = required_cols - actual_cols
    if missing:
        return None, f"Missing required columns: {', '.join(sorted(missing))}. Expected: ticker, name, weight"

    # Normalise column names
    df.columns = [c.strip().lower() for c in df.columns]
    df = df[["ticker", "name", "weight"]].copy()

    # Validate data types
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["name"] = df["name"].astype(str).str.strip()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")

    if df["weight"].isna().any():
        bad_rows = df[df["weight"].isna()].index.tolist()
        return None, f"Non-numeric values in 'weight' column at rows: {bad_rows}"

    if (df["weight"] < 0).any():
        return None, "Negative values found in 'weight' column."

    if df["name"].eq("").any():
        return None, "Empty values found in 'name' column."

    df = df[df["weight"] > 0].reset_index(drop=True)
    if df.empty:
        return None, "No holdings with weight > 0 found in the file."

    return df, ""

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Unified ETF Portfolio", layout="wide")

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

        # Use custom uploaded holdings if available, otherwise fetch
        if ticker in st.session_state.custom_holdings:
            df = st.session_state.custom_holdings[ticker].copy()
        else:
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

if "amundi_tickers" not in st.session_state:
    st.session_state.amundi_tickers = set()

if "unsupported_tickers" not in st.session_state:
    st.session_state.unsupported_tickers = set()

if "custom_holdings" not in st.session_state:
    st.session_state.custom_holdings = {}  # ticker -> pd.DataFrame


# ---------------------------------------------------------------------------
# Welcome screen / Portfolio editor
# ---------------------------------------------------------------------------

def show_portfolio_editor():
    st.title("Unified ETF Portfolio")
    st.markdown(
        "See your true exposure across multiple ETFs. "
        "Add your holdings below, then click **Analyse** to merge them."
    )

    st.markdown("---")
    with st.expander("How to use"):
        st.markdown(
            "**This tool supports European-listed ETFs only** (e.g. London, Frankfurt, Amsterdam).\n\n"
            "1. Enter the ETF ticker from Yahoo Finance including the exchange suffix "
            "(e.g. `CSPX.L` for London, `VWCE.DE` for Frankfurt, `CNDX.AS` for Amsterdam).\n"
            "2. Specify the number of shares you hold.\n"
            "3. Click **Add** to add the position to your portfolio.\n"
            "4. Once all ETFs are added, click **Analyse Portfolio** to see "
            "your true exposure across individual companies.\n\n"
            "**Note:** The exchange suffix (`.L`, `.DE`, `.AS`, etc.) is required. "
            "Without it, the tool cannot identify the correct ETF listing.\n\n"
            "**Unsupported providers:** If your ETF provider is not natively supported "
            "(iShares, Vanguard, Amundi), you can upload a custom holdings CSV file. "
            "The file must contain columns: `ticker`, `name`, `weight` "
            "(where weight is a percentage, e.g. 8.5 means 8.5%)."
        )

    # Warnings section
    warnings = []
    for t in st.session_state.amundi_tickers:
        if t in st.session_state.portfolio:
            warnings.append(
                f"**{t}** — Amundi ETF: only top 10 holdings will be used in the analysis. "
                f"You can upload a full holdings CSV using the upload button next to the ticker."
            )
    for t in st.session_state.unsupported_tickers:
        if t in st.session_state.portfolio:
            issue_url = _github_issue_url(
                title=f"Add support for provider: {t}",
                body=f"Ticker: {t}\nProvider: unknown\n\nPlease add full holdings support for this ETF provider.",
            )
            warnings.append(
                f"**{t}** — unsupported provider. Only top 10 holdings will be used. "
                f"You can upload a full holdings CSV using the upload button next to the ticker. "
                f"[Request provider support]({issue_url})"
            )

    if warnings:
        st.markdown("---")
        st.subheader("Warnings")
        for w in warnings:
            st.warning(w)

    st.markdown("---")
    st.subheader("Your ETFs")

    # Column headers
    col_h1, col_h2, col_h3, col_h4, col_h5 = st.columns([3, 2, 0.3, 0.3, 0.3])
    col_h1.markdown("**Yahoo Finance ticker**")
    col_h2.markdown("**Number of shares**")

    # Show existing entries
    to_remove = []
    for ticker, units in st.session_state.portfolio.items():
        col1, col2, col3, col4, col5 = st.columns([3, 2, 0.3, 0.3, 0.3])
        col1.text_input("Ticker", value=ticker, disabled=True, key=f"disp_{ticker}", label_visibility="collapsed")
        new_val = col2.number_input(
            "Units", value=float(units), min_value=0.0, step=0.1,
            key=f"val_{ticker}", label_visibility="collapsed"
        )
        st.session_state.portfolio[ticker] = new_val

        if col3.button("✕", key=f"rm_{ticker}"):
            to_remove.append(ticker)

        # Upload button for unsupported/limited providers
        if ticker in st.session_state.unsupported_tickers or ticker in st.session_state.amundi_tickers:
            with col4.popover("↑", use_container_width=True):
                st.markdown("**Upload holdings CSV**")
                st.caption("Columns: `ticker`, `name`, `weight`")
                uploaded = st.file_uploader(
                    "CSV file",
                    type=["csv"],
                    key=f"upload_{ticker}",
                    label_visibility="collapsed",
                )
                if uploaded is not None and ticker not in st.session_state.custom_holdings:
                    df, error = _validate_holdings_csv(uploaded)
                    if error:
                        st.error(error)
                    else:
                        st.session_state.custom_holdings[ticker] = df
                        st.rerun()

            # Green tick if data loaded
            if ticker in st.session_state.custom_holdings:
                col5.markdown(
                    "<span style='color: green; font-size: 1.5rem;'>&#10004;</span>",
                    unsafe_allow_html=True,
                )

    for t in to_remove:
        del st.session_state.portfolio[t]
        st.session_state.amundi_tickers.discard(t)
        st.session_state.unsupported_tickers.discard(t)
        st.session_state.custom_holdings.pop(t, None)
        st.rerun()

    # Input row at the bottom — moves down as entries are added
    col_a, col_b, col_c, _, _ = st.columns([3, 2, 0.3, 0.3, 0.3])
    new_ticker = col_a.text_input("Ticker input", placeholder="e.g. CSPX.L, VWCE.DE", key="new_ticker", label_visibility="collapsed")
    new_units = col_b.number_input("Units input", value=0.0, min_value=0.0, step=0.1, key="new_units", label_visibility="collapsed")
    if col_c.button("Add", type="primary", key="add_btn"):
        if new_ticker and new_units > 0:
            clean_ticker = new_ticker.strip().upper()
            if "." not in clean_ticker:
                st.warning(
                    "Ticker is missing an exchange suffix (e.g. `.L`, `.DE`, `.AS`). "
                    "Please use the full Yahoo Finance ticker format."
                )
            else:
                st.session_state.portfolio[clean_ticker] = new_units
                provider = _detect_ticker_provider(clean_ticker)
                if provider == "amundi":
                    st.session_state.amundi_tickers.add(clean_ticker)
                elif provider not in _SUPPORTED_PROVIDERS:
                    st.session_state.unsupported_tickers.add(clean_ticker)
                st.rerun()
        else:
            st.warning("Enter a ticker and units > 0")

    # Analyse button
    st.markdown("---")
    active = {k: v for k, v in st.session_state.portfolio.items() if v > 0}

    if active:
        st.info(f"**{len(active)} ETFs** ready for analysis")
        if st.button("Analyse Portfolio", type="primary", use_container_width=True):
            with st.spinner("Fetching data..."):
                final_df, total_value, sources = run_analysis(active)
            st.session_state.results = (final_df, total_value, sources)
            st.rerun()
    else:
        st.caption("Add at least one ETF to get started.")

    # Suggest Improvement section
    st.markdown("---")
    st.subheader("Suggest Improvement")
    st.markdown("Have an idea or want support for a new ETF provider? Open a GitHub issue:")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        provider_url = _github_issue_url(
            title="Add support for new provider",
            body="Provider name: \nExample ticker: \n\nPlease add support for this ETF provider.",
        )
        st.link_button("Request new provider", provider_url)
    with col_s2:
        suggestion_url = _github_issue_url(
            title="Suggestion: ",
            body="Describe your suggestion here:\n\n",
        )
        st.link_button("Suggest a change", suggestion_url)


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
    st.title("Portfolio Analysis")
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
