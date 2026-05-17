"""
Unified ETF Portfolio Dashboard — Streamlit App

Run with:  streamlit run app.py
"""

import logging
import urllib.parse

import altair as alt
import pandas as pd
import plotly.express as px
import streamlit as st

from analysis import enrich_holdings, merge_holdings
from etf_holdings import HoldingsFetcher, detect_provider
from holding_metadata import fetch_holdings_metadata
from models import AnalysisResult
from portfolio import SUPPORTED_CURRENCIES, TARGET_CURRENCY
from pricing import fetch_prices, get_fund_family
from validation import validate_holdings_csv, validate_ticker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider detection helpers
# ---------------------------------------------------------------------------

_SUPPORTED_PROVIDERS = {"ishares", "vanguard", "invesco", "xtrackers", "amundi"}

_GITHUB_REPO = "JermusGH/JedenETF"


def _github_issue_url(title: str, body: str = "") -> str:
    """Build a GitHub new-issue URL with pre-filled title and body."""
    params = urllib.parse.urlencode({"title": title, "body": body})
    return f"https://github.com/{_GITHUB_REPO}/issues/new?{params}"


def _detect_ticker_provider(ticker: str) -> str:
    """Detect the provider for a ticker using Yahoo Finance fund family metadata."""
    try:
        family = get_fund_family(ticker)
        return detect_provider(family)
    except Exception:
        return "unknown"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Unified ETF Portfolio", layout="wide")

# ---------------------------------------------------------------------------
# Analysis logic
# ---------------------------------------------------------------------------

def run_analysis(portfolio: dict[str, float], target_currency: str = "PLN") -> AnalysisResult:
    """Fetch prices and holdings, merge into unified DataFrame.

    Returns an AnalysisResult dataclass.
    """
    tickers = list(portfolio.keys())
    prices = fetch_prices(tickers, target_currency=target_currency)

    frames: list[pd.DataFrame] = []
    total_value = 0.0
    failed_tickers: list[str] = []

    fetcher = HoldingsFetcher()

    progress = st.progress(0, text="Loading holdings...")
    for i, (ticker, units) in enumerate(portfolio.items()):
        progress.progress((i + 1) / len(portfolio), text=f"Loading {ticker}...")

        if units <= 0:
            continue
        price = prices.get(ticker)
        if price is None:
            failed_tickers.append(ticker)
            continue

        etf_value = units * price

        # Use custom uploaded holdings if available, otherwise fetch
        if ticker in st.session_state.custom_holdings:
            df = st.session_state.custom_holdings[ticker].copy()
        else:
            fund_family = get_fund_family(ticker)
            df = fetcher.fetch(yf_ticker=ticker, fund_family=fund_family)

        if df is None or df.empty:
            failed_tickers.append(ticker)
            continue

        enriched = enrich_holdings(df, etf_value, ticker)
        total_value += etf_value
        frames.append(enriched)

    progress.empty()

    justetf_tickers = sorted(fetcher.justetf_tickers)

    if not frames or total_value == 0:
        return AnalysisResult(
            df=pd.DataFrame(),
            total_value=0.0,
            failed_tickers=failed_tickers,
            justetf_tickers=justetf_tickers,
        )

    sources = sorted(set(s for f in frames for s in f["source"].unique()))
    final = merge_holdings(frames, total_value)
    final = final.sort_values("weight_%", ascending=False)
    return AnalysisResult(
        df=final,
        total_value=total_value,
        sources=sources,
        failed_tickers=failed_tickers,
        justetf_tickers=justetf_tickers,
    )


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

if "target_currency" not in st.session_state:
    st.session_state.target_currency = "PLN"

if "_prev_target_currency" not in st.session_state:
    st.session_state._prev_target_currency = st.session_state.target_currency

# ---------------------------------------------------------------------------
# Sidebar — Currency selector
# ---------------------------------------------------------------------------

st.session_state.target_currency = st.sidebar.selectbox(
    "Target Currency",
    options=SUPPORTED_CURRENCIES,
    index=SUPPORTED_CURRENCIES.index(st.session_state.target_currency),
)

# Invalidate results when currency changes
if st.session_state.target_currency != st.session_state._prev_target_currency:
    st.session_state._prev_target_currency = st.session_state.target_currency
    if st.session_state.results is not None:
        st.session_state.results = None
        st.rerun()


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
            "---\n\n"
            "**Provider support levels:**\n\n"
            "| Level | Providers | Holdings |\n"
            "|-------|-----------|----------|\n"
            "| Full | iShares (BlackRock), Vanguard, Invesco, Xtrackers (DWS) | All holdings |\n"
            "| Limited | Amundi | Top 10 via justETF or upload CSV |\n"
            "| Unsupported | All others | Top 10 via justETF or upload CSV |\n\n"
            "For limited/unsupported providers you can upload a custom holdings CSV file "
            "with columns: `ticker`, `name`, `weight` "
            "(where weight is a percentage, e.g. 8.5 means 8.5%).\n\n"
            "---\n\n"
            "**Not supported:** Swap-based, money market, and commodity ETFs (e.g. XEON, XEOD) "
            "do not hold individual securities and cannot be analysed by this tool."
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
                f"**{t}** — unsupported provider. Only top 10 holdings will be used (via justETF). "
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
                    df, error = validate_holdings_csv(uploaded)
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
        cleaned_ticker, error = validate_ticker(new_ticker)
        if error:
            st.warning(error)
        elif new_units <= 0:
            st.warning("Please enter a number of shares greater than zero.")
        else:
            st.session_state.portfolio[cleaned_ticker] = new_units
            provider = _detect_ticker_provider(cleaned_ticker)
            if provider == "amundi":
                st.session_state.amundi_tickers.add(cleaned_ticker)
            elif provider not in _SUPPORTED_PROVIDERS:
                st.session_state.unsupported_tickers.add(cleaned_ticker)
            st.rerun()

    # Analyse button
    st.markdown("---")
    active = {k: v for k, v in st.session_state.portfolio.items() if v > 0}

    if active:
        st.info(f"**{len(active)} ETFs** ready for analysis")
        if st.button("Analyse Portfolio", type="primary", use_container_width=True):
            with st.spinner("Fetching data..."):
                result = run_analysis(active, target_currency=st.session_state.target_currency)
            st.session_state.results = result
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
    result: AnalysisResult = st.session_state.results

    if result.is_empty:
        st.error("Could not fetch holdings for any ETF. Check tickers and internet connection.")
        if result.failed_tickers:
            st.warning(
                f"**Failed tickers:** {', '.join(result.failed_tickers)}\n\n"
                "Swap-based, money market, and commodity ETFs do not hold individual "
                "securities and cannot be analysed by this tool."
            )
        if st.button("← Back to editor"):
            st.session_state.results = None
            st.rerun()
        return

    final_df = result.df
    total_value = result.total_value
    sources = result.sources
    failed_tickers = result.failed_tickers
    justetf_tickers = result.justetf_tickers

    # Header
    st.title("Portfolio Analysis")
    if st.button("← Edit Portfolio"):
        st.session_state.results = None
        st.rerun()

    # Warning for failed tickers
    if failed_tickers:
        st.warning(
            f"**Skipped:** {', '.join(failed_tickers)} — no holdings data available. "
            "Swap-based, money market, and commodity ETFs do not hold individual "
            "securities and cannot be analysed."
        )

    # Warning for justETF fallback tickers
    if justetf_tickers:
        st.info(
            f"**Top 10 only (via justETF):** {', '.join(justetf_tickers)} — "
            "full holdings are not available for these ETFs. "
            "Only the top 10 positions are included in the analysis."
        )

    # Metrics
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Value", f"{total_value:,.0f} {st.session_state.target_currency}")
    col2.metric("Unique Holdings", f"{len(final_df):,}")
    col3.metric("ETFs Loaded", f"{len(sources)}")

    # Top holdings table
    st.markdown("---")
    st.subheader("Top 20 Holdings")

    top20 = final_df.head(20).copy()
    display_df = top20[["ticker", "name", "weight_%", "value"]].copy()
    currency = st.session_state.target_currency
    display_df.columns = ["Ticker", "Company", "Weight (%)", f"Value ({currency})"]
    display_df = display_df.reset_index(drop=True)
    display_df.index += 1
    st.dataframe(
        display_df.style.format({"Weight (%)": "{:.2f}%", f"Value ({currency})": "{:,.2f}"}),
        use_container_width=True,
    )

    # Sector & Country Distribution
    st.markdown("---")

    # Resolve metadata for holdings
    holding_tickers = final_df["ticker"].tolist()
    with st.spinner("Resolving sector & country data..."):
        metadata = fetch_holdings_metadata(holding_tickers)

    # Build sector and country distribution data
    sector_rows = []
    country_rows = []
    for _, row in final_df.iterrows():
        ticker = row["ticker"]
        meta = metadata.get(ticker)
        if meta:
            sector_rows.append({"sector": meta["sector"], "weight_%": row["weight_%"], "value": row["value"]})
            country_rows.append({"country": meta["country"], "weight_%": row["weight_%"], "value": row["value"]})

    # Charts row
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Sector Distribution")
        if sector_rows:
            sector_df = pd.DataFrame(sector_rows)
            sector_agg = sector_df.groupby("sector", as_index=False).agg(
                {"weight_%": "sum", "value": "sum"}
            ).sort_values("weight_%", ascending=False)

            # Top 5 + Other for pie chart
            top5_sectors = sector_agg.head(5).copy()
            other_weight = sector_agg.iloc[5:]["weight_%"].sum()
            other_value = sector_agg.iloc[5:]["value"].sum()
            if other_weight > 0:
                other_row = pd.DataFrame([{"sector": "Other", "weight_%": other_weight, "value": other_value}])
                pie_data = pd.concat([top5_sectors, other_row], ignore_index=True)
            else:
                pie_data = top5_sectors

            pie_chart = (
                alt.Chart(pie_data)
                .mark_arc(innerRadius=50)
                .encode(
                    theta=alt.Theta("weight_%:Q", title="Weight (%)"),
                    color=alt.Color("sector:N", title="Sector"),
                    tooltip=[
                        alt.Tooltip("sector:N", title="Sector"),
                        alt.Tooltip("weight_%:Q", title="Weight (%)", format=".2f"),
                    ],
                )
                .properties(height=300)
            )
            st.altair_chart(pie_chart, use_container_width=True)

            # Table below
            sector_agg = sector_agg.reset_index(drop=True)
            sector_agg.index += 1
            currency = st.session_state.target_currency
            display_sector = sector_agg.rename(columns={
                "sector": "Sector",
                "weight_%": "Weight (%)",
                "value": f"Value ({currency})",
            })
            st.dataframe(
                display_sector.style.format({
                    "Weight (%)": "{:.2f}%",
                    f"Value ({currency})": "{:,.0f}",
                }),
                use_container_width=True,
            )
        else:
            st.caption("No sector data available.")

    with chart_col2:
        st.subheader("Country Distribution")
        if country_rows:
            country_df = pd.DataFrame(country_rows)
            country_agg = country_df.groupby("country", as_index=False).agg(
                {"weight_%": "sum", "value": "sum"}
            ).sort_values("weight_%", ascending=False)

            # Country name to ISO-3 code mapping for the choropleth
            country_to_iso3 = {
                "United States": "USA", "Japan": "JPN", "China": "CHN",
                "Canada": "CAN", "Taiwan": "TWN", "Korea (South)": "KOR",
                "United Kingdom": "GBR", "Germany": "DEU", "France": "FRA",
                "Australia": "AUS", "Switzerland": "CHE", "Sweden": "SWE",
                "Netherlands": "NLD", "Hong Kong": "HKG", "South Africa": "ZAF",
                "Italy": "ITA", "Spain": "ESP", "Singapore": "SGP",
                "Mexico": "MEX", "Indonesia": "IDN", "Israel": "ISR",
                "Denmark": "DNK", "United Arab Emirates": "ARE", "Malaysia": "MYS",
                "Finland": "FIN", "Poland": "POL", "Norway": "NOR",
                "Turkey": "TUR", "Belgium": "BEL", "Russian Federation": "RUS",
                "Thailand": "THA", "Ireland": "IRL", "Qatar": "QAT",
                "Portugal": "PRT", "Philippines": "PHL", "New Zealand": "NZL",
                "Austria": "AUT", "Chile": "CHL", "Greece": "GRC",
                "Peru": "PER", "Kuwait": "KWT", "Czech Republic": "CZE",
                "Hungary": "HUN", "Egypt": "EGY", "Brazil": "BRA",
                "India": "IND", "Colombia": "COL", "Saudi Arabia": "SAU",
            }

            map_data = country_agg.copy()
            map_data["iso3"] = map_data["country"].map(country_to_iso3)
            map_data = map_data[map_data["iso3"].notna()]

            if not map_data.empty:
                fig = px.choropleth(
                    map_data,
                    locations="iso3",
                    color="weight_%",
                    hover_name="country",
                    hover_data={"weight_%": ":.2f", "iso3": False},
                    color_continuous_scale="Blues",
                    labels={"weight_%": "Weight (%)"},
                )
                fig.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=300,
                    geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth"),
                    coloraxis_colorbar=dict(title="Weight %"),
                )
                st.plotly_chart(fig, use_container_width=True)

            # Table below
            country_agg = country_agg.reset_index(drop=True)
            country_agg.index += 1
            currency = st.session_state.target_currency
            display_country = country_agg.rename(columns={
                "country": "Country",
                "weight_%": "Weight (%)",
                "value": f"Value ({currency})",
            })
            st.dataframe(
                display_country.style.format({
                    "Weight (%)": "{:.2f}%",
                    f"Value ({currency})": "{:,.0f}",
                }),
                use_container_width=True,
            )
        else:
            st.caption("No country data available.")

    # Coverage info
    resolved_weight = sum(r["weight_%"] for r in sector_rows)
    if resolved_weight < 95:
        st.caption(
            f"Coverage: {resolved_weight:.1f}% of portfolio weight resolved. "
            f"Remaining holdings lack sector/country data."
        )

    # Per-ETF breakdown
    st.markdown("---")
    st.subheader("Per-ETF Contribution (Top 20)")
    contrib_df = top20[["name"] + sources].copy().set_index("name")
    # Add Total % column as sum of all ETF contributions (second position)
    contrib_df.insert(0, "Total %", contrib_df.sum(axis=1))
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
