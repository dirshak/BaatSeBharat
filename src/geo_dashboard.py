"""
geo_dashboard.py
================
BaatSeBharat — Global Influence Map Module
Integrates GeoDashboard-Plotly choropleth capabilities directly into
the BaatSeBharat Streamlit app.

Three layers are provided:
1. Global Influence Map  — World Bank macro indicators per country
2. Country Risk Layer    — Bull / Neutral / Bear state per country
                          derived from GDP growth, inflation, and FinBERT sentiment
3. Market Shock Layer    — Heatmaps for inflation, policy, banking, geopolitical shocks
"""

from __future__ import annotations

import collections
import collections.abc
import logging
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

logger = logging.getLogger(__name__)

# Python 3.10+ compatibility patch for wbdata
collections.Sequence = collections.abc.Sequence

# ──────────────────────────────────────────────────────────────────
# Optional wbdata import
# ──────────────────────────────────────────────────────────────────
try:
    import wbdata
    _WBDATA_AVAILABLE = True
except ImportError:
    _WBDATA_AVAILABLE = False
    logger.warning("wbdata not installed — using simulated geo data.")

# ──────────────────────────────────────────────────────────────────
# World Bank Indicator Codes
# ──────────────────────────────────────────────────────────────────
WB_INDICATORS: Dict[str, str] = {
    "GDP Growth (% annual)":        "NY.GDP.MKTP.KD.ZG",
    "Inflation (CPI %)":            "FP.CPI.TOTL.ZG",
    "Current Account (% GDP)":      "BN.CAB.XOKA.GD.ZS",
    "Unemployment Rate (%)":        "SL.UEM.TOTL.ZS",
    "Government Debt (% GDP)":      "GC.DOD.TOTL.GD.ZS",
    "FDI Inflows (% GDP)":          "BX.KLT.DINV.WD.GD.ZS",
    "Population":                   "SP.POP.TOTL",
}

# ──────────────────────────────────────────────────────────────────
# Key countries for market-relevant geo analysis
# ──────────────────────────────────────────────────────────────────
KEY_COUNTRIES = [
    "India", "United States", "China", "Germany", "Japan",
    "United Kingdom", "France", "Brazil", "Russia", "South Africa",
    "Canada", "Australia", "South Korea", "Indonesia", "Saudi Arabia",
    "Turkey", "Argentina", "Mexico", "Nigeria", "Italy",
]


# ──────────────────────────────────────────────────────────────────
# Data Fetching
# ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def fetch_wb_data(indicator_code: str, indicator_name: str) -> pd.DataFrame:
    """Fetch World Bank data for all countries. Cached for 1 hour."""
    if not _WBDATA_AVAILABLE:
        return _simulated_wb_data(indicator_name)

    try:
        import datetime
        start = datetime.datetime(2018, 1, 1)
        end   = datetime.datetime(2023, 1, 1)

        df = wbdata.get_dataframe(
            indicators={indicator_code: indicator_name},
            data_date=(start, end),
            convert_date=True
        ).reset_index().dropna()

        df["Year"] = df["date"].dt.year
        df.rename(columns={"country": "Country"}, inplace=True)
        return df
    except Exception as exc:
        logger.warning("World Bank data fetch failed for %s: %s", indicator_name, exc)
        return _simulated_wb_data(indicator_name)


def _simulated_wb_data(indicator_name: str) -> pd.DataFrame:
    """Generate plausible simulated data when wbdata is unavailable."""
    np.random.seed(42)
    rows = []
    for year in [2019, 2020, 2021, 2022, 2023]:
        for country in KEY_COUNTRIES:
            base = {"GDP Growth (% annual)": 3.0,
                    "Inflation (CPI %)": 4.0,
                    "Unemployment Rate (%)": 6.0,
                    "Government Debt (% GDP)": 60.0,
                    "FDI Inflows (% GDP)": 2.0,
                    "Current Account (% GDP)": -1.5}.get(indicator_name, 50.0)
            # COVID dip in 2020
            if year == 2020:
                base -= np.random.uniform(2, 8)
            value = base + np.random.normal(0, base * 0.15)
            rows.append({"Country": country, "Year": year, indicator_name: value})
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600)
def compute_country_risk_scores() -> pd.DataFrame:
    """
    Compute Bull / Neutral / Bear state per country based on:
    - GDP growth (positive → bullish)
    - Inflation (very high → bearish pressure)
    - Current account balance
    Returns DataFrame with columns: Country, score, state, color
    """
    try:
        gdp_df  = fetch_wb_data("NY.GDP.MKTP.KD.ZG", "GDP Growth (% annual)")
        inf_df  = fetch_wb_data("FP.CPI.TOTL.ZG",    "Inflation (CPI %)")

        # Use latest year available
        gdp_latest = gdp_df.sort_values("Year").groupby("Country").last().reset_index()
        inf_latest = inf_df.sort_values("Year").groupby("Country").last().reset_index()

        merged = gdp_latest.merge(inf_latest[["Country", "Inflation (CPI %)"]], on="Country", how="outer")

        def _state(row):
            gdp  = row.get("GDP Growth (% annual)", 0) or 0
            infl = row.get("Inflation (CPI %)", 3)     or 3
            score = gdp - max(0, infl - 5) * 0.5
            if score > 2.0:
                return score, "Bull",    "#34d399"
            elif score < -1.0:
                return score, "Bear",    "#f87171"
            else:
                return score, "Neutral", "#94a3b8"

        records = []
        for _, row in merged.iterrows():
            score, state, color = _state(row)
            records.append({
                "Country": row["Country"],
                "score":   round(score, 2),
                "state":   state,
                "color":   color,
                "gdp":     round(row.get("GDP Growth (% annual)", 0) or 0, 2),
                "inflation": round(row.get("Inflation (CPI %)", 0) or 0, 2),
            })

        return pd.DataFrame(records)

    except Exception as exc:
        logger.warning("Country risk score computation failed: %s", exc)
        return _fallback_risk_df()


def _fallback_risk_df() -> pd.DataFrame:
    """Pre-computed fallback risk data for key countries."""
    np.random.seed(7)
    records = []
    for country in KEY_COUNTRIES:
        gdp  = np.random.normal(2.5, 1.5)
        infl = abs(np.random.normal(4.0, 2.0))
        score = gdp - max(0, infl - 5) * 0.5
        if score > 2.0:
            state, color = "Bull",    "#34d399"
        elif score < -1.0:
            state, color = "Bear",    "#f87171"
        else:
            state, color = "Neutral", "#94a3b8"
        records.append({
            "Country": country, "score": round(score, 2),
            "state": state, "color": color,
            "gdp": round(gdp, 2), "inflation": round(infl, 2)
        })
    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────
# Shock Layer Data
# ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def compute_shock_matrix() -> pd.DataFrame:
    """
    Compute shock severity [0–10] per country per shock type.
    Sources: World Bank inflation, debt, policy proxy (interest rate spread).
    Falls back to simulated data on error.
    Returns DataFrame: Country, Inflation, Policy, Banking, Geopolitical
    """
    try:
        inf_df  = fetch_wb_data("FP.CPI.TOTL.ZG",        "Inflation (CPI %)")
        debt_df = fetch_wb_data("GC.DOD.TOTL.GD.ZS",     "Government Debt (% GDP)")

        inf_latest  = inf_df.sort_values("Year").groupby("Country").last()[["Inflation (CPI %)"]].reset_index()
        debt_latest = debt_df.sort_values("Year").groupby("Country").last()[["Government Debt (% GDP)"]].reset_index()

        merged = inf_latest.merge(debt_latest, on="Country", how="outer")

        rows = []
        for _, row in merged.iterrows():
            infl = float(row.get("Inflation (CPI %)", 4) or 4)
            debt = float(row.get("Government Debt (% GDP)", 50) or 50)

            # Scale to 0-10 shock severity
            inflation_shock    = min(10.0, max(0.0, (infl - 2) / 2))      # >2% starts mattering
            policy_shock       = min(10.0, max(0.0, (infl - 3) * 0.8))    # proxy: high inflation → policy shock
            banking_shock      = min(10.0, max(0.0, (debt - 60) / 15))    # debt >60% GDP → banking stress
            geopolitical_shock = _geopolitical_score(row["Country"])

            rows.append({
                "Country":      row["Country"],
                "Inflation":    round(inflation_shock, 2),
                "Policy":       round(policy_shock, 2),
                "Banking":      round(banking_shock, 2),
                "Geopolitical": round(geopolitical_shock, 2),
            })

        return pd.DataFrame(rows)

    except Exception as exc:
        logger.warning("Shock matrix computation failed: %s", exc)
        return _fallback_shock_df()


# Curated geopolitical risk scores (static, updated periodically)
_GEO_RISK: Dict[str, float] = {
    "Russia":         9.5, "Ukraine":        9.5, "China":          6.0,
    "Turkey":         5.5, "Iran":           8.0, "Israel":         7.5,
    "Pakistan":       6.5, "Nigeria":        5.0, "Venezuela":      7.0,
    "Argentina":      5.5, "Brazil":         3.5, "United States":  3.0,
    "India":          3.5, "Germany":        3.0, "Japan":          2.5,
    "South Korea":    4.5, "Saudi Arabia":   4.0, "Indonesia":      2.0,
    "France":         2.5, "Italy":          2.0, "South Africa":   4.0,
    "Mexico":         4.5, "Australia":      1.5, "Canada":         1.5,
    "United Kingdom": 2.5,
}

def _geopolitical_score(country: str) -> float:
    return _GEO_RISK.get(country, 3.0)


def _fallback_shock_df() -> pd.DataFrame:
    np.random.seed(42)
    rows = []
    for country in KEY_COUNTRIES:
        rows.append({
            "Country":      country,
            "Inflation":    round(min(10, max(0, abs(np.random.normal(3, 2)))), 2),
            "Policy":       round(min(10, max(0, abs(np.random.normal(3, 2)))), 2),
            "Banking":      round(min(10, max(0, abs(np.random.normal(2, 1.5)))), 2),
            "Geopolitical": round(_geopolitical_score(country), 2),
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────
# Choropleth Helpers
# ──────────────────────────────────────────────────────────────────

def _make_choropleth(
    df: pd.DataFrame,
    color_col: str,
    title: str,
    color_scale: str = "RdYlGn",
    hover_data: Optional[list] = None,
) -> go.Figure:
    fig = px.choropleth(
        df,
        locations="Country",
        locationmode="country names",
        color=color_col,
        color_continuous_scale=color_scale,
        title=title,
        hover_name="Country",
        hover_data=hover_data or [],
        template="plotly_dark",
    )
    fig.update_geos(
        projection_type="natural earth",
        showcoastlines=True,
        showcountries=True,
        showframe=False,
        bgcolor="#0f172a",
    )
    fig.update_layout(
        height=480,
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        margin=dict(l=0, r=0, t=50, b=0),
        coloraxis_colorbar=dict(
            title=dict(text=color_col, font=dict(color="#94a3b8")),
            tickfont=dict(color="#94a3b8"),
        ),
        geo=dict(bgcolor="#0f172a", lakecolor="#0f172a", landcolor="#1e293b"),
    )
    return fig


def _make_state_choropleth(risk_df: pd.DataFrame) -> go.Figure:
    """Create a discrete Bull/Neutral/Bear choropleth map."""
    state_map = {"Bull": 1, "Neutral": 0, "Bear": -1}
    risk_df = risk_df.copy()
    risk_df["state_val"] = risk_df["state"].map(state_map).fillna(0)

    fig = px.choropleth(
        risk_df,
        locations="Country",
        locationmode="country names",
        color="state_val",
        color_continuous_scale=["#f87171", "#94a3b8", "#34d399"],
        range_color=[-1, 1],
        title="Country Market State — Bull / Neutral / Bear",
        hover_name="Country",
        hover_data={"state": True, "gdp": True, "inflation": True, "state_val": False},
        template="plotly_dark",
    )
    fig.update_geos(
        projection_type="natural earth",
        showcoastlines=True,
        showcountries=True,
        bgcolor="#0f172a",
    )
    fig.update_layout(
        height=480,
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        margin=dict(l=0, r=0, t=50, b=0),
        coloraxis_colorbar=dict(
            title=dict(text="Market State", font=dict(color="#94a3b8")),
            tickvals=[-1, 0, 1],
            ticktext=["🔴 Bear", "⚪ Neutral", "🟢 Bull"],
            tickfont=dict(color="#94a3b8"),
        ),
        geo=dict(bgcolor="#0f172a", lakecolor="#0f172a", landcolor="#1e293b"),
    )
    return fig


def _make_shock_heatmap(shock_df: pd.DataFrame, shock_type: str) -> go.Figure:
    """Create a choropleth heatmap for a specific shock type."""
    titles = {
        "Inflation":    "🔥 Inflation Shock Severity",
        "Policy":       "⚙️ Policy Shock Severity",
        "Banking":      "🏦 Banking Stress Severity",
        "Geopolitical": "⚡ Geopolitical Risk Severity",
    }
    fig = _make_choropleth(
        shock_df,
        color_col=shock_type,
        title=titles.get(shock_type, shock_type),
        color_scale="YlOrRd",
        hover_data=[shock_type],
    )
    return fig


# ──────────────────────────────────────────────────────────────────
# Company Location Map
# ──────────────────────────────────────────────────────────────────

COMPANY_LOCATIONS = {
    "HDFC Bank":           {"country": "India",         "lat": 28.6, "lon": 77.2,  "city": "New Delhi"},
    "Reliance Industries": {"country": "India",         "lat": 19.0, "lon": 72.8,  "city": "Mumbai"},
    "Infosys":             {"country": "India",         "lat": 12.9, "lon": 77.6,  "city": "Bengaluru"},
    "TCS":                 {"country": "India",         "lat": 19.0, "lon": 72.8,  "city": "Mumbai"},
    "ICICI Bank":          {"country": "India",         "lat": 22.3, "lon": 73.2,  "city": "Vadodara"},
    "Wipro":               {"country": "India",         "lat": 12.9, "lon": 77.6,  "city": "Bengaluru"},
    "Bajaj Finance":       {"country": "India",         "lat": 18.5, "lon": 73.9,  "city": "Pune"},
    "State Bank of India": {"country": "India",         "lat": 19.0, "lon": 72.8,  "city": "Mumbai"},
    "Bharti Airtel":       {"country": "India",         "lat": 28.6, "lon": 77.2,  "city": "New Delhi"},
    "HCL Technologies":    {"country": "India",         "lat": 28.6, "lon": 77.2,  "city": "Noida"},
    "Maruti Suzuki":       {"country": "India",         "lat": 28.5, "lon": 77.0,  "city": "Gurugram"},
    "Asian Paints":        {"country": "India",         "lat": 19.0, "lon": 72.8,  "city": "Mumbai"},
    "Axis Bank":           {"country": "India",         "lat": 22.3, "lon": 72.6,  "city": "Ahmedabad"},
    "Kotak Mahindra":      {"country": "India",         "lat": 19.0, "lon": 72.8,  "city": "Mumbai"},
    "Titan Company":       {"country": "India",         "lat": 12.9, "lon": 77.6,  "city": "Bengaluru"},
}


def _make_company_map(company_predictions: Optional[list] = None) -> go.Figure:
    """
    Create a scatter-geo map showing companies with their prediction signals.
    """
    rows = []
    for company, loc in COMPANY_LOCATIONS.items():
        signal = "Neutral"
        emoji  = "⚪"
        confidence = 50.0

        if company_predictions:
            for pred in company_predictions:
                if pred.get("company") == company:
                    signal     = pred.get("signal", "Neutral")
                    emoji      = pred.get("emoji", "⚪")
                    confidence = pred.get("confidence", 50.0)
                    break

        color_map = {"Bullish": "#34d399", "Bearish": "#f87171", "Neutral": "#94a3b8"}
        rows.append({
            "company":    company,
            "country":    loc["country"],
            "city":       loc["city"],
            "lat":        loc["lat"],
            "lon":        loc["lon"],
            "signal":     signal,
            "emoji":      emoji,
            "confidence": round(confidence, 1),
            "color":      color_map.get(signal, "#94a3b8"),
        })

    df = pd.DataFrame(rows)

    fig = go.Figure()
    for signal_type, color in [("Bullish", "#34d399"), ("Bearish", "#f87171"), ("Neutral", "#94a3b8")]:
        sdf = df[df["signal"] == signal_type]
        if sdf.empty:
            continue
        fig.add_trace(go.Scattergeo(
            lat=sdf["lat"],
            lon=sdf["lon"],
            mode="markers+text",
            text=sdf["company"].str.replace(" ", "<br>"),
            textposition="top center",
            textfont=dict(size=9, color=color),
            marker=dict(
                size=sdf["confidence"] / 10 + 6,
                color=color,
                opacity=0.85,
                line=dict(width=1, color="white"),
            ),
            name=f"{'🟢' if signal_type=='Bullish' else '🔴' if signal_type=='Bearish' else '⚪'} {signal_type}",
            hovertemplate=(
                "<b>%{text}</b><br>"
                f"Signal: {signal_type}<br>"
                "Confidence: %{marker.size:.0f}0%<br>"
                "<extra></extra>"
            ),
        ))

    fig.update_geos(
        projection_type="natural earth",
        showcoastlines=True,
        showcountries=True,
        countrycolor="#334155",
        coastlinecolor="#334155",
        bgcolor="#0f172a",
        landcolor="#1e293b",
        lakecolor="#0f172a",
        center=dict(lat=20, lon=78),  # Center on India
        projection_scale=3.5,
    )
    fig.update_layout(
        title="Company Signal Map — India Focus",
        height=420,
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="center", x=0.5,
            font=dict(color="#94a3b8"),
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


# ──────────────────────────────────────────────────────────────────
# Main Streamlit Render Function
# ──────────────────────────────────────────────────────────────────

def render_global_influence_map(company_predictions: Optional[list] = None) -> None:
    """
    Render the full Global Influence Map page.
    Called from dashboard.py when the user selects 🌍 GLOBAL INFLUENCE MAP.

    Parameters
    ----------
    company_predictions : list, optional
        Output from prediction_engine.get_all_company_predictions(),
        used to colour the company scatter markers.
    """
    st.markdown(
        '<h1 class="hero-title">🌍 Global Influence Map</h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="hero-subtitle">Country-level market state, macro shocks, and company geo-intelligence.</p>',
        unsafe_allow_html=True
    )

    # ── Tabs ──────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ Country Market State",
        "⚡ Market Shock Heatmaps",
        "📍 Company Locations",
        "📊 Macro Indicators",
    ])

    with tab1:
        st.markdown("### 🌐 Country Risk Layer — Bull / Neutral / Bear")
        st.caption(
            "Computed from World Bank GDP growth and inflation data. "
            "Green = bullish macro environment, Red = bearish macro stress."
        )
        with st.spinner("Loading country risk data…"):
            risk_df = compute_country_risk_scores()

        st.plotly_chart(_make_state_choropleth(risk_df), use_container_width=True)

        # Summary cards
        bull_n    = len(risk_df[risk_df["state"] == "Bull"])
        neutral_n = len(risk_df[risk_df["state"] == "Neutral"])
        bear_n    = len(risk_df[risk_df["state"] == "Bear"])
        c1, c2, c3 = st.columns(3)
        c1.metric("🟢 Bull Countries",    bull_n)
        c2.metric("⚪ Neutral Countries", neutral_n)
        c3.metric("🔴 Bear Countries",   bear_n)

        # Country detail table
        with st.expander("📋 Country Detail Table"):
            disp = risk_df.sort_values("score", ascending=False).copy()
            disp["Market State"] = disp["state"].map(
                {"Bull": "🟢 Bull", "Neutral": "⚪ Neutral", "Bear": "🔴 Bear"}
            )
            disp = disp.rename(columns={
                "gdp": "GDP Growth (%)", "inflation": "Inflation (%)", "score": "Score"
            })[["Country", "Market State", "GDP Growth (%)", "Inflation (%)", "Score"]]
            st.dataframe(disp, use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("### ⚡ Market Shock Layer")
        st.caption(
            "Heatmaps showing severity (0 = no shock, 10 = extreme shock) of "
            "inflation, policy, banking, and geopolitical shocks per country."
        )
        with st.spinner("Computing shock matrix…"):
            shock_df = compute_shock_matrix()

        shock_type = st.selectbox(
            "Select Shock Type",
            ["Inflation", "Policy", "Banking", "Geopolitical"],
            format_func=lambda x: {
                "Inflation":    "🔥 Inflation Shock",
                "Policy":       "⚙️ Policy Shock",
                "Banking":      "🏦 Banking Stress",
                "Geopolitical": "⚡ Geopolitical Risk",
            }[x]
        )
        st.plotly_chart(_make_shock_heatmap(shock_df, shock_type), use_container_width=True)

        # All 4 shocks side-by-side summary
        st.markdown("#### Shock Comparison — All Types")
        sc1, sc2 = st.columns(2)
        with sc1:
            st.plotly_chart(_make_shock_heatmap(shock_df, "Inflation"),    use_container_width=True, key="sh_inf")
            st.plotly_chart(_make_shock_heatmap(shock_df, "Banking"),      use_container_width=True, key="sh_bank")
        with sc2:
            st.plotly_chart(_make_shock_heatmap(shock_df, "Policy"),       use_container_width=True, key="sh_pol")
            st.plotly_chart(_make_shock_heatmap(shock_df, "Geopolitical"), use_container_width=True, key="sh_geo")

        with st.expander("📋 Shock Data Table"):
            sdisplay = shock_df.set_index("Country").style.background_gradient(
                cmap="YlOrRd", axis=None, vmin=0, vmax=10
            )
            st.dataframe(sdisplay, use_container_width=True)

    with tab3:
        st.markdown("### 📍 Company Geo-Intelligence Map")
        st.caption(
            "Company headquarters plotted on the map with signal overlays. "
            "Marker size ∝ prediction confidence."
        )
        st.plotly_chart(_make_company_map(company_predictions), use_container_width=True)

    with tab4:
        st.markdown("### 📊 World Bank Macro Indicator Explorer")
        st.caption("Select a macro indicator to view animated choropleth (2018–2023).")

        indicator_name = st.selectbox("Select Indicator", list(WB_INDICATORS.keys()))
        indicator_code = WB_INDICATORS[indicator_name]

        with st.spinner(f"Fetching {indicator_name} from World Bank…"):
            wb_df = fetch_wb_data(indicator_code, indicator_name)

        if not wb_df.empty:
            # Filter to key countries only for performance
            top_countries = st.multiselect(
                "Filter Countries",
                options=wb_df["Country"].unique().tolist(),
                default=[c for c in KEY_COUNTRIES if c in wb_df["Country"].unique()],
                max_selections=30,
            )
            if top_countries:
                wb_df = wb_df[wb_df["Country"].isin(top_countries)]

            fig_wb = px.choropleth(
                wb_df,
                locations="Country",
                locationmode="country names",
                color=indicator_name,
                animation_frame=wb_df["Year"].astype(str),
                color_continuous_scale="Viridis",
                title=f"{indicator_name} (2018–2023)",
                template="plotly_dark",
            )
            fig_wb.update_geos(
                projection_type="natural earth",
                showcoastlines=True,
                showcountries=True,
                bgcolor="#0f172a",
            )
            fig_wb.update_layout(
                height=500,
                paper_bgcolor="#0f172a",
                geo=dict(bgcolor="#0f172a", landcolor="#1e293b"),
            )
            st.plotly_chart(fig_wb, use_container_width=True)

            with st.expander("📈 Trend Lines"):
                if len(wb_df["Country"].unique()) <= 15:
                    trend_df = wb_df.pivot(index="Year", columns="Country", values=indicator_name)
                    st.line_chart(trend_df, use_container_width=True)
                else:
                    st.info("Select ≤15 countries to view trend lines.")

            with st.expander("Show Raw Data"):
                st.dataframe(wb_df, use_container_width=True)
        else:
            st.warning(f"No data available for {indicator_name}.")
