# dashboard.py
# ==========================================================
# BAATSEBHARAT | LRDR-MRPRA DASHBOARD
# LDA + NMF + FinBERT VERSION  — cache-backed fast edition
# + TradingAgents AI Predictions (Phase 2)
# + GeoDashboard Global Influence Map (Phase 3)
# ==========================================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import sqlite3
import sys
import numpy as np

DB_PATH = "./data/market_rhetoric.db"

# ── Path setup for integrated modules ─────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_BASE_DIR, "src"))

# ── Import new integration modules (non-fatal) ────────────
try:
    from prediction_engine import (
        get_all_company_predictions,
        get_all_sector_predictions,
        get_company_prediction,
        get_sector_prediction,
        COMPANY_UNIVERSE,
        SECTOR_COMPANIES,
        _llm_mode_available,
    )
    _PREDICTION_ENGINE_OK = True
except ImportError as _e:
    _PREDICTION_ENGINE_OK = False
    _pe_error = str(_e)

try:
    from geo_dashboard import render_global_influence_map
    _GEO_DASHBOARD_OK = True
except ImportError as _e:
    _GEO_DASHBOARD_OK = False
    _geo_error = str(_e)

try:
    from prediction_history import compute_prediction_vs_actual, summarize as summarize_prediction_history
    _PREDICTION_HISTORY_OK = True
except ImportError as _e:
    _PREDICTION_HISTORY_OK = False
    _pred_history_error = str(_e)

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="BaatSeBharat | Analytics",
    page_icon="📈",
    layout="wide"
)

# ==========================================================
# PATHS
# ==========================================================

DATA_DIR  = "content"
CACHE_DIR = os.path.join(DATA_DIR, "cache")

# Source files (only used if cache is missing)
TOPIC_DATASET = os.path.join(DATA_DIR, "topic_dataset.csv")
FINAL_TOPICS  = os.path.join(DATA_DIR, "final_topics.csv")
NMF_TOPICS    = os.path.join(DATA_DIR, "nmf_topics.csv")

# Pre-computed cache files
C_SECTOR_WEEKLY    = os.path.join(CACHE_DIR, "sector_weekly.csv")
C_SECTOR_QUARTERLY = os.path.join(CACHE_DIR, "sector_quarterly.csv")
C_REGIME_WEEKLY    = os.path.join(CACHE_DIR, "regime_weekly.csv")
C_REGIME_QUARTERLY = os.path.join(CACHE_DIR, "regime_quarterly.csv")
C_SECTOR_AVG       = os.path.join(CACHE_DIR, "sector_avg.csv")
C_SENTIMENT        = os.path.join(CACHE_DIR, "sentiment_timeline.csv")
C_TOPICS           = os.path.join(CACHE_DIR, "topics_enriched.csv")
C_FORECAST_Q       = os.path.join(CACHE_DIR, "forecast_quarterly.csv")
C_FORECAST_W       = os.path.join(CACHE_DIR, "forecast_weekly.csv")

# ==========================================================
# CACHE CHECK
# ==========================================================

cache_files = [
    C_SECTOR_WEEKLY, C_SECTOR_QUARTERLY,
    C_REGIME_WEEKLY, C_REGIME_QUARTERLY,
    C_SECTOR_AVG,
    C_FORECAST_Q, C_FORECAST_W,
]

if not all(os.path.exists(f) for f in cache_files):
    st.error(
        "⚠️ Cache files missing. Run once from your terminal:\n\n"
        "```\npython precompute_cache.py\n```"
    )
    st.stop()

# ==========================================================
# THEME
# ==========================================================

st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #0f172a;
    color: #f1f5f9;
}
.hero-title {
    color: #38bdf8;
    font-weight: 700;
    font-size: 2.4rem;
}
.hero-subtitle {
    color: #94a3b8;
    font-size: 1.05rem;
}
.stCard {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 15px;
}
section[data-testid="stSidebar"] {
    background-color: #1e293b;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# DATA LOADING  (cached — tiny files, instant reads)
# ==========================================================

SECTOR_COLORS = {
    "Banking":      "#38bdf8",
    "IT":           "#a78bfa",
    "Pharma":       "#34d399",
    "Auto":         "#fb923c",
    "Energy":       "#fbbf24",
    "Broad Market": "#f472b6",
}
REGIME_THRESH = 0.005


@st.cache_data
def load_sector_weekly():
    return pd.read_csv(C_SECTOR_WEEKLY)


@st.cache_data
def load_sector_quarterly():
    return pd.read_csv(C_SECTOR_QUARTERLY)


@st.cache_data
def load_regime_weekly():
    return pd.read_csv(C_REGIME_WEEKLY)


@st.cache_data
def load_regime_quarterly():
    return pd.read_csv(C_REGIME_QUARTERLY)


@st.cache_data
def load_sector_avg():
    return pd.read_csv(C_SECTOR_AVG)


@st.cache_data
def load_sentiment_timeline():
    if os.path.exists(C_SENTIMENT):
        return pd.read_csv(C_SENTIMENT, parse_dates=["date"])
    return pd.DataFrame()


@st.cache_data
def load_topics():
    if os.path.exists(C_TOPICS):
        return pd.read_csv(C_TOPICS)
    return pd.DataFrame()


@st.cache_data
def load_nmf_topics():
    if os.path.exists(NMF_TOPICS):
        return pd.read_csv(NMF_TOPICS)
    return pd.DataFrame()


@st.cache_data
def load_forecast_quarterly():
    return pd.read_csv(C_FORECAST_Q)


@st.cache_data
def load_forecast_weekly():
    return pd.read_csv(C_FORECAST_W)


@st.cache_data(ttl=1800)
def load_overview_stats():
    """Live DB summary for the Overview page: total speeches, per-source
    breakdown, and date range. Small aggregate query, not the CSV cache
    tier -- always reflects the current DB regardless of when
    precompute_cache.py was last run."""
    if not os.path.exists(DB_PATH):
        return {"total": 0, "by_source": pd.DataFrame(), "min_date": None, "max_date": None}
    conn = sqlite3.connect(DB_PATH)
    try:
        by_source = pd.read_sql_query(
            "SELECT source, COUNT(*) as n FROM speeches GROUP BY source ORDER BY n DESC", conn
        )
        bounds = pd.read_sql_query(
            "SELECT MIN(date) as min_date, MAX(date) as max_date FROM speeches "
            "WHERE date IS NOT NULL AND date != 'N/A'", conn
        )
        return {
            "total": int(by_source["n"].sum()) if not by_source.empty else 0,
            "by_source": by_source,
            "min_date": bounds["min_date"].iloc[0] if not bounds.empty else None,
            "max_date": bounds["max_date"].iloc[0] if not bounds.empty else None,
        }
    finally:
        conn.close()


@st.cache_data(ttl=1800)
def load_sentiment_by_source(n_days=None):
    """Per-source sentiment timeline for the Overview chart -- one line per
    speech source instead of a single blended line across sources with very
    different publication cadences (MKB monthly, ECB/Fed near-daily)."""
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query('''
            SELECT s.date, s.source, ss.compound
            FROM speeches s
            JOIN sentiment_scores ss ON ss.speech_id = s.id AND ss.segment_type = 'episode'
            WHERE s.date IS NOT NULL AND s.date != 'N/A'
        ''', conn)
    finally:
        conn.close()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    return (
        df.groupby(["source", pd.Grouper(key="date", freq="MS")])["compound"]
        .mean()
        .reset_index()
        .sort_values("date")
    )


@st.cache_data(ttl=1800)
def load_recent_speeches(n=6):
    """Most recent speeches (any source) with sentiment, for the Overview
    'Recent Speeches' cards -- replaces the old raw-text report dump."""
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query('''
            SELECT s.date, s.source, s.title, ss.compound
            FROM speeches s
            LEFT JOIN sentiment_scores ss ON ss.speech_id = s.id AND ss.segment_type = 'episode'
            WHERE s.date IS NOT NULL AND s.date != 'N/A'
            ORDER BY s.date DESC
            LIMIT ?
        ''', conn, params=(n,))
        return df
    finally:
        conn.close()


# ==========================================================
# LOAD
# ==========================================================

df_weekly     = load_sector_weekly()
df_quarterly  = load_sector_quarterly()
df_regime_w   = load_regime_weekly()
df_regime_q   = load_regime_quarterly()
df_sector_avg = load_sector_avg()
df_sentiment  = load_sentiment_timeline()
df_topics     = load_topics()
df_nmf        = load_nmf_topics()
df_forecast_q = load_forecast_quarterly()
df_forecast_w = load_forecast_weekly()

# --- Cached AI Predictions wrappers for high performance ---
@st.cache_data(ttl=1800, show_spinner=False)
def _cached_company_predictions(sentiment: float, topic_str: float, regime: str, hist_ret: float) -> list:
    if not _PREDICTION_ENGINE_OK:
        return []
    return get_all_company_predictions(
        sentiment_score=sentiment,
        topic_strength=topic_str,
        regime_label=regime,
        historical_return=hist_ret,
        use_llm=False
    )

@st.cache_data(ttl=1800, show_spinner=False)
def _cached_sector_predictions(sentiment: float, topic_str: float, sector_avg_json: str, regime_df_json: str) -> list:
    if not _PREDICTION_ENGINE_OK:
        return []
    import io
    import json
    # Reconstruct inputs if JSON strings are passed to keep hashable types.
    # io.StringIO(...) is required, not the raw string -- newer pandas
    # (3.x) treats a bare str argument to read_json as a file path, not
    # inline JSON content, and raises FileNotFoundError otherwise.
    sector_avg = pd.read_json(io.StringIO(sector_avg_json)) if sector_avg_json else None
    regime_df = pd.read_json(io.StringIO(regime_df_json)) if regime_df_json else None
    return get_all_sector_predictions(
        sentiment_score=sentiment,
        topic_strength=topic_str,
        sector_returns=sector_avg,
        regime_df=regime_df
    )

# ==========================================================
# HELPERS
# ==========================================================

def regime_label(ret):
    if ret > REGIME_THRESH * 100:
        return "🟢 BULL", "#34d399"
    elif ret < -REGIME_THRESH * 100:
        return "🔴 BEAR", "#f87171"
    return "⚪ NEUTRAL", "#94a3b8"


def build_sector_fig(sdf, sector, period_col, rcol_pct, period_label):
    """Build a per-sector chart showing Market Return vs Topic Strength."""
    color = SECTOR_COLORS.get(sector, "#94a3b8")
    strength_color = "#f472b6" # Pinkish for strength

    fig = go.Figure()

    # ── Regime bands (group contiguous) ───────────────────
    blocks, cur, start, bg = [], None, None, None
    for _, row in sdf.iterrows():
        r = row[rcol_pct]
        if r > REGIME_THRESH * 100:
            regime, c = "Bull", "rgba(52,211,153,0.13)"
        elif r < -REGIME_THRESH * 100:
            regime, c = "Bear", "rgba(248,113,113,0.13)"
        else:
            regime, c = "Neutral", "rgba(148,163,184,0.06)"

        if regime != cur:
            if cur is not None:
                blocks.append((cur, start, row[period_col], bg))
            cur, start, bg = regime, row[period_col], c

    if cur is not None:
        blocks.append((cur, start, sdf.iloc[-1][period_col], bg))

    for idx, (b_regime, b_start, b_end, b_color) in enumerate(blocks):
        fig.add_vrect(
            x0=b_start, x1=b_end,
            fillcolor=b_color, layer="below", line_width=0,
            annotation_text=b_regime if idx % max(1, len(blocks) // 4) == 0 else "",
            annotation_font_size=8, annotation_font_color="#94a3b8",
            annotation_position="top left"
        )

    # ── Topic Strength Line (Secondary Y) ──────────────────
    if "speech_count" in sdf.columns:
        fig.add_trace(go.Scatter(
            x=sdf[period_col], y=sdf["speech_count"],
            mode="lines+markers", name="Topic Strength",
            line=dict(color=strength_color, width=2, dash='dot'),
            marker=dict(size=4),
            yaxis="y2",
            hovertemplate=(
                f"{period_label}: %{{x}}<br>"
                "Strength (Speeches): %{y}<extra>Intelligence</extra>"
            )
        ))

    # ── Market Return line (Primary Y) ──────────────────────
    rt_info = ""
    if "return_time" in sdf.columns:
        # Include return time in hover
        sdf['hover_text'] = sdf.apply(lambda r: f"<br>Est. Return Time: {r['return_time']}", axis=1)
    else:
        sdf['hover_text'] = ""

    fig.add_trace(go.Scatter(
        x=sdf[period_col], y=sdf[rcol_pct],
        mode="lines+markers", name="Market Return",
        line=dict(color=color, width=2.5), marker=dict(size=6),
        customdata=sdf['return_time'] if 'return_time' in sdf.columns else [None]*len(sdf),
        hovertemplate=(
            f"{period_label}: %{{x}}<br>"
            f"Return: %{{y:.3f}}%<br>"
            "Est. Max Return Time: %{customdata}<extra>" + sector + "</extra>"
        )
    ))

    fig.add_hline(y=0, line_dash="dot", line_color="#475569", line_width=1)

    last_ret = sdf.iloc[-1][rcol_pct]
    rlabel, rcolor = regime_label(last_ret)

    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=(
                f"<b>{sector}</b>  "
                f"<span style='color:{rcolor};font-size:13px'>{rlabel}</span>"
            ),
            font=dict(size=15)
        ),
        height=350,
        xaxis=dict(title=period_label, tickangle=-45, nticks=10),
        yaxis=dict(title="Avg Return (%)", zeroline=True, zerolinecolor="#475569", side="left"),
        yaxis2=dict(
            title="Topic Strength (Speeches)",
            overlaying="y",
            side="right",
            showgrid=False,
            rangemode="tozero"
        ),
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
        margin=dict(t=55, b=50, l=50, r=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/bullish.png", width=50)
    st.markdown("## BaatSeBharat")
    st.markdown("---")
    page = st.radio(
        "SELECT VIEW",
        [
            "📊 OVERVIEW",
            "📈 MARKET DYNAMICS",
            "🔎 SPEECH AUDIT",
            "🧠 TOPIC EXPLORER",
            "🤖 AI PREDICTIONS",
            "🌍 GLOBAL INFLUENCE MAP",
            "🔮 GLOBAL PREVIEW",
        ]
    )
    st.markdown("---")
    if _PREDICTION_ENGINE_OK and _llm_mode_available():
        st.success("🤖 LLM Mode Active")
    else:
        st.caption("🤖 AI: rule-based mode")
    st.caption("v3.0 | TradingAgents + GeoDashboard")

# ==========================================================
# OVERVIEW
# ==========================================================

if page == "📊 OVERVIEW":

    st.markdown(
        '<h1 class="hero-title">Market Influence Analytics</h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="hero-subtitle">Leadership discourse vs market behaviour.</p>',
        unsafe_allow_html=True
    )

    SOURCE_COLORS = {
        "Mann Ki Baat": "#38bdf8",
        "ECB":          "#a78bfa",
        "Fed":          "#fb923c",
    }

    ov_stats = load_overview_stats()
    ov_sentiment_by_source = load_sentiment_by_source()

    # ── Summary metrics ────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Speeches", f"{ov_stats['total']:,}")
    with m2:
        st.metric("Sources Tracked", len(ov_stats["by_source"]) if not ov_stats["by_source"].empty else 0)
    with m3:
        st.metric("Topics Modeled", len(df_topics) if not df_topics.empty else "N/A")
    with m4:
        if not ov_sentiment_by_source.empty:
            avg_sent = round(ov_sentiment_by_source["compound"].mean(), 3)
            st.metric("Avg Sentiment", avg_sent)
        else:
            st.metric("Avg Sentiment", "N/A")

    # ── Source coverage pills ──────────────────────────────
    if not ov_stats["by_source"].empty:
        pills = "".join(
            f"<span style='background:{SOURCE_COLORS.get(row['source'], '#334155')}22;"
            f"color:{SOURCE_COLORS.get(row['source'], '#94a3b8')};"
            f"border:1px solid {SOURCE_COLORS.get(row['source'], '#334155')};"
            f"border-radius:999px;padding:5px 14px;margin-right:8px;"
            f"font-size:0.85rem;font-weight:600;display:inline-block;margin-bottom:8px'>"
            f"{row['source']} · {row['n']:,}</span>"
            for _, row in ov_stats["by_source"].iterrows()
        )
        date_range = ""
        if ov_stats["min_date"] and ov_stats["max_date"]:
            date_range = (
                f"<span style='color:#64748b;font-size:0.85rem;margin-left:4px'>"
                f"&nbsp;·&nbsp;{ov_stats['min_date']} → {ov_stats['max_date']}</span>"
            )
        st.markdown(f"<div style='margin:4px 0 20px'>{pills}{date_range}</div>", unsafe_allow_html=True)

    # ── Sentiment timeline, one line per source ────────────
    st.markdown("### Sentiment Timeline")
    if not ov_sentiment_by_source.empty:
        fig = go.Figure()
        for source, sdf in ov_sentiment_by_source.groupby("source"):
            fig.add_trace(go.Scatter(
                x=sdf["date"], y=sdf["compound"],
                mode="lines+markers", name=source,
                line=dict(color=SOURCE_COLORS.get(source, "#94a3b8"), width=2),
                marker=dict(size=4),
            ))
        fig.add_hline(y=0, line_color="#334155", line_width=1)
        fig.update_layout(
            template="plotly_dark", height=420,
            plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
            yaxis_title="Avg. Monthly Sentiment (FinBERT compound)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            margin=dict(t=40, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No sentiment data yet — run the preprocessing/sentiment pipeline to populate this chart.")

    # ── Recent speeches, as cards ───────────────────────────
    st.markdown("### 📰 Recent Speeches")
    recent = load_recent_speeches(6)
    if not recent.empty:
        rcols = st.columns(3)
        for i, (_, row) in enumerate(recent.iterrows()):
            compound = row["compound"] if pd.notna(row["compound"]) else 0.0
            if compound > 0.15:
                tone_color, tone_label = "#34d399", "🟢 Positive"
            elif compound < -0.15:
                tone_color, tone_label = "#f87171", "🔴 Negative"
            else:
                tone_color, tone_label = "#94a3b8", "⚪ Neutral"
            src_color = SOURCE_COLORS.get(row["source"], "#94a3b8")
            with rcols[i % 3]:
                st.markdown(
                    f"""
                    <div style='background:#1e293b;border-radius:12px;padding:16px;
                                border:1px solid #334155;border-left:3px solid {src_color};
                                margin-bottom:16px;min-height:150px'>
                        <div style='color:{src_color};font-size:0.75rem;font-weight:700;
                                    text-transform:uppercase;letter-spacing:0.05em'>
                            {row['source']} &nbsp;·&nbsp; {row['date']}
                        </div>
                        <div style='color:#f1f5f9;font-weight:600;margin:8px 0;
                                    font-size:0.95rem;line-height:1.35'>
                            {row['title']}
                        </div>
                        <div style='color:{tone_color};font-size:0.85rem;font-weight:600'>
                            {tone_label} ({compound:+.2f})
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.info("No speeches available yet.")

# ==========================================================
# MARKET DYNAMICS
# ==========================================================

elif page == "📈 MARKET DYNAMICS":

    st.markdown(
        '<h1 class="hero-title">Market Dynamics — Sector Intelligence</h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="hero-subtitle">Per-sector return timelines, regime shifts and topic influence.</p>',
        unsafe_allow_html=True
    )

    # ── Controls ──────────────────────────────────────────
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        granularity = st.radio(
            "Time Granularity", ["Quarterly", "Weekly"],
            horizontal=True, key="mkt_gran"
        )
    with col_ctrl2:
        return_horizon = st.radio(
            "Return Horizon", ["5-Day (1 Week)", "10-Day (2 Weeks)"],
            horizontal=True, key="mkt_ret"
        )

    use_5d       = "5-Day" in return_horizon
    rcol_raw     = "return_5d"  if use_5d else "return_10d"
    rcol_pct_col = rcol_raw + "_pct"  # we'll create this below

    if granularity == "Weekly":
        df_agg   = df_weekly.copy()
        df_reg   = df_regime_w.copy()
        period_col   = "week"
        period_label = "Week"
    else:
        df_agg   = df_quarterly.copy()
        df_reg   = df_regime_q.copy()
        period_col   = "quarter"
        period_label = "Quarter"

    # Convert to % for display
    df_agg[rcol_pct_col] = df_agg[rcol_raw] * 100

    SECTORS = list(df_agg["sector"].unique())

    st.markdown("---")
    st.markdown("### 📊 Per-Sector Return + Regime Shift")

    left_sectors  = SECTORS[:len(SECTORS) // 2 + len(SECTORS) % 2]
    right_sectors = SECTORS[len(SECTORS) // 2 + len(SECTORS) % 2:]

    col_left, col_right = st.columns(2)

    with col_left:
        for sec in left_sectors:
            sdf = df_agg[df_agg["sector"] == sec].sort_values(period_col)
            st.plotly_chart(
                build_sector_fig(sdf, sec, period_col, rcol_pct_col, period_label),
                use_container_width=True,
                key=f"left_{sec}_{granularity}_{return_horizon}"
            )

    with col_right:
        for sec in right_sectors:
            sdf = df_agg[df_agg["sector"] == sec].sort_values(period_col)
            st.plotly_chart(
                build_sector_fig(sdf, sec, period_col, rcol_pct_col, period_label),
                use_container_width=True,
                key=f"right_{sec}_{granularity}_{return_horizon}"
            )

    # ── Regime summary table ───────────────────────────────
    st.markdown("---")
    st.markdown("### 🔄 Current Market Regime — All Sectors")

    regime_display = df_reg.copy()
    regime_display["Current Regime"] = regime_display["regime"].map({
        "Bull":    "🟢 Bull",
        "Bear":    "🔴 Bear",
        "Neutral": "⚪ Neutral",
    })
    regime_display = regime_display.rename(columns={
        "sector":      "Sector",
        "last_period": f"Latest {period_label}",
        "last_ret_5d": "Latest Return (%)",
        "avg_ret_5d":  "Avg Return (%)",
        "bull_pct":    "🟢 Bull %",
        "bear_pct":    "🔴 Bear %",
        "neutral_pct": "⚪ Neutral %",
    })[[
        "Sector", "Current Regime", f"Latest {period_label}",
        "Latest Return (%)", "Avg Return (%)",
        "🟢 Bull %", "🔴 Bear %", "⚪ Neutral %"
    ]]
    st.dataframe(regime_display, use_container_width=True, hide_index=True)

    # ── ✨ FUTURE REGIME FORECAST ──────────────────────────
    st.markdown("---")
    st.markdown(
        "### 🔮 Future Market Regime Forecast"
    )
    st.markdown(
        '<p class="hero-subtitle">'
        'Markov chain transition model · Linear return extrapolation · 95% confidence band'
        '</p>',
        unsafe_allow_html=True
    )

    df_fc = df_forecast_q.copy() if granularity == "Quarterly" else df_forecast_w.copy()
    fc_period_col = "period"

    REGIME_FC_COLORS = {
        "Bull":    "#34d399",
        "Bear":    "#f87171",
        "Neutral": "#94a3b8",
    }
    REGIME_EMOJI = {
        "Bull": "🟢", "Bear": "🔴", "Neutral": "⚪"
    }

    fc_left = SECTORS[:len(SECTORS) // 2 + len(SECTORS) % 2]
    fc_right = SECTORS[len(SECTORS) // 2 + len(SECTORS) % 2:]

    def build_forecast_fig(sector):
        fdf = df_fc[df_fc["sector"] == sector].copy()
        if fdf.empty:
            return go.Figure()

        sec_color = SECTOR_COLORS.get(sector, "#94a3b8")

        fig = go.Figure()

        # ── Stacked probability bars ───────────────────────
        fig.add_trace(go.Bar(
            x=fdf[fc_period_col], y=fdf["bull_prob"],
            name="🟢 Bull %", marker_color="rgba(52,211,153,0.75)",
            hovertemplate="%{x}<br>Bull: %{y:.1f}%<extra></extra>",
            yaxis="y1"
        ))
        fig.add_trace(go.Bar(
            x=fdf[fc_period_col], y=fdf["neutral_prob"],
            name="⚪ Neutral %", marker_color="rgba(148,163,184,0.55)",
            hovertemplate="%{x}<br>Neutral: %{y:.1f}%<extra></extra>",
            yaxis="y1"
        ))
        fig.add_trace(go.Bar(
            x=fdf[fc_period_col], y=fdf["bear_prob"],
            name="🔴 Bear %", marker_color="rgba(248,113,113,0.75)",
            hovertemplate="%{x}<br>Bear: %{y:.1f}%<extra></extra>",
            yaxis="y1"
        ))

        # ── Confidence band (shaded area) ──────────────────
        fig.add_trace(go.Scatter(
            x=list(fdf[fc_period_col]) + list(fdf[fc_period_col])[::-1],
            y=list(fdf["ret_upper"]) + list(fdf["ret_lower"])[::-1],
            fill="toself",
            fillcolor=f"rgba({int(sec_color[1:3],16)},{int(sec_color[3:5],16)},{int(sec_color[5:7],16)},0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% CI",
            hoverinfo="skip",
            yaxis="y2"
        ))

        # ── Forecast return line ───────────────────────────
        fig.add_trace(go.Scatter(
            x=fdf[fc_period_col], y=fdf["forecasted_return"],
            mode="lines+markers+text",
            name="Forecast Return",
            line=dict(color=sec_color, width=2.5, dash="dash"),
            marker=dict(size=7, symbol="diamond"),
            text=[f"{v:+.2f}%" for v in fdf["forecasted_return"]],
            textposition="top center",
            textfont=dict(size=9, color=sec_color),
            hovertemplate="%{x}<br>Forecast: %{y:+.3f}%<extra></extra>",
            yaxis="y2"
        ))

        # ── Predicted regime badges as annotations ─────────
        for _, row in fdf.iterrows():
            fig.add_annotation(
                x=row[fc_period_col],
                y=103,
                xref="x", yref="y1",
                text=REGIME_EMOJI.get(row["predicted_regime"], ""),
                showarrow=False,
                font=dict(size=14)
            )

        top_regime = fdf["predicted_regime"].mode()[0]
        rc = REGIME_FC_COLORS[top_regime]

        fig.update_layout(
            template="plotly_dark",
            title=dict(
                text=(
                    f"<b>{sector}</b>  "
                    f"<span style='color:{rc};font-size:12px'>"
                    f"→ {REGIME_EMOJI[top_regime]} {top_regime.upper()} EXPECTED</span>"
                ),
                font=dict(size=14)
            ),
            barmode="stack",
            height=370,
            xaxis=dict(title=period_label, tickangle=-30),
            yaxis=dict(
                title="Regime Probability (%)",
                range=[0, 115],
                side="left",
                showgrid=False
            ),
            yaxis2=dict(
                title="Forecast Return (%)",
                overlaying="y",
                side="right",
                showgrid=False,
                zeroline=True,
                zerolinecolor="#475569"
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom", y=-0.38,
                xanchor="center", x=0.5,
                font=dict(size=9)
            ),
            plot_bgcolor="#0f172a",
            paper_bgcolor="#0f172a",
            margin=dict(t=55, b=90, l=55, r=60)
        )
        return fig

    fc_col_left, fc_col_right = st.columns(2)
    with fc_col_left:
        for sec in fc_left:
            st.plotly_chart(
                build_forecast_fig(sec),
                use_container_width=True,
                key=f"fc_left_{sec}_{granularity}"
            )
    with fc_col_right:
        for sec in fc_right:
            st.plotly_chart(
                build_forecast_fig(sec),
                use_container_width=True,
                key=f"fc_right_{sec}_{granularity}"
            )

    # ── Forecast summary table ─────────────────────────────
    st.markdown("#### Forecast Summary")
    fc_summary = df_fc.copy()
    fc_summary["Predicted Regime"] = fc_summary["predicted_regime"].map({
        "Bull": "🟢 Bull", "Bear": "🔴 Bear", "Neutral": "⚪ Neutral"
    })
    fc_summary = fc_summary.rename(columns={
        "sector":            "Sector",
        "period":            period_label,
        "bull_prob":         "🟢 Bull %",
        "bear_prob":         "🔴 Bear %",
        "neutral_prob":      "⚪ Neutral %",
        "forecasted_return": "Forecast Return (%)",
        "ret_lower":         "Lower 95%",
        "ret_upper":         "Upper 95%",
    })[["Sector", period_label, "Predicted Regime",
        "🟢 Bull %", "🔴 Bear %", "⚪ Neutral %",
        "Forecast Return (%)", "Lower 95%", "Upper 95%"]]
    st.dataframe(fc_summary, use_container_width=True, hide_index=True)

    # ── Topic strength vs sector overlay ──────────────────
    st.markdown("---")
    st.markdown("### 🧩 Topic Strength vs Sector Returns")

    sa = df_sector_avg.copy()
    rcol_avg = "return_5d" if use_5d else "return_10d"
    sa["avg_pct"] = sa[rcol_avg] * 100

    fig_ts = go.Figure()

    fig_ts.add_trace(go.Bar(
        x=df_topics["short_kw"],
        y=df_topics["score"],
        name="Topic Strength (LDA)",
        marker_color="#a78bfa",
        text=df_topics["score"].round(3),
        textposition="outside",
        hovertemplate=(
            "<b>Topic %{customdata[0]}</b><br>"
            "Keywords: %{customdata[1]}<br>"
            "Score: %{y:.3f}<extra></extra>"
        ),
        customdata=df_topics[["topic_id", "keywords"]].values,
        yaxis="y1"
    ))

    # Legend markers
    for _, row in sa.iterrows():
        fig_ts.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color=SECTOR_COLORS.get(row["sector"], "#94a3b8"), size=10),
            name=f"{row['sector']} ({row['avg_pct']:.3f}%)",
            yaxis="y2"
        ))

    for _, row in sa.iterrows():
        fig_ts.add_hline(
            y=row["avg_pct"],
            line_color=SECTOR_COLORS.get(row["sector"], "#94a3b8"),
            line_dash="dash", line_width=1.5,
            annotation_text=row["sector"],
            annotation_position="right",
            annotation_font_size=10,
            annotation_font_color=SECTOR_COLORS.get(row["sector"], "#94a3b8"),
            yref="y2"
        )

    fig_ts.update_layout(
        template="plotly_dark", height=420,
        title="Topic Strength (bars) vs Sector Avg Returns (dashed lines)",
        xaxis=dict(title="Topic Keywords", tickangle=-30),
        yaxis=dict(title="Topic Score", side="left"),
        yaxis2=dict(title="Avg Return (%)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=-0.45, xanchor="center", x=0.5),
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
        margin=dict(t=55, b=130, l=55, r=90)
    )
    st.plotly_chart(fig_ts, use_container_width=True)

# ==========================================================
# SPEECH AUDIT
# ==========================================================

elif page == "🔎 SPEECH AUDIT":

    st.markdown(
        '<h1 class="hero-title">Speech Intelligence Explorer</h1>',
        unsafe_allow_html=True
    )

    # Load full speech data only on this page
    @st.cache_data
    def load_speeches():
        df = pd.read_csv(
            TOPIC_DATASET,
            usecols=["date", "source", "filename", "text", "positive", "negative", "neutral"],
            parse_dates=["date"]
        )
        df = df.dropna(subset=["date"])
        df["sentiment"] = df["positive"] - df["negative"]
        df["title"] = df["filename"].str.replace(".txt", "", regex=False)
        return df

    df_speeches = load_speeches()

    search = st.text_input("Filter by keyword")
    filtered = df_speeches.copy()
    if search:
        filtered = filtered[
            filtered["title"].str.contains(search, case=False, na=False)
        ]

    st.dataframe(
        filtered[["date", "source", "title", "positive", "negative", "neutral"]]
        .sort_values("date", ascending=False),
        use_container_width=True, hide_index=True
    )

    st.markdown("---")
    if not filtered.empty:
        selected_title = st.selectbox("Choose speech", filtered["title"].unique())
        speech = filtered[filtered["title"] == selected_title].iloc[0]

        st.markdown("### Speech Text")
        st.text_area("Transcript", speech["text"], height=280)

        fig_s = go.Figure(data=[go.Bar(
            x=["Positive", "Negative", "Neutral"],
            y=[speech["positive"], speech["negative"], speech["neutral"]],
            marker_color=["#34d399", "#f87171", "#94a3b8"]
        )])
        fig_s.update_layout(
            title="FinBERT Sentiment", template="plotly_dark",
            plot_bgcolor="#0f172a", paper_bgcolor="#0f172a"
        )
        st.plotly_chart(fig_s, use_container_width=True)

# ==========================================================
# TOPIC EXPLORER
# ==========================================================

elif page == "🧠 TOPIC EXPLORER":

    st.markdown(
        '<h1 class="hero-title">Financial Topic Intelligence</h1>',
        unsafe_allow_html=True
    )

    st.markdown("### LDA Topics")
    st.dataframe(df_topics, use_container_width=True, hide_index=True)

    st.markdown("### NMF Topics")
    st.dataframe(df_nmf, use_container_width=True, hide_index=True)

    top_scores = (
        df_topics.sort_values("score", ascending=False).head(10).copy()
    )

    fig = go.Figure(go.Bar(
        x=top_scores["short_kw"],
        y=top_scores["score"],
        marker_color="#38bdf8",
        text=top_scores["score"].round(3),
        textposition="outside",
        customdata=top_scores[["topic_id", "keywords"]].values,
        hovertemplate=(
            "<b>Topic %{customdata[0]}</b><br>"
            "<b>Keywords:</b> %{customdata[1]}<br>"
            "Score: %{y:.3f}<extra></extra>"
        )
    ))

    fig.update_layout(
        template="plotly_dark",
        title="Top Financial Topics by Score",
        xaxis=dict(title="Topic Keywords (first 4)", tickangle=-30),
        yaxis=dict(title="Topic Score"),
        height=420,
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
        margin=dict(t=55, b=120, l=55, r=20)
    )
    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# AI PREDICTIONS  (Phase 2 — TradingAgents Integration)
# ==========================================================

elif page == "🤖 AI PREDICTIONS":

    st.markdown(
        '<h1 class="hero-title">🤖 AI Market Predictions</h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="hero-subtitle">'
        'Company- and sector-level forecasts powered by BaatSeBharat NLP signals.</p>',
        unsafe_allow_html=True
    )

    if not _PREDICTION_ENGINE_OK:
        st.error(f"Prediction engine unavailable: {_pe_error}")
        st.stop()

    # ── Signal inputs (from BaatSeBharat cache) ───────────────
    st.markdown("---")
    st.markdown("### ⚙️ Signal Inputs")
    st.caption(
        "These values are derived from the BaatSeBharat NLP pipeline "
        "(FinBERT sentiment, topic strengths, regime labels). "
        "You can override them manually below."
    )

    # Load live values from cache
    _sentiment_default = 0.0
    _topic_default     = 0.5
    _regime_default    = "Neutral"
    _hist_ret_default  = 0.0

    if not df_sentiment.empty and "sentiment" in df_sentiment.columns:
        _sentiment_default = float(df_sentiment["sentiment"].tail(30).mean())

    if not df_topics.empty and "score" in df_topics.columns:
        _topic_default = float(df_topics["score"].max())

    if not df_sector_avg.empty:
        _hist_ret_default = float(df_sector_avg["return_5d"].mean())

    if not df_regime_q.empty and "regime" in df_regime_q.columns:
        _regime_default = str(df_regime_q["regime"].value_counts().idxmax())

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        user_sentiment = st.slider(
            "FinBERT Sentiment", -1.0, 1.0, float(round(_sentiment_default, 3)), 0.01
        )
    with col_s2:
        user_topic_str = st.slider(
            "Topic Strength", 0.0, 1.0, float(round(_topic_default, 3)), 0.01
        )
    with col_s3:
        user_regime = st.selectbox(
            "Market Regime",
            ["Bull", "Neutral", "Bear", "Stable", "Transitional", "Volatile"],
            index=["Bull", "Neutral", "Bear", "Stable", "Transitional", "Volatile"].index(
                _regime_default if _regime_default in
                ["Bull", "Neutral", "Bear", "Stable", "Transitional", "Volatile"]
                else "Neutral"
            )
        )
    with col_s4:
        use_llm = st.checkbox(
            "Use LLM Mode",
            value=_llm_mode_available(),
            disabled=not _llm_mode_available(),
            help="Requires OPENAI_API_KEY or GOOGLE_API_KEY in .env"
        )

    horizons = [1, 5, 10]

    # ── Tabs: Company vs Sector ────────────────────────────────
    pred_tab1, pred_tab2 = st.tabs(["🏢 Company Predictions", "📦 Sector Predictions"])

    with pred_tab1:
        st.markdown("### 🏢 Company-Level Predictions")
        st.caption(
            "Predictions for 15 major Indian companies. "
            "Signal derived from sentiment + topic + regime + price momentum."
        )

        sel_company = st.selectbox(
            "Select Company for Detailed View",
            list(COMPANY_UNIVERSE.keys())
        )

        with st.spinner(f"Generating prediction for {sel_company}…"):
            pred = get_company_prediction(
                sel_company,
                sentiment_score=user_sentiment,
                topic_strength=user_topic_str,
                regime_label=user_regime,
                historical_return=_hist_ret_default,
                use_llm=use_llm,
            )

        # ── Signal card ──────────────────────────────────────
        signal_color = {"Bullish": "#34d399", "Bearish": "#f87171", "Neutral": "#94a3b8"}[
            pred["signal"]
        ]
        st.markdown(
            f"""
            <div style='background:#1e293b;border-radius:12px;padding:20px;border:1px solid {signal_color};margin-bottom:16px'>
                <div style='font-size:2rem;font-weight:700;color:{signal_color}'>
                    {pred['emoji']} {pred['signal'].upper()}
                </div>
                <div style='color:#94a3b8;font-size:0.95rem;margin-top:4px'>
                    {sel_company} &nbsp;|&nbsp; Ticker: {pred['ticker'] or 'N/A'}
                    &nbsp;|&nbsp; Mode: {pred['mode'].upper()}
                    &nbsp;|&nbsp; Confidence: <b style='color:{signal_color}'>{pred['confidence']:.0f}%</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Current price ─────────────────────────────────────
        if pred.get("current_price"):
            st.metric(
                f"Current Price ({pred['ticker']})",
                f"₹{pred['current_price']:,.2f}"
            )

        # ── Horizon forecast cards ────────────────────────────
        st.markdown("#### 📅 Forecast Horizons")
        h_cols = st.columns(3)
        for idx, h in enumerate(horizons):
            fcast = pred["predictions"].get(h, {})
            ret   = fcast.get("return_pct", 0.0)
            rlow  = fcast.get("return_low", 0.0)
            rhigh = fcast.get("return_high", 0.0)
            ret_color = "#34d399" if ret > 0 else ("#f87171" if ret < 0 else "#94a3b8")

            with h_cols[idx]:
                p_mid  = fcast.get("price_mid",  None)
                p_low  = fcast.get("price_low",  None)
                p_high = fcast.get("price_high", None)

                price_str = ""
                if p_mid:
                    price_str = (
                        f"<div style='font-size:0.8rem;color:#64748b;margin-top:4px'>"
                        f"₹{p_low:,.0f} – ₹{p_high:,.0f}</div>"
                    )

                st.markdown(
                    f"""
                    <div style='background:#1e293b;border-radius:10px;padding:16px;border:1px solid #334155;text-align:center'>
                        <div style='color:#94a3b8;font-size:0.85rem'>{fcast.get('label','')}</div>
                        <div style='font-size:1.6rem;font-weight:700;color:{ret_color}'>
                            {ret:+.2f}%
                        </div>
                        <div style='font-size:0.75rem;color:#475569'>
                            Range: {rlow:+.1f}% to {rhigh:+.1f}%
                        </div>
                        {price_str}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ── Input signals summary ─────────────────────────────
        with st.expander("🔍 Input Signals Used"):
            inputs = pred.get("inputs", {})
            si_df = pd.DataFrame([
                {"Signal": "FinBERT Sentiment",       "Value": f"{inputs.get('sentiment', 0):+.3f}"},
                {"Signal": "Dominant Topic Strength", "Value": f"{inputs.get('topic_strength', 0):.3f}"},
                {"Signal": "Market Regime",           "Value": inputs.get('regime', 'N/A')},
                {"Signal": "5-Day Price Momentum",    "Value": f"{inputs.get('momentum_5d_pct', 0):+.2f}%"},
                {"Signal": "Historical Avg Return",   "Value": f"{inputs.get('historical_return_pct', 0):+.2f}%"},
            ])
            st.dataframe(si_df, use_container_width=True, hide_index=True)

        if use_llm and pred.get("llm_decision"):
            with st.expander("🤖 LLM Decision Reasoning"):
                st.text(pred["llm_decision"])

        st.markdown("---")

        # ── All companies grid ────────────────────────────────
        st.markdown("### 📋 All Companies Overview")
        with st.spinner("Loading cached bulk predictions…"):
            all_preds = _cached_company_predictions(
                round(user_sentiment, 3),
                round(user_topic_str, 3),
                user_regime,
                round(_hist_ret_default, 6)
            )

        # Summary bar chart
        pred_rows = []
        for p in all_preds:
            fcast_1d = p["predictions"].get(1, {})
            fcast_5d = p["predictions"].get(5, {})
            fcast_10d = p["predictions"].get(10, {})
            pred_rows.append({
                "Company":     p["company"],
                "Signal":      p["signal"],
                "Confidence":  p["confidence"],
                "Score":       p["score"],
                "1D Forecast": fcast_1d.get("return_pct", 0),
                "5D Forecast": fcast_5d.get("return_pct", 0),
                "10D Forecast": fcast_10d.get("return_pct", 0),
            })
        pred_df = pd.DataFrame(pred_rows).sort_values("Score", ascending=False)

        # Color by signal
        signal_color_map = {"Bullish": "#34d399", "Neutral": "#94a3b8", "Bearish": "#f87171"}
        bar_colors = [signal_color_map.get(s, "#94a3b8") for s in pred_df["Signal"]]

        fig_companies = go.Figure(go.Bar(
            x=pred_df["Company"],
            y=pred_df["5D Forecast"],
            marker_color=bar_colors,
            text=[f"{v:+.2f}%" for v in pred_df["5D Forecast"]],
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "5D Forecast: %{y:+.2f}%<br>"
                "<extra></extra>"
            ),
        ))
        fig_companies.add_hline(y=0, line_dash="dot", line_color="#475569")
        fig_companies.update_layout(
            title="All Companies — 5-Day Return Forecast",
            template="plotly_dark",
            height=380,
            xaxis_tickangle=-35,
            yaxis_title="Forecast Return (%)",
            plot_bgcolor="#0f172a",
            paper_bgcolor="#0f172a",
        )
        st.plotly_chart(fig_companies, use_container_width=True)

        # Confidence scatter
        fig_conf = px.scatter(
            pred_df,
            x="Score",
            y="Confidence",
            color="Signal",
            size="Confidence",
            hover_name="Company",
            color_discrete_map=signal_color_map,
            title="Signal Score vs Confidence",
            template="plotly_dark",
        )
        fig_conf.update_layout(
            plot_bgcolor="#0f172a", paper_bgcolor="#0f172a", height=350
        )
        st.plotly_chart(fig_conf, use_container_width=True)

        with st.expander("📋 Full Predictions Table"):
            disp_pred = pred_df.copy()
            disp_pred["Signal"] = pred_df["Signal"].map(
                {"Bullish": "🟢 Bullish", "Neutral": "⚪ Neutral", "Bearish": "🔴 Bearish"}
            )
            for col in ["1D Forecast", "5D Forecast", "10D Forecast"]:
                disp_pred[col] = disp_pred[col].map(lambda x: f"{x:+.2f}%")
            disp_pred["Confidence"] = disp_pred["Confidence"].map(lambda x: f"{x:.0f}%")
            st.dataframe(disp_pred, use_container_width=True, hide_index=True)

    with pred_tab2:
        st.markdown("### 📦 Sector-Level Predictions")
        st.caption(
            "Aggregated sector forecasts using constituent company momentum "
            "+ BaatSeBharat regime & sector return data."
        )

        with st.spinner("Loading cached sector predictions…"):
            df_sector_avg_json = df_sector_avg.to_json() if not df_sector_avg.empty else ""
            df_regime_q_json = df_regime_q.to_json() if not df_regime_q.empty else ""
            sector_preds = _cached_sector_predictions(
                round(user_sentiment, 3),
                round(user_topic_str, 3),
                df_sector_avg_json,
                df_regime_q_json
            )

        # Sector cards in 3-column grid
        sector_rows = []
        for sp in sector_preds:
            fcast_1d  = sp["predictions"].get(1, {})
            fcast_5d  = sp["predictions"].get(5, {})
            fcast_10d = sp["predictions"].get(10, {})
            sector_rows.append({
                "Sector":      sp["sector"],
                "Signal":      sp["signal"],
                "Emoji":       sp["emoji"],
                "Confidence":  sp["confidence"],
                "Score":       sp["score"],
                "1D Forecast": fcast_1d.get("return_pct", 0),
                "5D Forecast": fcast_5d.get("return_pct", 0),
                "10D Forecast": fcast_10d.get("return_pct", 0),
            })

        sec_df = pd.DataFrame(sector_rows).sort_values("Score", ascending=False)

        # Sector cards
        n_sectors = len(sec_df)
        for i in range(0, n_sectors, 3):
            cols = st.columns(3)
            for j, row in enumerate(sec_df.iloc[i:i+3].itertuples()):
                ret5 = row._7  # 5D Forecast
                ret_color = "#34d399" if ret5 > 0 else ("#f87171" if ret5 < 0 else "#94a3b8")
                sig_color = {"Bullish": "#34d399", "Bearish": "#f87171", "Neutral": "#94a3b8"}[
                    row.Signal
                ]
                with cols[j]:
                    st.markdown(
                        f"""
                        <div style='background:#1e293b;border-radius:12px;padding:18px;
                                    border:1px solid {sig_color};margin-bottom:12px'>
                            <div style='font-size:1.1rem;font-weight:700;color:{sig_color}'>
                                {row.Emoji} {row.Sector}
                            </div>
                            <div style='color:#94a3b8;font-size:0.8rem;margin-top:2px'>
                                Confidence: {row.Confidence:.0f}%
                            </div>
                            <table style='width:100%;margin-top:10px;font-size:0.85rem'>
                                <tr>
                                    <td style='color:#64748b'>1D</td>
                                    <td style='color:{ret_color};text-align:right'>{row._6:+.2f}%</td>
                                </tr>
                                <tr>
                                    <td style='color:#64748b'>1W</td>
                                    <td style='color:{ret_color};text-align:right'>{ret5:+.2f}%</td>
                                </tr>
                                <tr>
                                    <td style='color:#64748b'>10D</td>
                                    <td style='color:{ret_color};text-align:right'>{row._8:+.2f}%</td>
                                </tr>
                            </table>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # Sector comparison bar chart
        sec_bar_colors = [
            signal_color_map.get(row.Signal, "#94a3b8") for row in sec_df.itertuples()
        ]
        fig_sec = go.Figure()
        for h_days, h_label, dash in [
            ("1D Forecast", "1-Day", "solid"),
            ("5D Forecast", "1-Week", "dot"),
            ("10D Forecast", "10-Day", "dash")
        ]:
            fig_sec.add_trace(go.Bar(
                name=h_label,
                x=sec_df["Sector"],
                y=sec_df[h_days],
                text=[f"{v:+.2f}%" for v in sec_df[h_days]],
                textposition="outside",
            ))

        fig_sec.add_hline(y=0, line_dash="dot", line_color="#475569")
        fig_sec.update_layout(
            title="Sector Return Forecasts — 1D / 1W / 10D",
            barmode="group",
            template="plotly_dark",
            height=380,
            yaxis_title="Forecast Return (%)",
            plot_bgcolor="#0f172a",
            paper_bgcolor="#0f172a",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_sec, use_container_width=True)

        st.info(
            "💡 **Note:** Predictions are derived from BaatSeBharat NLP signals "
            "(FinBERT sentiment, topic modeling, regime detection) combined with live price momentum. "
            "Enable LLM Mode with an API key for AI-enhanced analysis."
        )

# ==========================================================
# GLOBAL INFLUENCE MAP  (Phase 3 — GeoDashboard Integration)
# ==========================================================

elif page == "🌍 GLOBAL INFLUENCE MAP":

    if not _GEO_DASHBOARD_OK:
        st.error(f"GeoDashboard module unavailable: {_geo_error}")
        st.info("Install dependencies: `pip install wbdata kaleido`")
        st.stop()

    # Pass company predictions from the prediction engine if available
    _company_preds_for_map = None
    if _PREDICTION_ENGINE_OK:
        try:
            # Quick rule-based predictions for geo map (no LLM for speed)
            _sentiment_map = 0.0
            _topic_map     = 0.5
            if not df_sentiment.empty and "sentiment" in df_sentiment.columns:
                _sentiment_map = float(df_sentiment["sentiment"].tail(30).mean())
            if not df_topics.empty and "score" in df_topics.columns:
                _topic_map = float(df_topics["score"].max())
            _company_preds_for_map = _cached_company_predictions(
                round(_sentiment_map, 3),
                round(_topic_map, 3),
                "Neutral",
                0.0
            )
        except Exception:
            _company_preds_for_map = None

    render_global_influence_map(company_predictions=_company_preds_for_map)

# ==========================================================
# GLOBAL PREVIEW  (past speeches: predicted vs. actual)
# ==========================================================

elif page == "🔮 GLOBAL PREVIEW":

    st.markdown(
        '<h1 class="hero-title">Global Preview</h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="hero-subtitle">Past speeches: what the pipeline predicted vs. what actually happened.</p>',
        unsafe_allow_html=True
    )

    if not _PREDICTION_HISTORY_OK:
        st.error(f"Prediction history module unavailable: {_pred_history_error}")
        st.stop()

    gp_col1, gp_col2 = st.columns(2)
    with gp_col1:
        gp_source = st.selectbox(
            "Source", ["All", "Mann Ki Baat", "ECB", "Fed"], index=0
        )
    with gp_col2:
        gp_company = st.selectbox(
            "Company", ["All"] + list(COMPANY_UNIVERSE.keys()), index=0
        )

    with st.spinner("Replaying past predictions against realized outcomes…"):
        gp_df = compute_prediction_vs_actual(
            source=None if gp_source == "All" else gp_source,
            company=None if gp_company == "All" else gp_company,
        )

    if gp_df.empty:
        st.info(
            "Not enough data yet to build a Global Preview. This needs speeches with "
            "computed market impact (speech_market_impact) for the selected filters — "
            "run the prototype/impact pipeline first."
        )
        st.stop()

    gp_summary = summarize_prediction_history(gp_df)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        hr = gp_summary.get("overall_hit_rate")
        st.metric("Directional Hit Rate (5d)", f"{hr*100:.1f}%" if hr is not None else "N/A")
    with m2:
        st.metric("Mean Abs. Error (1d)", f"{gp_summary.get('mean_abs_error_1d', 0):.2f}%")
    with m3:
        st.metric("Mean Abs. Error (5d)", f"{gp_summary.get('mean_abs_error_5d', 0):.2f}%")
    with m4:
        st.metric("Speeches Covered", gp_summary.get("n_events", 0))

    st.markdown("### 📉 Predicted vs. Actual Return (5-Day)")
    gp_plot_df = gp_df.dropna(subset=["hit"]).copy()
    if not gp_plot_df.empty:
        gp_plot_df["Result"] = gp_plot_df["hit"].map({True: "Hit", False: "Miss"})
        fig_gp = px.scatter(
            gp_plot_df,
            x="predicted_return_5d", y="actual_return_5d",
            color="Result", hover_data=["date", "source", "company"],
            color_discrete_map={"Hit": "#34d399", "Miss": "#f87171"},
        )
        fig_gp.add_shape(
            type="line", x0=gp_plot_df["predicted_return_5d"].min(), y0=gp_plot_df["predicted_return_5d"].min(),
            x1=gp_plot_df["predicted_return_5d"].max(), y1=gp_plot_df["predicted_return_5d"].max(),
            line=dict(color="#64748b", dash="dot")
        )
        fig_gp.update_layout(
            template="plotly_dark", height=420,
            plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
            xaxis_title="Predicted 5d Return (%)", yaxis_title="Actual 5d Return (%)",
            margin=dict(t=20, b=40)
        )
        st.plotly_chart(fig_gp, use_container_width=True)
    else:
        st.caption("No events with a non-zero actual return to plot yet.")

    st.markdown("### 📋 Speech-Level Detail")
    st.dataframe(
        gp_df.sort_values("date", ascending=False)[[
            "date", "source", "company", "predicted_signal",
            "predicted_return_1d", "predicted_return_5d",
            "actual_return_1d", "actual_return_5d", "hit"
        ]],
        use_container_width=True,
        hide_index=True,
    )

    if gp_summary.get("per_company") is not None and not gp_summary["per_company"].empty:
        st.markdown("### 🏢 Accuracy by Company")
        st.dataframe(gp_summary["per_company"], use_container_width=True, hide_index=True)

    if gp_summary.get("per_source") is not None and not gp_summary["per_source"].empty:
        st.markdown("### 🗂️ Accuracy by Source")
        st.dataframe(gp_summary["per_source"], use_container_width=True, hide_index=True)
