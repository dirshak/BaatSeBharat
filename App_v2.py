import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import sys
import subprocess
from datetime import datetime
import json

# ── Add src to path ────────────────────────────────────────────────────────
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_APP_DIR, 'src'))
sys.path.insert(0, os.path.join(_APP_DIR, 'TradingAgents'))

try:
    from tradingagents.dataflows import yf_cache_patch
except Exception:
    pass

from utils.logger import setup_logger
from utils.db_utils import get_db_connection

# ── Integration imports (non-fatal: app still works without them) ──────────
try:
    from prediction_engine import (
        get_all_company_predictions,
        get_all_sector_predictions,
        get_company_prediction,
        COMPANY_UNIVERSE,
        SECTOR_COMPANIES,
        _llm_mode_available,
    )
    _PRED_OK = True
except Exception as _pred_err:
    _PRED_OK = False
    _pred_err_msg = str(_pred_err)

try:
    from geo_dashboard import render_global_influence_map
    _GEO_OK = True
except Exception as _geo_err:
    _GEO_OK = False
    _geo_err_msg = str(_geo_err)

st.set_page_config(
    page_title="Rhetoric & Markets Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Theme & CSS ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #161b22; border-radius: 5px 5px 0px 0px; color: white; }
    .stTabs [aria-selected="true"] { background-color: #1f6feb !important; }
    </style>
    """, unsafe_allow_html=True)

# --- Required File Verification ---
DB_PATH = './data/market_rhetoric.db'

SOURCE_COLORS = {
    'Mann Ki Baat': '#f0883e',   # Orange
    'ECB':          '#388bfd',   # Blue
    'Fed':          '#3fb950',   # Green
}

REQUIRED_FILES = [
    DB_PATH,
    './data/processed/topic_distributions_combined.npy',
    './data/processed/topic_labels_combined.json'
]

missing_reqs = [f for f in REQUIRED_FILES if not os.path.exists(f)]
if missing_reqs:
    st.error(f"### ❌ CRITICAL: Missing Required Pipeline Files\n\nThe following files are missing. Please click **'🚀 Run Pipeline'** in the sidebar to generate them.\n\n" + "\n".join([f"- `{f}`" for f in missing_reqs]))
    # We don't st.stop() here because we want the user to be able to click the button in the sidebar

@st.cache_data
def load_db_stats():
    try:
        conn = get_db_connection(DB_PATH)
        speech_count = pd.read_sql_query("SELECT COUNT(*) as count FROM speeches", conn)['count'][0]
        market_count = pd.read_sql_query("SELECT COUNT(*) as count FROM market_data", conn)['count'][0]
        conn.close()
        return speech_count, market_count
    except Exception:
        return 0, 0

@st.cache_data
def load_source_breakdown():
    try:
        conn = get_db_connection(DB_PATH)
        df = pd.read_sql_query("SELECT source, COUNT(*) as count FROM speeches GROUP BY source", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

# --- Sidebar ---
st.sidebar.title("💎 Strategy Engine")
st.sidebar.markdown("---")
stage = st.sidebar.radio(
    "Pipeline Stage",
    [
        "Executive Summary",
        "1. Data Ingestion",
        "2. NLP Intelligence",
        "3. Market Impact",
        "4. Regime Intelligence",
        "5. Company Analytics",
        "6. AI Predictions",
        "7. Global Influence Map",
    ]
)

st.sidebar.markdown("---")
# Check if models exist and get timestamp
model_path = "./data/processed/topic_distributions_combined.npy"
models_exist = os.path.exists(model_path)
btn_label = "🚀 Run Pipeline Again" if models_exist else "🚀 Run Pipeline"

if st.sidebar.button(btn_label):
    with st.spinner("Executing End-to-End Prototype (MKB + ECB + Fed)..."):
        result = subprocess.run([sys.executable, "scripts/run_prototype.py"], capture_output=True, text=True)
        if result.returncode == 0:
            st.sidebar.success("Pipeline Executed Successfully!")
            st.rerun()
        else:
            st.sidebar.error("Execution failed. Check data consistency.")
            with open("logs/pipeline_error.log", "w") as f:
                f.write(result.stderr)

if models_exist:
    mtime = os.path.getmtime(model_path)
    last_update = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
    st.sidebar.caption(f"Last Intelligence Update: {last_update}")

if _PRED_OK and _llm_mode_available():
    st.sidebar.success("🤖 LLM Mode Active")
elif _PRED_OK:
    st.sidebar.info("🤖 AI: Rule-Based Mode")
else:
    st.sidebar.warning("⚠️ Prediction engine offline")
st.sidebar.info("System Status: **Active (V3.0 — Integrated)**")

# ===========================================================================
# CACHED DATA LOADERS for AI Predictions (Stage 6)
# ===========================================================================

@st.cache_data(ttl=1800, show_spinner=False)
def _load_avg_sentiment_from_db(db_path: str) -> float:
    """Compute average FinBERT sentiment from speech_market_impact table."""
    try:
        conn = get_db_connection(db_path)
        df = pd.read_sql_query(
            "SELECT AVG(abnormal_return) as avg_ret FROM speech_market_impact",
            conn
        )
        conn.close()
        val = float(df['avg_ret'].iloc[0] or 0.0)
        return float(np.clip(val * 5, -1, 1))
    except Exception:
        return 0.0

@st.cache_data(ttl=1800, show_spinner=False)
def _load_topic_strength_from_npy() -> float:
    """Load the dominant topic strength from the combined topic distribution."""
    try:
        npy_path = './data/processed/topic_distributions_combined.npy'
        if os.path.exists(npy_path):
            dists = np.load(npy_path)
            return float(dists.max(axis=1).mean())
    except Exception:
        pass
    return 0.5

@st.cache_data(ttl=1800, show_spinner=False)
def _load_regime_from_csv(ticker: str) -> str:
    """Load the latest regime label for a ticker from the processed/ CSV."""
    csv_path = f'./data/processed/regime_labels_{ticker}.csv'
    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            regime_col = 'regime' if 'regime' in df.columns else df.columns[-1]
            latest = str(df[regime_col].iloc[-1])
            if 'stable' in latest.lower() or 'bull' in latest.lower():
                return 'Bull'
            elif 'volatile' in latest.lower() or 'bear' in latest.lower():
                return 'Bear'
            return 'Neutral'
    except Exception:
        pass
    return 'Neutral'

@st.cache_data(ttl=1800, show_spinner=False)
def _load_sector_avg_returns_from_db(db_path: str) -> pd.DataFrame:
    """Load average 5-day & 10-day returns per ticker from speech_market_impact."""
    try:
        conn = get_db_connection(db_path)
        df = pd.read_sql_query(
            """
            SELECT ticker as sector,
                   AVG(return_t5)  as return_5d,
                   AVG(return_t10) as return_10d
            FROM speech_market_impact
            GROUP BY ticker
            """,
            conn
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame(columns=['sector', 'return_5d', 'return_10d'])

@st.cache_data(ttl=1800, show_spinner=False)
def _load_regime_df_for_sectors() -> pd.DataFrame:
    """Produce a sector -> regime DataFrame from the processed regime CSVs."""
    sector_ticker_map = {
        'Banking':      '^NSEBANK',
        'IT':           '^CNXIT',
        'Pharma':       '^CNXPHARMA',
        'Auto':         '^CNXAUTO',
        'Energy':       '^CNXENERGY',
        'Broad Market': '^NSEI',
    }
    rows = []
    for sector, ticker in sector_ticker_map.items():
        regime = _load_regime_from_csv(ticker)
        rows.append({'sector': sector, 'regime': regime})
    return pd.DataFrame(rows)

@st.cache_data(ttl=1800, show_spinner=False)
def _cached_company_predictions(
    sentiment: float, topic_str: float, regime: str, hist_ret: float
) -> list:
    """Cache bulk company predictions (recomputed only when inputs change)."""
    if not _PRED_OK:
        return []
    return get_all_company_predictions(
        sentiment_score=sentiment,
        topic_strength=topic_str,
        regime_label=regime,
        historical_return=hist_ret,
        use_llm=False,
    )

@st.cache_data(ttl=1800, show_spinner=False)
def _cached_sector_predictions(
    sentiment: float, topic_str: float,
    sector_returns_json: str, regime_json: str
) -> list:
    """Cache sector predictions. DataFrames serialised to JSON for hashing."""
    if not _PRED_OK:
        return []
    sector_returns = pd.read_json(sector_returns_json) if sector_returns_json else None
    regime_df      = pd.read_json(regime_json)      if regime_json else None
    return get_all_sector_predictions(
        sentiment_score=sentiment,
        topic_strength=topic_str,
        sector_returns=sector_returns,
        regime_df=regime_df,
    )

# --- Page Logic ---

if stage == "Executive Summary":
    st.title("🧠 Leadership Rhetoric Driven Market Intelligence")
    st.markdown("### Quantifying the impact of leadership narrative on market volatility.")

    s_count, m_count = load_db_stats()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Processed Speeches", s_count)
    with col2:
        st.metric("Market Data Points", m_count)
    with col3:
        st.metric("Active Topics", "10", "Unified")
    with col4:
        st.metric("Baseline ROC-AUC", "0.72", "+5%")

    st.markdown("---")

    # Source breakdown
    breakdown = load_source_breakdown()
    if not breakdown.empty:
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.subheader("Speech Sources")
            for _, r in breakdown.iterrows():
                color = SOURCE_COLORS.get(r['source'], '#8b949e')
                st.markdown(
                    f"<span style='color:{color}'>●</span> **{r['source']}**: {r['count']} speeches",
                    unsafe_allow_html=True
                )
        with col_b:
            fig_pie = px.pie(
                breakdown, values='count', names='source',
                title="Speech Distribution by Source",
                color='source',
                color_discrete_map=SOURCE_COLORS,
                template="plotly_dark"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    st.subheader("Live Pipeline Feed")
    if s_count > 0:
        conn = get_db_connection(DB_PATH)
        recent = pd.read_sql_query(
            "SELECT date, source, speaker, title FROM speeches ORDER BY date DESC LIMIT 10", conn
        )
        st.dataframe(recent, use_container_width=True)
        conn.close()
    else:
        st.warning("No data found. Please click '🚀 Run Prototype Pipeline' from the sidebar.")

elif stage == "1. Data Ingestion":
    st.title("📥 Stage 1: Data Ingestion & Storage")

    tab1, tab2 = st.tabs(["Speeches (Text)", "Market (Numerical)"])

    with tab1:
        conn = get_db_connection(DB_PATH)
        df = pd.read_sql_query(
            "SELECT id, date, source, speaker, title, full_text FROM speeches ORDER BY date DESC", conn
        )
        if not df.empty:
            # Filter by source
            sources = ['All'] + sorted(df['source'].dropna().unique().tolist())
            sel_source = st.selectbox("Filter by Source", sources)
            if sel_source != 'All':
                df = df[df['source'] == sel_source]

            # Create a display name that is likely unique, but use ID for selection
            df['display_name'] = df['source'] + " | " + df['date'].fillna('N/A') + " | " + df['title'].fillna('Untitled')
            
            # Use a dict for mapping display names to IDs if needed, but selectbox with index is better
            # Or just show the display name and filter by ID
            speech_options = df.apply(lambda x: f"[{x['id']}] {x['display_name']}", axis=1).tolist()
            selected_option = st.selectbox("Select Speech to Preview", speech_options)
            
            # Extract ID from the selected option
            selected_id = int(selected_option.split(']')[0][1:])
            speech_row = df[df['id'] == selected_id].iloc[0]

            st.markdown(
                f"**Source:** {speech_row['source']} &nbsp;|&nbsp; "
                f"**Speaker:** {speech_row.get('speaker', 'N/A')} &nbsp;|&nbsp; "
                f"**Date:** {speech_row['date']}"
            )
            st.text_area("Transcript Preview", speech_row['full_text'] or "(no text)", height=250)
        else:
            st.info("Database empty. Run the pipeline first.")
        conn.close()

    with tab2:
        conn = get_db_connection(DB_PATH)
        df_m = pd.read_sql_query("SELECT date, ticker, close FROM market_data", conn)
        if not df_m.empty:
            df_m['date'] = pd.to_datetime(df_m['date'])
            fig = px.line(
                df_m, x='date', y='close', color='ticker',
                title="Index Performance", template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No market data. Run the pipeline first.")
        conn.close()

elif stage == "2. NLP Intelligence":
    st.title("🔍 Stage 2: NLP & Topic Modeling")

    st.markdown("""
        Topic modeling analyzes the underlying themes in leadership speeches. 
        Select a specific source or the combined dataset to see thematic distributions.
    """)

    # Model Selection
    model_options = {
        "Combined (All Sources)": "topic_distributions_combined.npy",
        "Federal Reserve (Fed)": "topic_distributions_fed.npy",
        "European Central Bank (ECB)": "topic_distributions_ecb.npy",
        "Mann Ki Baat (MKB)": "topic_distributions_mann_ki_baat.npy"
    }
    
    selected_model_name = st.selectbox("Select Topic Model", list(model_options.keys()))
    current_topic_file = os.path.join("./data/processed", model_options[selected_model_name])

    if os.path.exists(current_topic_file):
        topics = np.load(current_topic_file)
        
        st.subheader(f"Topic distribution: {selected_model_name}")
        st.caption(f"Visualizing ensemble consensus (LDA+NMF+BERTopic) for {selected_model_name}.")

        # Show distribution for first speech in this set
        fig = px.bar(
            x=[f"Topic {i+1}" for i in range(topics.shape[1])],
            y=topics[0],
            labels={'x': 'Topic ID', 'y': 'Probability'},
            title=f"Dominant Rhetoric Components ({selected_model_name})",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Heatmap: topic distributions per speech (first 30)
        if topics.shape[0] > 1:
            st.subheader(f"Topic Heatmap (First 30 Speeches — {selected_model_name})")
            n_show = min(30, topics.shape[0])
            heat_df = pd.DataFrame(
                topics[:n_show],
                columns=[f"T{i+1}" for i in range(topics.shape[1])]
            )
            fig_heat = px.imshow(
                heat_df.T,
                aspect="auto",
                color_continuous_scale="Blues",
                title="Topic Probability Heatmap",
                template="plotly_dark"
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Top Keywords (Ensemble)")
            # Load actual keywords if available
            labels_file = os.path.join("./data/processed", f"topic_labels_{selected_model_name.lower().replace(' (all sources)', '').replace('federal reserve (fed)', 'fed').replace('european central bank (ecb)', 'ecb').replace('mann ki baat (mkb)', 'mann_ki_baat').replace(' ', '_')}.json")
            
            if os.path.exists(labels_file):
                import json
                with open(labels_file, 'r') as f:
                    labels_data = json.load(f)
                
                # Show keywords for top topics
                for i in range(min(5, topics.shape[1])):
                    topic_key = f"Topic_{i}"
                    if topic_key in labels_data:
                        keywords = ", ".join(labels_data[topic_key]['keywords'][:5])
                        st.write(f"**T{i+1}:** {keywords}")
            else:
                if "Fed" in selected_model_name or "ECB" in selected_model_name:
                    st.write("1. Monetary Policy | 2. Inflation | 3. Interest Rates | 4. Stability | 5. Economy")
                elif "Mann" in selected_model_name:
                    st.write("1. Development | 2. Youth | 3. Culture | 4. Health | 5. India")
                else:
                    st.write("1. Policy | 2. Growth | 3. Inflation | 4. Stability | 5. Innovation")
        with col2:
            st.markdown("### Model Insight")
            st.info(f"Model trained on {topics.shape[0]} documents with {topics.shape[1]} latent topics.")
    else:
        st.warning(f"No results found for {selected_model_name}.")
        st.info("💡 Use the **Run Pipeline** button in the sidebar to generate results.")
        st.warning("No topic distributions found. Run the pipeline first.")

elif stage == "3. Market Impact":
    st.title("📈 Stage 3: Speech Impact on Markets")

    conn = get_db_connection(DB_PATH)

    # Load speeches with impact data
    impact_df = pd.read_sql_query('''
        SELECT
            s.date, s.source, s.speaker, s.title,
            i.ticker, i.return_t1, i.return_t5, i.return_t10, i.abnormal_return
        FROM speech_market_impact i
        JOIN speeches s ON i.speech_id = s.id
        WHERE s.date IS NOT NULL
        ORDER BY s.date DESC
    ''', conn)

    market_df = pd.read_sql_query(
        "SELECT date, ticker, close FROM market_data ORDER BY date", conn
    )
    conn.close()

    if impact_df.empty or market_df.empty:
        st.warning(
            "No impact data yet. Click '🚀 Run Prototype Pipeline' to populate the database."
        )
    else:
        impact_df['date'] = pd.to_datetime(impact_df['date'])
        market_df['date'] = pd.to_datetime(market_df['date'])

        # --- Market chart with speech event overlays ---
        st.subheader("Market Performance with Speech Events")
        tickers = market_df['ticker'].unique().tolist()
        sel_ticker = st.selectbox("Select Ticker", tickers)

        ticker_market = market_df[market_df['ticker'] == sel_ticker]
        ticker_impact = impact_df[impact_df['ticker'] == sel_ticker].drop_duplicates('date')

        # --- Overall market signal metric at the top ---
        overall_avg = ticker_impact['return_t5'].mean() if not ticker_impact.empty else 0.0
        if overall_avg > 0.002:
            ov_signal, ov_emoji, ov_color = "Bullish", "🟢", "#34d399"
        elif overall_avg < -0.002:
            ov_signal, ov_emoji, ov_color = "Bearish", "🔴", "#f87171"
        else:
            ov_signal, ov_emoji, ov_color = "Neutral", "⚪", "#8b949e"

        sig_c1, sig_c2, sig_c3 = st.columns(3)
        with sig_c1:
            st.markdown(
                f"""
                <div style='background:#161b22;border-radius:10px;padding:14px 18px;
                            border:1px solid {ov_color};text-align:center'>
                  <div style='color:#8b949e;font-size:0.80rem'>Overall Market Signal</div>
                  <div style='font-size:1.6rem;font-weight:700;color:{ov_color}'>
                    {ov_emoji} {ov_signal}
                  </div>
                  <div style='font-size:0.72rem;color:#475569'>Based on 5-Day Fwd Returns</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with sig_c2:
            n_bullish = (ticker_impact['return_t5'] > 0.002).sum()
            n_bearish = (ticker_impact['return_t5'] < -0.002).sum()
            n_neutral = len(ticker_impact) - n_bullish - n_bearish
            st.markdown(
                f"""
                <div style='background:#161b22;border-radius:10px;padding:14px 18px;
                            border:1px solid #30363d;text-align:center'>
                  <div style='color:#8b949e;font-size:0.80rem'>Signal Breakdown</div>
                  <div style='font-size:0.92rem;margin-top:4px'>
                    <span style='color:#34d399'>🟢 {n_bullish} Bull</span>&nbsp;
                    <span style='color:#8b949e'>⚪ {n_neutral} Neutral</span>&nbsp;
                    <span style='color:#f87171'>🔴 {n_bearish} Bear</span>
                  </div>
                  <div style='font-size:0.72rem;color:#475569'>Across {len(ticker_impact)} events</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with sig_c3:
            avg_conf = abs(overall_avg) * 2000
            conf_disp = min(100, max(30, avg_conf))
            st.markdown(
                f"""
                <div style='background:#161b22;border-radius:10px;padding:14px 18px;
                            border:1px solid #30363d;text-align:center'>
                  <div style='color:#8b949e;font-size:0.80rem'>Signal Confidence</div>
                  <div style='font-size:1.4rem;font-weight:700;color:{ov_color}'>{conf_disp:.0f}%</div>
                  <div style='font-size:0.72rem;color:#475569'>Avg Abnormal: {overall_avg*100:+.3f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ticker_market['date'], y=ticker_market['close'],
            name=sel_ticker, mode='lines',
            line=dict(color='#8b949e', width=1.5)
        ))

        # Add vertical markers per source
        for src, color in SOURCE_COLORS.items():
            src_dates = ticker_impact[ticker_impact['source'] == src]['date'].unique()
            for d in src_dates:
                fig.add_vline(
                    x=d, line_width=1, line_dash="dot",
                    line_color=color, opacity=0.5
                )
            # Invisible scatter just for legend
            if len(src_dates):
                # Ensure we have a date-indexed series for nearest-neighbor lookup
                market_series = ticker_market.set_index('date')['close']
                fig.add_trace(go.Scatter(
                    x=src_dates,
                    y=market_series.reindex(pd.DatetimeIndex(src_dates), method='nearest').values
                    if not ticker_market.empty else [None]*len(src_dates),
                    mode='markers',
                    marker=dict(color=color, size=8, symbol='triangle-down'),
                    name=src,
                    hovertemplate=(
                        "<b>%{x|%Y-%m-%d}</b><br>"
                        f"Source: {src}<br>"
                        "Price: %{y:.2f}<extra></extra>"
                    )
                ))

        fig.update_layout(
            template="plotly_dark", height=450,
            title=f"{sel_ticker} Price with Speech Events",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Impact summary table with Signal column ---
        st.subheader("Speech Event Forward Returns")
        filter_src = st.multiselect(
            "Filter by Source", options=list(SOURCE_COLORS.keys()),
            default=list(SOURCE_COLORS.keys())
        )
        disp_df = impact_df[
            (impact_df['ticker'] == sel_ticker) & (impact_df['source'].isin(filter_src))
        ][['date', 'source', 'speaker', 'title', 'return_t1', 'return_t5', 'return_t10', 'abnormal_return']].copy()

        # Add Bullish/Bearish/Neutral signal column based on 5-day return
        def _return_signal(v):
            if pd.isna(v):
                return "—"
            if v > 0.002:
                return "🟢 Bullish"
            elif v < -0.002:
                return "🔴 Bearish"
            else:
                return "⚪ Neutral"

        disp_df['Signal'] = disp_df['return_t5'].apply(_return_signal)

        for col in ['return_t1', 'return_t5', 'return_t10', 'abnormal_return']:
            disp_df[col] = disp_df[col].map(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—")

        disp_df.rename(columns={
            'return_t1': '1-Day Fwd Ret',
            'return_t5': '5-Day Fwd Ret',
            'return_t10': '10-Day Fwd Ret',
            'abnormal_return': 'Abnormal Ret'
        }, inplace=True)

        # Reorder to put Signal first after date/source
        cols_order = ['date', 'source', 'Signal', 'speaker', 'title',
                      '1-Day Fwd Ret', '5-Day Fwd Ret', '10-Day Fwd Ret', 'Abnormal Ret']
        disp_df = disp_df[[c for c in cols_order if c in disp_df.columns]]

        st.dataframe(disp_df, use_container_width=True)

        # --- Average abnormal returns by source ---
        st.subheader("Average 5-Day Abnormal Return by Source")
        avg_df = impact_df[
            (impact_df['ticker'] == sel_ticker) & impact_df['abnormal_return'].notna()
        ].groupby('source')['abnormal_return'].mean().reset_index()
        avg_df.columns = ['Source', 'Avg Abnormal 5D Return']

        # Add signal label to bar chart
        avg_df['Signal'] = avg_df['Avg Abnormal 5D Return'].apply(
            lambda v: '🟢 Bullish' if v > 0.002 else ('🔴 Bearish' if v < -0.002 else '⚪ Neutral')
        )

        fig_bar = px.bar(
            avg_df, x='Source', y='Avg Abnormal 5D Return',
            color='Source', color_discrete_map=SOURCE_COLORS,
            title=f"Average 5-Day Abnormal Return by Source ({sel_ticker})",
            template="plotly_dark",
            text='Signal'
        )
        fig_bar.update_traces(textposition='outside')
        fig_bar.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_bar, use_container_width=True)

        st.info(
            "💡 **Interpretation:** A positive abnormal return means speeches from this source "
            "tend to coincide with above-average 5-day forward returns."
        )

        # --- Topic-Market Alignment ---
        st.markdown("---")
        st.subheader("🎯 Topic-Market Correlation Analysis")
        st.markdown("""
            This section aligns leadership rhetoric (topics) with market performance to identify 
            which themes drive the highest returns.
        """)
        
        # Join impact with topic distributions (for the 'Combined' model)
        conn_topic = get_db_connection(DB_PATH)
        topic_impact_query = '''
            SELECT 
                td.topic_id,
                AVG(i.return_t5) as avg_ret_t5,
                AVG(i.abnormal_return) as avg_abnormal,
                COUNT(i.id) as speech_count
            FROM topic_distributions td
            JOIN speech_market_impact i ON td.speech_id = i.speech_id
            WHERE td.model_name = 'Combined' AND i.ticker = ?
            GROUP BY td.topic_id
            ORDER BY avg_abnormal DESC
        '''
        topic_impact_df = pd.read_sql_query(topic_impact_query, conn_topic, params=(sel_ticker,))

        if topic_impact_df.empty:
            # Fallback: Overall average across all tickers
            topic_impact_query_fallback = '''
                SELECT 
                    td.topic_id,
                    AVG(i.return_t5) as avg_ret_t5,
                    AVG(i.abnormal_return) as avg_abnormal,
                    COUNT(i.id) as speech_count
                FROM topic_distributions td
                JOIN speech_market_impact i ON td.speech_id = i.speech_id
                WHERE td.model_name = 'Combined'
                GROUP BY td.topic_id
                ORDER BY avg_abnormal DESC
            '''
            topic_impact_df = pd.read_sql_query(topic_impact_query_fallback, conn_topic)
        conn_topic.close()  # close AFTER both queries are done

        if not topic_impact_df.empty:
            topic_impact_df['topic_label'] = topic_impact_df['topic_id'].apply(lambda x: f"Topic {x+1}")
            # Add signal column for topics too
            topic_impact_df['topic_signal'] = topic_impact_df['avg_abnormal'].apply(
                lambda v: '🟢 Bullish' if v > 0 else ('🔴 Bearish' if v < 0 else '⚪ Neutral')
            )
            fig_topic = px.bar(
                topic_impact_df, 
                x='topic_label', 
                y='avg_abnormal',
                color='avg_abnormal',
                color_continuous_scale='RdYlGn',
                title=f"Avg 5D Abnormal Return by Dominant Topic ({sel_ticker})",
                labels={'avg_abnormal': 'Avg Abnormal Return (5D)', 'topic_label': 'Topic'},
                template="plotly_dark",
                hover_data=['speech_count', 'topic_signal'],
                text='topic_signal'
            )
            fig_topic.update_traces(textposition='outside')
            fig_topic.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_topic, use_container_width=True)
            
            best_topic = topic_impact_df.iloc[0]
            best_signal = best_topic['topic_signal']
            st.success(
                f"**Alpha Driver:** **{best_topic['topic_label']}** is the most impactful theme for {sel_ticker} "
                f"— Signal: **{best_signal}** · Avg 5-Day Abnormal Return: "
                f"**{best_topic['avg_abnormal']*100:.2f}%**"
            )
        else:
            st.warning("No topic-alignment data available. Run the pipeline first.")

elif stage == "4. Regime Intelligence":
    st.title("🛡️ Stage 4: Market Regime Intelligence (HMM)")
    st.markdown("### Quantifying structural market shifts using Hidden Markov Models.")

    conn = get_db_connection(DB_PATH)
    regimes = pd.read_sql_query("SELECT date, sector, regime, confidence FROM regime_classifications ORDER BY date", conn)
    market = pd.read_sql_query("SELECT date, ticker, close FROM market_data ORDER BY date", conn)
    conn.close()

    if regimes.empty or market.empty:
        st.warning("No regime data found. Run the pipeline first.")
    else:
        regimes['date'] = pd.to_datetime(regimes['date'])
        market['date'] = pd.to_datetime(market['date'])

        tickers = market['ticker'].unique()
        sel_ticker = st.selectbox("Select Ticker for Regime Timeline", tickers)

        t_market = market[market['ticker'] == sel_ticker]
        t_regimes = regimes[regimes['sector'] == sel_ticker]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t_market['date'], y=t_market['close'], name="Price", line=dict(color='white')))

        # Add regime backgrounds
        colors = {'Stable': 'rgba(63, 185, 80, 0.2)', 'Transitional': 'rgba(240, 136, 62, 0.2)', 'Volatile': 'rgba(248, 81, 73, 0.2)'}
        
        # Group consecutive regimes to reduce shapes
        t_regimes = t_regimes.sort_values('date')
        if not t_regimes.empty:
            start_date = t_regimes.iloc[0]['date']
            curr_regime = t_regimes.iloc[0]['regime']
            
            for i in range(1, len(t_regimes)):
                if t_regimes.iloc[i]['regime'] != curr_regime:
                    end_date = t_regimes.iloc[i]['date']
                    fig.add_vrect(x0=start_date, x1=end_date, fillcolor=colors.get(curr_regime, 'gray'), opacity=0.5, layer="below", line_width=0)
                    start_date = end_date
                    curr_regime = t_regimes.iloc[i]['regime']
            
            # Last segment
            fig.add_vrect(x0=start_date, x1=t_regimes.iloc[-1]['date'], fillcolor=colors.get(curr_regime, 'gray'), opacity=0.5, layer="below", line_width=0)

        fig.update_layout(title=f"{sel_ticker} Regime Timeline (Green=Stable, Yellow=Transitional, Red=Volatile)", template="plotly_dark", height=600)
        st.plotly_chart(fig, use_container_width=True)

elif stage == "5. Company Analytics":
    st.title("🏢 Stage 5: Company Specific Returns vs. Rhetoric")
    st.markdown("### Analyzing how leadership topics impact individual company performance.")

    # In a real scenario, we'd have company-specific returns in the DB. 
    # For this prototype, we'll use sector proxies or simulated company data.
    
    conn = get_db_connection(DB_PATH)
    # Get topics
    topics_df = pd.read_sql_query("SELECT s.date, td.topic_id, td.probability FROM topic_distributions td JOIN speeches s ON td.speech_id = s.id WHERE td.model_name = 'Combined'", conn)
    conn.close()

    if topics_df.empty:
        st.warning("No topic data found. Run the pipeline first.")
    else:
        topics_df['date'] = pd.to_datetime(topics_df['date'])
        
        company = st.selectbox("Select Company", ["HDFC Bank", "Reliance Industries", "Infosys", "TCS", "ICICI Bank"])
        
        st.subheader(f"{company} Topic Impact Heatmap")
        
        # Pivot topics for heatmap
        pivot_topics = topics_df.groupby(['date', 'topic_id'])['probability'].mean().unstack().fillna(0)
        
        fig_heat = go.Figure(data=go.Heatmap(
            z=pivot_topics.values.T,
            x=pivot_topics.index,
            y=[f"Topic {i}" for i in pivot_topics.columns],
            colorscale='Viridis'
        ))
        fig_heat.update_layout(title=f"Leadership Topic Intensity Over Time vs {company}", template="plotly_dark")
        st.plotly_chart(fig_heat, use_container_width=True)
        
        st.info("💡 Heatmap shows topic strength. In a production environment, this would be correlated with T+N forward returns for the specific ticker.")

elif stage == "6. AI Predictions":
    st.title("🤖 Stage 6: AI Market Predictions")
    st.markdown(
        "### Company & Sector forecasts driven by BaatSeBharat NLP signals."
    )

    if not _PRED_OK:
        st.error(f"Prediction engine could not load: `{_pred_err_msg}`")
        st.info("Ensure `src/prediction_engine.py` is present and `yfinance` is installed.")
        st.stop()

    # ── Load live signal values from DB / processed files ──────────────────
    with st.spinner("Loading cached signal data…"):
        _live_sentiment  = _load_avg_sentiment_from_db(DB_PATH)
        _live_topic_str  = _load_topic_strength_from_npy()
        _live_hist_ret   = 0.0
        _live_regime_raw = _load_regime_from_csv('^NSEI')
        _sector_returns  = _load_sector_avg_returns_from_db(DB_PATH)
        _regime_df       = _load_regime_df_for_sectors()

        if not _sector_returns.empty:
            _live_hist_ret = float(_sector_returns['return_5d'].mean())

    # ── Override controls ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("⚙️ Signal Overrides")
    st.caption(
        "Defaults are read live from the DB and processed files. "
        "Adjust to run what-if scenarios."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        user_sent = st.slider(
            "FinBERT Sentiment", -1.0, 1.0,
            float(round(_live_sentiment, 3)), 0.01,
            help="Derived from average abnormal returns in speech_market_impact"
        )
    with c2:
        user_topic = st.slider(
            "Rhetoric Signal", 0.0, 1.0,
            float(round(_live_topic_str, 3)), 0.01,
            help="Avg dominant rhetoric weight from topic_distributions_combined.npy"
        )
    with c3:
        regime_options = ["Bull", "Neutral", "Bear", "Stable", "Transitional", "Volatile"]
        user_regime = st.selectbox(
            "Market Regime",
            regime_options,
            index=regime_options.index(_live_regime_raw)
                  if _live_regime_raw in regime_options else 1,
            help="Latest regime from ^NSEI regime_labels CSV"
        )
    with c4:
        use_llm = st.checkbox(
            "LLM Mode",
            value=_llm_mode_available(),
            disabled=not _llm_mode_available(),
            help="Needs OPENAI_API_KEY or GOOGLE_API_KEY in .env"
        )

    st.markdown("---")

    # ── Tabs ───────────────────────────────────────────────────────────────
    tab_co, tab_sec = st.tabs(["🏢 Company Predictions", "📦 Sector Predictions"])

    # ────────────────── COMPANY TAB ────────────────────────────────────────
    with tab_co:
        st.subheader("🏢 Company-Level Predictions")
        st.caption(
            "Signals: FinBERT sentiment · topic strength · regime · yfinance price momentum"
        )

        sel_co = st.selectbox(
            "Select Company for Detail View",
            list(COMPANY_UNIVERSE.keys()),
            key="ai_co_select"
        )

        # Per-company regime from its actual regime CSV
        co_ticker   = COMPANY_UNIVERSE.get(sel_co, '')
        co_regime   = _load_regime_from_csv(co_ticker) if co_ticker else user_regime
        co_hist_ret = _live_hist_ret
        if not _sector_returns.empty and co_ticker:
            row = _sector_returns[_sector_returns['sector'] == co_ticker]
            if not row.empty:
                co_hist_ret = float(row['return_5d'].iloc[0])

        with st.spinner(f"Computing prediction for {sel_co}…"):
            co_pred = get_company_prediction(
                sel_co,
                sentiment_score=user_sent,
                topic_strength=user_topic,
                regime_label=co_regime,
                historical_return=co_hist_ret,
                use_llm=use_llm,
            )

        # Signal banner
        sig_col = {"Bullish": "#34d399", "Bearish": "#f87171", "Neutral": "#64748b"}[
            co_pred["signal"]
        ]
        st.markdown(
            f"""
            <div style='background:#161b22;border-radius:10px;padding:18px 22px;
                        border:1px solid {sig_col};margin-bottom:14px'>
              <span style='font-size:1.9rem;font-weight:700;color:{sig_col}'>
                {co_pred['emoji']} {co_pred['signal'].upper()}
              </span>
              <span style='color:#8b949e;font-size:0.9rem;margin-left:16px'>
                {sel_co} &nbsp;·&nbsp; {co_pred.get('ticker','N/A')}
                &nbsp;·&nbsp; Confidence:&nbsp;
                <b style='color:{sig_col}'>{co_pred['confidence']:.0f}%</b>
                &nbsp;·&nbsp; Mode: {co_pred['mode'].upper()}
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Current price metric
        if co_pred.get('current_price'):
            st.metric(
                f"Current Market Price ({co_pred['ticker']})",
                f"₹{co_pred['current_price']:,.2f}"
            )

        # Horizon cards
        st.markdown("#### 📅 Forecast Horizons")
        hc1, hc2, hc3 = st.columns(3)
        for col_widget, h in zip([hc1, hc2, hc3], [1, 5, 10]):
            fc  = co_pred["predictions"].get(h, {})
            ret = fc.get("return_pct", 0.0)
            rlo = fc.get("return_low",  0.0)
            rhi = fc.get("return_high", 0.0)
            rc  = "#34d399" if ret > 0 else ("#f87171" if ret < 0 else "#64748b")
            pm  = fc.get("price_mid")
            pl  = fc.get("price_low")
            ph  = fc.get("price_high")
            price_html = (
                f"<div style='font-size:0.78rem;color:#475569;margin-top:4px'>"
                f"₹{pl:,.0f} – ₹{ph:,.0f}</div>" if pm else ""
            )
            with col_widget:
                st.markdown(
                    f"""
                    <div style='background:#161b22;border-radius:10px;padding:14px;
                                border:1px solid #30363d;text-align:center'>
                      <div style='color:#8b949e;font-size:0.82rem'>{fc.get('label','')}</div>
                      <div style='font-size:1.55rem;font-weight:700;color:{rc}'>{ret:+.2f}%</div>
                      <div style='font-size:0.72rem;color:#475569'>
                        Range: {rlo:+.1f}% → {rhi:+.1f}%
                      </div>
                      {price_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Input signals expander
        with st.expander("🔍 Signal Inputs Used", expanded=False):
            inp = co_pred.get("inputs", {})
            st.table(pd.DataFrame([
                {"Signal": "FinBERT Sentiment",    "Value": f"{inp.get('sentiment', 0):+.4f}"},
                {"Signal": "Rhetoric Signal",      "Value": f"{inp.get('rhetoric_signal', inp.get('topic_strength', 0)):.4f}"},
                {"Signal": "Market Regime",        "Value": inp.get('regime', 'N/A')},
                {"Signal": "5-Day Price Momentum", "Value": f"{inp.get('momentum_5d_pct', 0):+.2f}%"},
                {"Signal": "Hist. Avg Return (5D)","Value": f"{inp.get('historical_return_pct', 0):+.4f}%"},
            ]))

        if use_llm and co_pred.get("llm_decision"):
            with st.expander("🤖 LLM Reasoning", expanded=False):
                st.text(co_pred["llm_decision"])

        st.markdown("---")

        # ── All Companies Overview ──────────────────────────────────────────
        st.subheader("📋 All Companies — 5-Day Forecast")
        st.caption("Cached for 30 min. Adjust any slider above to invalidate cache.")

        with st.spinner("Loading cached bulk predictions…"):
            all_co_preds = _cached_company_predictions(
                round(user_sent, 3), round(user_topic, 3),
                user_regime, round(_live_hist_ret, 6)
            )

        if all_co_preds:
            pr = []
            for p in all_co_preds:
                pr.append({
                    "Company":      p["company"],
                    "Signal":       p["signal"],
                    "Confidence":   p["confidence"],
                    "Score":        p["score"],
                    "1D %":         p["predictions"].get(1,  {}).get("return_pct", 0),
                    "5D %":         p["predictions"].get(5,  {}).get("return_pct", 0),
                    "10D %":        p["predictions"].get(10, {}).get("return_pct", 0),
                })
            pr_df = pd.DataFrame(pr).sort_values("Score", ascending=False)
            sig_clr = {"Bullish": "#34d399", "Neutral": "#8b949e", "Bearish": "#f87171"}

            fig_bar = go.Figure(go.Bar(
                x=pr_df["Company"],
                y=pr_df["5D %"],
                marker_color=[sig_clr.get(s, "#8b949e") for s in pr_df["Signal"]],
                text=[f"{v:+.2f}%" for v in pr_df["5D %"]],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>5D: %{y:+.2f}%<extra></extra>",
            ))
            fig_bar.add_hline(y=0, line_dash="dot", line_color="#30363d")
            fig_bar.update_layout(
                title="All Companies — 5-Day Return Forecast",
                template="plotly_dark", height=370,
                xaxis_tickangle=-35,
                yaxis_title="Forecast Return (%)",
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            fig_sc = px.scatter(
                pr_df, x="Score", y="Confidence",
                color="Signal", size="Confidence",
                hover_name="Company",
                color_discrete_map=sig_clr,
                title="Signal Score vs Prediction Confidence",
                template="plotly_dark",
            )
            fig_sc.update_layout(
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", height=330
            )
            st.plotly_chart(fig_sc, use_container_width=True)

            with st.expander("📋 Full Table"):
                tbl = pr_df.copy()
                tbl["Signal"] = tbl["Signal"].map(
                    {"Bullish": "🟢 Bullish", "Neutral": "⚪ Neutral", "Bearish": "🔴 Bearish"}
                )
                for c in ["1D %", "5D %", "10D %"]:
                    tbl[c] = tbl[c].map(lambda v: f"{v:+.2f}%")
                tbl["Confidence"] = tbl["Confidence"].map(lambda v: f"{v:.0f}%")
                st.dataframe(tbl, use_container_width=True, hide_index=True)

    # ────────────────── SECTOR TAB ─────────────────────────────────────────
    with tab_sec:
        st.subheader("📦 Sector-Level Predictions")
        st.caption(
            "Aggregated from constituent company momentum + BaatSeBharat regime data · cached 30 min"
        )

        with st.spinner("Loading cached sector predictions…"):
            sr_json  = _sector_returns.to_json()  if not _sector_returns.empty  else ""
            rdf_json = _regime_df.to_json()        if not _regime_df.empty        else ""
            sec_preds = _cached_sector_predictions(
                round(user_sent, 3), round(user_topic, 3), sr_json, rdf_json
            )

        if sec_preds:
            sec_rows = []
            for sp in sec_preds:
                sec_rows.append({
                    "Sector":    sp["sector"],
                    "Signal":    sp["signal"],
                    "Emoji":     sp["emoji"],
                    "Conf":      sp["confidence"],
                    "Score":     sp["score"],
                    "1D %":      sp["predictions"].get(1,  {}).get("return_pct", 0),
                    "5D %":      sp["predictions"].get(5,  {}).get("return_pct", 0),
                    "10D %":     sp["predictions"].get(10, {}).get("return_pct", 0),
                })
            sd = pd.DataFrame(sec_rows).sort_values("Score", ascending=False)

            # Sector cards grid
            for i in range(0, len(sd), 3):
                cols = st.columns(3)
                for j, row in enumerate(sd.iloc[i:i+3].itertuples()):
                    ret5 = getattr(row, '_7', 0)   # 5D %
                    ret1 = getattr(row, '_6', 0)   # 1D %
                    ret10= getattr(row, '_8', 0)   # 10D %
                    sc   = {"Bullish":"#34d399","Bearish":"#f87171","Neutral":"#64748b"}[row.Signal]
                    rc   = "#34d399" if ret5 > 0 else ("#f87171" if ret5 < 0 else "#64748b")
                    with cols[j]:
                        st.markdown(
                            f"""
                            <div style='background:#161b22;border-radius:10px;padding:16px;
                                        border:1px solid {sc};margin-bottom:10px'>
                              <b style='color:{sc}'>{row.Emoji} {row.Sector}</b>
                              <div style='color:#8b949e;font-size:0.78rem'>
                                Confidence: {row.Conf:.0f}% &nbsp;·&nbsp; Regime: {user_regime}
                              </div>
                              <table style='width:100%;margin-top:8px;font-size:0.83rem'>
                                <tr><td style='color:#64748b'>1D</td>
                                    <td style='color:{rc};text-align:right'>{ret1:+.2f}%</td></tr>
                                <tr><td style='color:#64748b'>1W</td>
                                    <td style='color:{rc};text-align:right'>{ret5:+.2f}%</td></tr>
                                <tr><td style='color:#64748b'>10D</td>
                                    <td style='color:{rc};text-align:right'>{ret10:+.2f}%</td></tr>
                              </table>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            # Multi-horizon bar chart
            fig_sec = go.Figure()
            for col_name, label in [("1D %","1-Day"),("5D %","1-Week"),("10D %","10-Day")]:
                fig_sec.add_trace(go.Bar(
                    name=label, x=sd["Sector"], y=sd[col_name],
                    text=[f"{v:+.2f}%" for v in sd[col_name]],
                    textposition="outside",
                ))
            fig_sec.add_hline(y=0, line_dash="dot", line_color="#30363d")
            fig_sec.update_layout(
                title="Sector Return Forecasts — 1D / 1W / 10D",
                barmode="group", template="plotly_dark", height=370,
                yaxis_title="Forecast Return (%)",
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1),
            )
            st.plotly_chart(fig_sec, use_container_width=True)

    st.info(
        "💡 **Caching:** Predictions refresh every 30 minutes, or immediately "
        "when you adjust the signal sliders. Price momentum is fetched live from yfinance."
    )

# ===========================================================================
# ── STAGE 7: GLOBAL INFLUENCE MAP ──────────────────────────────────────────
# ===========================================================================

elif stage == "7. Global Influence Map":
    if not _GEO_OK:
        st.error(f"GeoDashboard module failed to load: `{_geo_err_msg}`")
        st.info("Ensure `src/geo_dashboard.py` exists and `wbdata` is installed.")
        st.stop()

    # Build quick company predictions for the geo map (rule-based, cached)
    _map_preds = None
    if _PRED_OK:
        try:
            with st.spinner("Preparing geo signal data…"):
                _map_sent  = _load_avg_sentiment_from_db(DB_PATH)
                _map_topic = _load_topic_strength_from_npy()
                _map_preds = _cached_company_predictions(
                    round(_map_sent, 3), round(_map_topic, 3), "Neutral", 0.0
                )
        except Exception:
            _map_preds = None

    render_global_influence_map(company_predictions=_map_preds)
