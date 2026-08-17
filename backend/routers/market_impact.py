"""
Stage 3 — Speech Impact on Markets, ported from App.py's
`elif stage == "3. Market Impact":` block.
"""
import os
import json
import numpy as np
import pandas as pd
from fastapi import APIRouter
from typing import Optional, List

from utils.db_utils import get_db_connection
from backend.cache import ttl_cache
from backend.colors import apply_chart_theme, COLORS, SOURCE_COLORS, DIVERGING_SCALE
from backend.jsonsafe import records

router = APIRouter()

DB_PATH = './data/market_rhetoric.db'


@ttl_cache(ttl_seconds=1800)
def _load_stage3_data(db_path):
    conn = get_db_connection(db_path)
    impact = pd.read_sql_query('''
        SELECT
            s.date, s.source, s.speaker, s.title,
            i.ticker, i.return_t1, i.return_t5, i.return_t10, i.abnormal_return
        FROM speech_market_impact i
        JOIN speeches s ON i.speech_id = s.id
        WHERE s.date IS NOT NULL
        ORDER BY s.date DESC
    ''', conn)
    market = pd.read_sql_query(
        "SELECT date, ticker, close FROM market_data ORDER BY date", conn
    )
    conn.close()
    return impact, market


@ttl_cache(ttl_seconds=1800)
def _load_topic_impact(db_path, ticker):
    conn_topic = get_db_connection(db_path)
    topic_impact_query = '''
        SELECT
            td.topic_id,
            SUM(td.probability * i.return_t5) / NULLIF(SUM(td.probability), 0) as avg_ret_t5,
            SUM(td.probability * i.abnormal_return) / NULLIF(SUM(td.probability), 0) as avg_abnormal,
            COUNT(DISTINCT i.id) as speech_count
        FROM topic_distributions td
        JOIN speech_market_impact i ON td.speech_id = i.speech_id
        WHERE td.model_name = 'Combined' AND i.ticker = ?
        GROUP BY td.topic_id
        ORDER BY avg_abnormal DESC
    '''
    result = pd.read_sql_query(topic_impact_query, conn_topic, params=(ticker,))

    if result.empty:
        topic_impact_query_fallback = '''
            SELECT
                td.topic_id,
                SUM(td.probability * i.return_t5) / NULLIF(SUM(td.probability), 0) as avg_ret_t5,
                SUM(td.probability * i.abnormal_return) / NULLIF(SUM(td.probability), 0) as avg_abnormal,
                COUNT(DISTINCT i.id) as speech_count
            FROM topic_distributions td
            JOIN speech_market_impact i ON td.speech_id = i.speech_id
            WHERE td.model_name = 'Combined'
            GROUP BY td.topic_id
            ORDER BY avg_abnormal DESC
        '''
        result = pd.read_sql_query(topic_impact_query_fallback, conn_topic)
    conn_topic.close()
    return result


def _return_signal(v):
    if pd.isna(v):
        return "—"
    if v > 0.002:
        return "🟢 Bullish"
    elif v < -0.002:
        return "🔴 Bearish"
    else:
        return "⚪ Neutral"


@router.get("/market-impact/tickers")
def list_tickers():
    _, market_df = _load_stage3_data(DB_PATH)
    return {"tickers": market_df['ticker'].unique().tolist() if not market_df.empty else []}


@router.get("/market-impact")
def get_market_impact(ticker: Optional[str] = None, sources: Optional[str] = None):
    """sources: comma-separated list; defaults to all of SOURCE_COLORS."""
    import plotly.graph_objects as go
    import plotly.express as px

    impact_df, market_df = _load_stage3_data(DB_PATH)
    if impact_df.empty or market_df.empty:
        return {"empty": True}

    impact_df = impact_df.copy()
    market_df = market_df.copy()
    impact_df['date'] = pd.to_datetime(impact_df['date'])
    market_df['date'] = pd.to_datetime(market_df['date'])

    tickers = market_df['ticker'].unique().tolist()
    sel_ticker = ticker if ticker in tickers else (tickers[0] if tickers else None)
    if sel_ticker is None:
        return {"empty": True}

    filter_src: List[str] = sources.split(",") if sources else list(SOURCE_COLORS.keys())

    ticker_market = market_df[market_df['ticker'] == sel_ticker]
    ticker_impact = impact_df[impact_df['ticker'] == sel_ticker].drop_duplicates('date')

    overall_avg = float(ticker_impact['return_t5'].mean()) if not ticker_impact.empty else 0.0
    if overall_avg > 0.002:
        ov_signal, ov_emoji, ov_color = "Bullish", "🟢", COLORS["green"]
    elif overall_avg < -0.002:
        ov_signal, ov_emoji, ov_color = "Bearish", "🔴", COLORS["rust"]
    else:
        ov_signal, ov_emoji, ov_color = "Neutral", "⚪", COLORS["ink_dim"]

    n_bullish = int((ticker_impact['return_t5'] > 0.002).sum())
    n_bearish = int((ticker_impact['return_t5'] < -0.002).sum())
    n_neutral = len(ticker_impact) - n_bullish - n_bearish
    avg_conf = abs(overall_avg) * 2000
    conf_disp = min(100, max(30, avg_conf))

    # --- Price chart with speech-event overlays ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ticker_market['date'], y=ticker_market['close'],
        name=sel_ticker, mode='lines',
        line=dict(color=COLORS["ink"], width=1.8)
    ))

    y_lo = float(ticker_market['close'].min()) if not ticker_market.empty else 0.0
    y_hi = float(ticker_market['close'].max()) if not ticker_market.empty else 1.0
    for src, color in SOURCE_COLORS.items():
        src_dates = ticker_impact[ticker_impact['source'] == src]['date'].unique()
        if len(src_dates):
            line_x, line_y = [], []
            for d in src_dates:
                line_x.extend([d, d, None])
                line_y.extend([y_lo, y_hi, None])
            fig.add_trace(go.Scatter(
                x=line_x, y=line_y, mode='lines',
                line=dict(color=color, width=1, dash='dot'),
                opacity=0.22, showlegend=False, hoverinfo='skip',
            ))
        if len(src_dates):
            market_series = ticker_market.set_index('date')['close']
            fig.add_trace(go.Scatter(
                x=src_dates,
                y=market_series.reindex(pd.DatetimeIndex(src_dates), method='nearest').values
                if not ticker_market.empty else [None] * len(src_dates),
                mode='markers',
                marker=dict(color=color, size=8, symbol='triangle-down'),
                name=src,
                hovertemplate=(
                    "<b>%{x|%Y-%m-%d}</b><br>"
                    f"Source: {src}<br>"
                    "Price: %{y:.2f}<extra></extra>"
                )
            ))

    apply_chart_theme(fig, height=450)
    fig.update_layout(
        title=f"{sel_ticker} Price with Speech Events",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # --- Event table ---
    disp_df = impact_df[
        (impact_df['ticker'] == sel_ticker) & (impact_df['source'].isin(filter_src))
    ][['date', 'source', 'speaker', 'title', 'return_t1', 'return_t5', 'return_t10', 'abnormal_return']].copy()
    disp_df['Signal'] = disp_df['return_t5'].apply(_return_signal)
    for col in ['return_t1', 'return_t5', 'return_t10', 'abnormal_return']:
        disp_df[col] = disp_df[col].map(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—")
    disp_df.rename(columns={
        'return_t1': '1-Day Fwd Ret', 'return_t5': '5-Day Fwd Ret',
        'return_t10': '10-Day Fwd Ret', 'abnormal_return': 'Abnormal Ret'
    }, inplace=True)
    cols_order = ['date', 'source', 'Signal', 'speaker', 'title',
                  '1-Day Fwd Ret', '5-Day Fwd Ret', '10-Day Fwd Ret', 'Abnormal Ret']
    disp_df = disp_df[[c for c in cols_order if c in disp_df.columns]]
    disp_df['date'] = disp_df['date'].astype(str)

    # --- Avg abnormal return by source bar ---
    avg_df = impact_df[
        (impact_df['ticker'] == sel_ticker) & impact_df['abnormal_return'].notna()
    ].groupby('source')['abnormal_return'].mean().reset_index()
    avg_df.columns = ['Source', 'Avg Abnormal 5D Return']
    avg_df['Signal'] = avg_df['Avg Abnormal 5D Return'].apply(
        lambda v: '🟢 Bullish' if v > 0.002 else ('🔴 Bearish' if v < -0.002 else '⚪ Neutral')
    )
    fig_bar = px.bar(
        avg_df, x='Source', y='Avg Abnormal 5D Return',
        color='Source', color_discrete_map=SOURCE_COLORS,
        title=f"Average 5-Day Abnormal Return by Source ({sel_ticker})",
        text='Signal'
    )
    fig_bar.update_traces(textposition='outside')
    fig_bar.add_hline(y=0, line_dash="dash", line_color=COLORS["line"])
    apply_chart_theme(fig_bar, height=380)

    # --- Topic-market correlation ---
    topic_impact_df = _load_topic_impact(DB_PATH, sel_ticker)
    topic_fig = None
    alpha_driver = None
    if not topic_impact_df.empty:
        topic_impact_df = topic_impact_df.copy()
        labels_path = "./data/processed/topic_labels_combined.json"
        topic_label_map = {}
        if os.path.exists(labels_path):
            with open(labels_path, 'r', encoding='utf-8') as f:
                topic_label_map = {int(k): v['label'] for k, v in json.load(f).items()}
        topic_impact_df['topic_label'] = topic_impact_df['topic_id'].apply(
            lambda x: topic_label_map.get(x, f"Topic {x+1}")
        )
        topic_impact_df['topic_signal'] = topic_impact_df['avg_abnormal'].apply(
            lambda v: '🟢 Bullish' if v > 0 else ('🔴 Bearish' if v < 0 else '⚪ Neutral')
        )
        fig_topic = px.bar(
            topic_impact_df,
            x='topic_label', y='avg_abnormal', color='avg_abnormal',
            color_continuous_scale=DIVERGING_SCALE, color_continuous_midpoint=0,
            title=f"Avg 5D Abnormal Return by Dominant Topic ({sel_ticker})",
            labels={'avg_abnormal': 'Avg Abnormal Return (5D)', 'topic_label': 'Topic'},
            hover_data=['speech_count', 'topic_signal'], text='topic_signal'
        )
        fig_topic.update_traces(textposition='outside')
        fig_topic.add_hline(y=0, line_dash="dash", line_color=COLORS["line"])
        fig_topic.update_layout(xaxis_tickangle=-30)
        apply_chart_theme(fig_topic, height=440)
        topic_fig = json.loads(fig_topic.to_json())

        best_topic = topic_impact_df.iloc[0]
        alpha_driver = {
            "topicLabel": best_topic['topic_label'],
            "signal": best_topic['topic_signal'],
            "avgAbnormal": float(best_topic['avg_abnormal']),
        }

    return {
        "empty": False,
        "tickers": tickers,
        "selectedTicker": sel_ticker,
        "signal": {"label": ov_signal, "emoji": ov_emoji, "color": ov_color},
        "breakdown": {"bullish": n_bullish, "neutral": n_neutral, "bearish": n_bearish, "total": len(ticker_impact)},
        "confidence": {"pct": conf_disp, "avgAbnormalPct": overall_avg * 100},
        "priceFig": json.loads(fig.to_json()),
        "eventTable": records(disp_df),
        "sourceColors": SOURCE_COLORS,
        "avgBySourceFig": json.loads(fig_bar.to_json()),
        "topicCorrelationFig": topic_fig,
        "alphaDriver": alpha_driver,
    }
