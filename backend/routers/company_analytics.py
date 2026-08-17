"""
Stage 5 — Company Specific Returns vs. Rhetoric, ported from App.py's
`elif stage == "5. Company Analytics":` block.
"""
import os
import json
import numpy as np
import pandas as pd
from fastapi import APIRouter

from utils.db_utils import get_db_connection
from backend.cache import ttl_cache
from backend.colors import apply_chart_theme, DIVERGING_SCALE

router = APIRouter()

DB_PATH = './data/market_rhetoric.db'

COMPANY_TICKER_MAP = {
    "HDFC Bank": "HDFCBANK.NS",
    "Reliance Industries": "RELIANCE.NS",
    "Infosys": "INFY.NS",
    "TCS": "TCS.NS",
    "ICICI Bank": "ICICIBANK.NS",
}


@ttl_cache(ttl_seconds=1800)
def _load_company_topics(db_path, ticker):
    conn = get_db_connection(db_path)
    result = pd.read_sql_query(
        """
        SELECT s.date, td.topic_id, td.probability, i.return_t5
        FROM topic_distributions td
        JOIN speeches s ON td.speech_id = s.id
        JOIN speech_market_impact i ON i.speech_id = s.id
        WHERE td.model_name = 'Combined' AND i.ticker = ?
        """,
        conn, params=(ticker,)
    )
    conn.close()
    return result


@router.get("/company-analytics/companies")
def list_companies():
    return {"companies": list(COMPANY_TICKER_MAP.keys())}


@router.get("/company-analytics")
def get_company_analytics(company: str = "HDFC Bank"):
    import plotly.graph_objects as go

    if company not in COMPANY_TICKER_MAP:
        return {"empty": True}
    company_ticker = COMPANY_TICKER_MAP[company]

    topics_df = _load_company_topics(DB_PATH, company_ticker)
    if topics_df.empty:
        return {"empty": True, "company": company}

    topics_df = topics_df.copy()
    topics_df['date'] = pd.to_datetime(topics_df['date'])
    topics_df['weighted_strength'] = topics_df['probability'] * topics_df['return_t5']

    labels_path = "./data/processed/topic_labels_combined.json"
    topic_label_map = {}
    if os.path.exists(labels_path):
        with open(labels_path, 'r', encoding='utf-8') as f:
            topic_label_map = {int(k): v['label'] for k, v in json.load(f).items()}

    topics_df['month'] = topics_df['date'].dt.to_period('M').dt.to_timestamp()
    pivot_topics = (
        topics_df.groupby(['month', 'topic_id'])['weighted_strength']
        .mean().unstack().fillna(0)
    )
    topic_names = [topic_label_map.get(i, f"Topic {i}") for i in pivot_topics.columns]

    zmax = float(np.abs(pivot_topics.values).max()) or 1.0
    fig_heat = go.Figure(data=go.Heatmap(
        z=pivot_topics.values.T,
        x=[str(d) for d in pivot_topics.index],
        y=topic_names,
        colorscale=DIVERGING_SCALE,
        zmid=0, zmin=-zmax, zmax=zmax,
        colorbar=dict(title="Avg 5D Fwd<br>Return × Topic<br>Probability", tickfont=dict(family="IBM Plex Mono")),
        hovertemplate="%{y}<br>%{x}<br>Impact: %{z:.4f}<extra></extra>",
    ))
    apply_chart_theme(fig_heat, height=450)
    fig_heat.update_layout(
        title=f"Leadership Topic Impact Over Time vs {company} (Monthly, 5D Fwd Return-Weighted)",
        xaxis_title="Month", yaxis_title="Topic",
    )

    return {
        "empty": False,
        "company": company,
        "fig": json.loads(fig_heat.to_json()),
    }
