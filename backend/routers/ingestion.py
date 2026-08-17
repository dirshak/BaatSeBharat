"""
Stage 1 — Data Ingestion & Storage, ported from App.py's
`elif stage == "1. Data Ingestion":` block.
"""
import json
import pandas as pd
from fastapi import APIRouter
from typing import Optional

from utils.db_utils import get_db_connection
from backend.cache import ttl_cache
from backend.colors import apply_chart_theme, CATEGORY_SEQUENCE
from backend.jsonsafe import records

router = APIRouter()

DB_PATH = './data/market_rhetoric.db'


@ttl_cache(ttl_seconds=1800)
def _load_speech_index(db_path):
    conn = get_db_connection(db_path)
    df = pd.read_sql_query(
        "SELECT id, date, source, speaker, title FROM speeches ORDER BY date DESC", conn
    )
    conn.close()
    return df


@ttl_cache(ttl_seconds=1800)
def _load_speech_text(db_path, speech_id):
    conn = get_db_connection(db_path)
    row = pd.read_sql_query(
        "SELECT full_text FROM speeches WHERE id = ?", conn, params=(speech_id,)
    )
    conn.close()
    return row['full_text'].iloc[0] if not row.empty else None


@router.get("/ingestion/speeches")
def list_speeches(source: Optional[str] = None):
    df = _load_speech_index(DB_PATH)
    sources = sorted(df['source'].dropna().unique().tolist()) if not df.empty else []
    if source and source != 'All' and not df.empty:
        df = df[df['source'] == source]
    return {"sources": sources, "speeches": records(df)}


@router.get("/ingestion/speeches/{speech_id}")
def get_speech_text(speech_id: int):
    text = _load_speech_text(DB_PATH, speech_id)
    return {"fullText": text}


@ttl_cache(ttl_seconds=1800)
def _load_market_close(db_path):
    conn = get_db_connection(db_path)
    df = pd.read_sql_query("SELECT date, ticker, close FROM market_data", conn)
    conn.close()
    return df


@router.get("/ingestion/market")
def get_market_chart():
    import plotly.express as px

    df_m = _load_market_close(DB_PATH)
    if df_m.empty:
        return {"fig": None}
    df_m = df_m.copy()
    df_m['date'] = pd.to_datetime(df_m['date'])
    fig = px.line(
        df_m, x='date', y='close', color='ticker',
        title="Index Performance",
        color_discrete_sequence=CATEGORY_SEQUENCE,
    )
    apply_chart_theme(fig, height=420)
    return {"fig": json.loads(fig.to_json())}
