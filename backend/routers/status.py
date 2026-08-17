"""
GET /api/status — top bar (logo/brand handled by frontend static asset) +
status strip (pipeline freshness, LLM mode) + Executive Summary source
breakdown, ported from App.py's top-bar/status-strip section and the
"Executive Summary" stage.
"""
import os
import json
import base64
from datetime import datetime

import pandas as pd
from fastapi import APIRouter

from utils.db_utils import get_db_connection
from backend.cache import ttl_cache
from backend.colors import COLORS, SOURCE_COLORS
from backend.jsonsafe import records

router = APIRouter()

DB_PATH = './data/market_rhetoric.db'
MODEL_PATH = "./data/processed/topic_distributions_combined.npy"

try:
    from prediction_engine import _llm_mode_available
    _PRED_OK = True
except Exception as _pred_err:
    _PRED_OK = False
    _pred_err_msg = str(_pred_err)

try:
    from geo_dashboard import render_global_influence_map  # noqa: F401 (import check only)
    _GEO_OK = True
except Exception as _geo_err:
    _GEO_OK = False
    _geo_err_msg = str(_geo_err)

try:
    from prediction_history import compute_prediction_vs_actual  # noqa: F401
    _PREDHIST_OK = True
except Exception as _predhist_err:
    _PREDHIST_OK = False
    _predhist_err_msg = str(_predhist_err)


@ttl_cache(ttl_seconds=3600)
def _logo_b64(path="logo.png"):
    if os.path.exists(path):
        return base64.b64encode(open(path, "rb").read()).decode()
    return ""


@router.get("/status")
def get_status():
    models_exist = os.path.exists(MODEL_PATH)
    last_update = None
    if models_exist:
        mtime = os.path.getmtime(MODEL_PATH)
        last_update = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')

    llm_active = bool(_PRED_OK and _llm_mode_available())

    required_files = [
        DB_PATH,
        './data/processed/topic_distributions_combined.npy',
        './data/processed/topic_labels_combined.json',
    ]
    missing = [f for f in required_files if not os.path.exists(f)]

    return {
        "modelsExist": models_exist,
        "lastUpdate": last_update,
        "predOk": _PRED_OK,
        "predErr": None if _PRED_OK else _pred_err_msg,
        "geoOk": _GEO_OK,
        "geoErr": None if _GEO_OK else _geo_err_msg,
        "predHistOk": _PREDHIST_OK,
        "predHistErr": None if _PREDHIST_OK else _predhist_err_msg,
        "llmModeActive": llm_active,
        "missingFiles": missing,
        "logoBase64": _logo_b64(),
        "colors": COLORS,
        "sourceColors": SOURCE_COLORS,
    }


@ttl_cache(ttl_seconds=1800)
def _load_db_stats():
    try:
        conn = get_db_connection(DB_PATH)
        speech_count = pd.read_sql_query("SELECT COUNT(*) as count FROM speeches", conn)['count'][0]
        market_count = pd.read_sql_query("SELECT COUNT(*) as count FROM market_data", conn)['count'][0]
        conn.close()
        return int(speech_count), int(market_count)
    except Exception:
        return 0, 0


@ttl_cache(ttl_seconds=1800)
def _load_source_breakdown():
    try:
        conn = get_db_connection(DB_PATH)
        df = pd.read_sql_query("SELECT source, COUNT(*) as count FROM speeches GROUP BY source", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


@ttl_cache(ttl_seconds=3600)
def _active_topic_count():
    path = "./data/processed/topic_labels_combined.json"
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return len(json.load(f))
    return 0


@ttl_cache(ttl_seconds=3600)
def _directional_backtest():
    try:
        from models.causal_validation import CausalValidator
    except ImportError:
        from src.models.causal_validation import CausalValidator
    try:
        return CausalValidator(db_path=DB_PATH).backtest_directional_hit_rate()
    except Exception:
        return {}


@router.get("/overview")
def get_overview():
    import plotly.express as px
    from backend.colors import apply_chart_theme

    s_count, m_count = _load_db_stats()
    bt = _directional_backtest()

    hit_rate = None
    delta_pp = None
    cutoff_date = None
    n_events = None
    if bt:
        hit_rate = bt['hit_rate'] * 100
        delta_pp = hit_rate - 50
        cutoff_date = bt.get('cutoff_date')
        n_events = bt.get('n_events')

    breakdown = _load_source_breakdown()
    source_pie = None
    source_rows = []
    if not breakdown.empty:
        for _, r in breakdown.iterrows():
            source_rows.append({"source": r['source'], "count": int(r['count'])})
        fig_pie = px.pie(
            breakdown, values='count', names='source',
            title="Speech Distribution by Source",
            color='source',
            color_discrete_map=SOURCE_COLORS,
            hole=0.55,
        )
        fig_pie.update_traces(textfont=dict(family="IBM Plex Mono", size=12),
                               marker=dict(line=dict(color=COLORS["surface"], width=2)))
        apply_chart_theme(fig_pie, height=340)
        source_pie = json.loads(fig_pie.to_json())

    recent = []
    if s_count > 0:
        conn = get_db_connection(DB_PATH)
        recent_df = pd.read_sql_query(
            "SELECT date, source, speaker, title FROM speeches ORDER BY date DESC LIMIT 10", conn
        )
        conn.close()
        recent = records(recent_df)

    return {
        "speechCount": s_count,
        "marketCount": m_count,
        "activeTopics": _active_topic_count(),
        "hitRatePct": hit_rate,
        "hitRateDeltaPp": delta_pp,
        "backtestCutoffDate": cutoff_date,
        "backtestNEvents": n_events,
        "sourceBreakdown": source_rows,
        "sourcePieFig": source_pie,
        "recentSpeeches": recent,
    }
