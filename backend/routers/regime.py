"""
Stage 4 — Market Regime Intelligence (HMM), ported from App.py's
`elif stage == "4. Regime Intelligence":` block.
"""
import json
import numpy as np
import pandas as pd
from fastapi import APIRouter
from typing import Optional

from utils.db_utils import get_db_connection
from backend.cache import ttl_cache
from backend.colors import apply_chart_theme, COLORS

router = APIRouter()

DB_PATH = './data/market_rhetoric.db'

REGIME_SHAPE_COLORS = {
    'Stable': 'rgba(47, 111, 78, 0.28)',
    'Transitional': 'rgba(201, 122, 43, 0.24)',
    'Volatile': 'rgba(166, 80, 58, 0.30)',
}


@ttl_cache(ttl_seconds=1800)
def _load_regime_and_market(db_path):
    conn = get_db_connection(db_path)
    regimes_df = pd.read_sql_query(
        "SELECT date, sector, regime, confidence FROM regime_classifications ORDER BY date", conn
    )
    market_df = pd.read_sql_query("SELECT date, ticker, close FROM market_data ORDER BY date", conn)
    conn.close()
    regimes_df['date'] = pd.to_datetime(regimes_df['date'])
    market_df['date'] = pd.to_datetime(market_df['date'])
    return regimes_df, market_df


@router.get("/regime/tickers")
def list_tickers():
    _, market = _load_regime_and_market(DB_PATH)
    return {"tickers": market['ticker'].unique().tolist() if not market.empty else []}


@router.get("/regime")
def get_regime(ticker: Optional[str] = None):
    import plotly.graph_objects as go

    regimes, market = _load_regime_and_market(DB_PATH)
    if regimes.empty or market.empty:
        return {"empty": True}

    tickers = market['ticker'].unique().tolist()
    sel_ticker = ticker if ticker in tickers else (tickers[0] if tickers else None)
    if sel_ticker is None:
        return {"empty": True}

    t_market = market[market['ticker'] == sel_ticker]
    t_regimes = regimes[regimes['sector'] == sel_ticker].sort_values('date')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_market['date'], y=t_market['close'], name="Price",
                              line=dict(color=COLORS["ink"], width=1.5)))

    if not t_regimes.empty:
        regime_vals = t_regimes['regime'].values
        dates_vals = t_regimes['date'].values
        segment_id = np.cumsum(np.concatenate(([0], regime_vals[1:] != regime_vals[:-1])))
        seg_df = pd.DataFrame({'segment': segment_id, 'regime': regime_vals, 'date': dates_vals})
        bounds = seg_df.groupby('segment').agg(regime=('regime', 'first'), x0=('date', 'first'), x1=('date', 'last'))

        shapes = [
            dict(
                type='rect', xref='x', yref='paper',
                x0=row.x0, x1=row.x1, y0=0, y1=1,
                fillcolor=REGIME_SHAPE_COLORS.get(row.regime, 'gray'), opacity=0.5,
                line_width=0, layer='below',
            )
            for row in bounds.itertuples()
        ]
        fig.update_layout(shapes=shapes)

    apply_chart_theme(fig, height=550)
    fig.update_layout(title=f"{sel_ticker} Regime Timeline (Green=Stable, Saffron=Transitional, Rust=Volatile)")

    return {
        "empty": False,
        "tickers": tickers,
        "selectedTicker": sel_ticker,
        "fig": json.loads(fig.to_json()),
        "hasRegimeData": not t_regimes.empty,
        "coveredTickers": sorted(regimes['sector'].unique().tolist()),
    }
