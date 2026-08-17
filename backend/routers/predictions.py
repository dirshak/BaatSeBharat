"""
Stage 6 — AI Market Predictions, ported from App.py's
`elif stage == "6. AI Predictions":` block (Company + Sector tabs),
plus a new Sector Predictions map endpoint.
"""
import os
import json
import numpy as np
import pandas as pd
from fastapi import APIRouter
from typing import Optional

from utils.db_utils import get_db_connection
from backend.cache import ttl_cache
from backend.colors import apply_chart_theme, COLORS, CATEGORY_SEQUENCE, DIVERGING_SCALE
from backend.jsonsafe import records

router = APIRouter()

DB_PATH = './data/market_rhetoric.db'

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
    from geo_dashboard import COMPANY_LOCATIONS
except Exception:
    COMPANY_LOCATIONS = {}


# ===========================================================================
# CACHED DATA LOADERS — ported from App.py lines 507-625, 1542-1551
# ===========================================================================

@ttl_cache(ttl_seconds=1800)
def _load_avg_sentiment_from_db(db_path: str) -> float:
    try:
        conn = get_db_connection(db_path)
        df = pd.read_sql_query("SELECT AVG(abnormal_return) as avg_ret FROM speech_market_impact", conn)
        conn.close()
        val = float(df['avg_ret'].iloc[0] or 0.0)
        return float(np.clip(val * 5, -1, 1))
    except Exception:
        return 0.0


@ttl_cache(ttl_seconds=1800)
def _load_topic_strength_from_npy() -> float:
    try:
        npy_path = './data/processed/topic_distributions_combined.npy'
        if os.path.exists(npy_path):
            dists = np.load(npy_path)
            return float(dists.max(axis=1).mean())
    except Exception:
        pass
    return 0.5


@ttl_cache(ttl_seconds=1800)
def _load_regime_from_csv(ticker: str) -> str:
    csv_path = f'./data/processed/regime_labels_{ticker}.csv'
    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for candidate in ('regime', 'regime_label'):
                if candidate in df.columns:
                    regime_col = candidate
                    break
            else:
                regime_col = df.columns[-1]
            latest = str(df[regime_col].iloc[-1])
            if 'stable' in latest.lower() or 'bull' in latest.lower():
                return 'Bull'
            elif 'volatile' in latest.lower() or 'bear' in latest.lower():
                return 'Bear'
            return 'Neutral'
    except Exception:
        pass
    return 'Neutral'


@ttl_cache(ttl_seconds=1800)
def _load_sector_avg_returns_from_db(db_path: str) -> pd.DataFrame:
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


@ttl_cache(ttl_seconds=1800)
def _load_regime_df_for_sectors() -> pd.DataFrame:
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


@ttl_cache(ttl_seconds=1800)
def _cached_company_predictions(sentiment: float, topic_str: float, regime: str, hist_ret: float) -> list:
    if not _PRED_OK:
        return []
    return get_all_company_predictions(
        sentiment_score=sentiment, topic_strength=topic_str,
        regime_label=regime, historical_return=hist_ret, use_llm=False,
    )


@ttl_cache(ttl_seconds=1800)
def _cached_sector_predictions(sentiment: float, topic_str: float, sector_returns_json: str, regime_json: str) -> list:
    if not _PRED_OK:
        return []
    import io
    sector_returns = pd.read_json(io.StringIO(sector_returns_json)) if sector_returns_json else None
    regime_df = pd.read_json(io.StringIO(regime_json)) if regime_json else None
    return get_all_sector_predictions(
        sentiment_score=sentiment, topic_strength=topic_str,
        sector_returns=sector_returns, regime_df=regime_df,
    )


@ttl_cache(ttl_seconds=1800)
def _companies_missing_market_data(db_path):
    conn = get_db_connection(db_path)
    tickers_with_data = set(pd.read_sql_query("SELECT DISTINCT ticker FROM market_data", conn)['ticker'])
    conn.close()
    return [c for c, t in COMPANY_UNIVERSE.items() if t not in tickers_with_data]


def _signal_defaults():
    """Live defaults for the signal-override sliders, as in App.py."""
    live_sentiment = _load_avg_sentiment_from_db(DB_PATH)
    live_topic_str = _load_topic_strength_from_npy()
    live_regime_raw = _load_regime_from_csv('^NSEI')
    sector_returns = _load_sector_avg_returns_from_db(DB_PATH)
    regime_df = _load_regime_df_for_sectors()
    live_hist_ret = float(sector_returns['return_5d'].mean()) if not sector_returns.empty else 0.0
    return live_sentiment, live_topic_str, live_regime_raw, sector_returns, regime_df, live_hist_ret


@router.get("/predictions/defaults")
def get_defaults():
    if not _PRED_OK:
        return {"predOk": False, "error": _pred_err_msg}
    live_sentiment, live_topic_str, live_regime_raw, _, _, live_hist_ret = _signal_defaults()
    return {
        "predOk": True,
        "sentiment": round(live_sentiment, 3),
        "topicStrength": round(live_topic_str, 3),
        "regime": live_regime_raw,
        "historicalReturn": live_hist_ret,
        "llmModeAvailable": _llm_mode_available(),
        "companies": list(COMPANY_UNIVERSE.keys()),
    }


@router.get("/predictions/company")
def company_prediction(
    company: str,
    sentiment: float = 0.0,
    topic: float = 0.5,
    regime: str = "Neutral",
    use_llm: bool = False,
):
    if not _PRED_OK:
        return {"predOk": False, "error": _pred_err_msg}

    _, _, _, sector_returns, _, live_hist_ret = _signal_defaults()

    co_ticker = COMPANY_UNIVERSE.get(company, '')
    co_regime = _load_regime_from_csv(co_ticker) if co_ticker else regime
    co_hist_ret = live_hist_ret
    if not sector_returns.empty and co_ticker:
        row = sector_returns[sector_returns['sector'] == co_ticker]
        if not row.empty:
            co_hist_ret = float(row['return_5d'].iloc[0])

    pred = get_company_prediction(
        company, sentiment_score=sentiment, topic_strength=topic,
        regime_label=co_regime, historical_return=co_hist_ret, use_llm=use_llm,
    )
    return {"predOk": True, "prediction": pred}


@router.get("/predictions/company/all")
def all_company_predictions(
    sentiment: float = 0.0,
    topic: float = 0.5,
    regime: str = "Neutral",
):
    import plotly.graph_objects as go
    import plotly.express as px

    if not _PRED_OK:
        return {"predOk": False, "error": _pred_err_msg}

    _, _, _, _, _, live_hist_ret = _signal_defaults()
    missing = _companies_missing_market_data(DB_PATH)

    all_preds = _cached_company_predictions(round(sentiment, 3), round(topic, 3), regime, round(live_hist_ret, 6))
    if not all_preds:
        return {"predOk": True, "predictions": [], "missingMarketData": missing, "barFig": None, "scatterFig": None}

    pr = []
    for p in all_preds:
        pr.append({
            "Company": p["company"], "Signal": p["signal"], "Confidence": p["confidence"], "Score": p["score"],
            "1D %": p["predictions"].get(1, {}).get("return_pct", 0),
            "5D %": p["predictions"].get(5, {}).get("return_pct", 0),
            "10D %": p["predictions"].get(10, {}).get("return_pct", 0),
        })
    pr_df = pd.DataFrame(pr).sort_values("Score", ascending=False)
    sig_clr = {"Bullish": COLORS["green"], "Neutral": COLORS["ink_dim"], "Bearish": COLORS["rust"]}

    fig_bar = go.Figure(go.Bar(
        x=pr_df["Company"], y=pr_df["5D %"],
        marker_color=[sig_clr.get(s, COLORS["ink_dim"]) for s in pr_df["Signal"]],
        text=[f"{v:+.2f}%" for v in pr_df["5D %"]], textposition="outside",
        hovertemplate="<b>%{x}</b><br>5D: %{y:+.2f}%<extra></extra>",
    ))
    fig_bar.add_hline(y=0, line_dash="dot", line_color=COLORS["line"])
    apply_chart_theme(fig_bar, height=370)
    fig_bar.update_layout(title="All Companies — 5-Day Return Forecast", xaxis_tickangle=-35, yaxis_title="Forecast Return (%)")

    fig_sc = px.scatter(
        pr_df, x="Score", y="Confidence", color="Signal", size="Confidence",
        hover_name="Company", color_discrete_map=sig_clr, title="Signal Score vs Prediction Confidence",
    )
    apply_chart_theme(fig_sc, height=330)

    return {
        "predOk": True,
        "predictions": records(pr_df),
        "missingMarketData": missing,
        "barFig": json.loads(fig_bar.to_json()),
        "scatterFig": json.loads(fig_sc.to_json()),
    }


@router.get("/predictions/sector")
def sector_predictions(sentiment: float = 0.0, topic: float = 0.5, regime: str = "Neutral"):
    import plotly.graph_objects as go

    if not _PRED_OK:
        return {"predOk": False, "error": _pred_err_msg}

    _, _, _, sector_returns, regime_df, _ = _signal_defaults()
    sr_json = sector_returns.to_json() if not sector_returns.empty else ""
    rdf_json = regime_df.to_json() if not regime_df.empty else ""
    sec_preds = _cached_sector_predictions(round(sentiment, 3), round(topic, 3), sr_json, rdf_json)

    if not sec_preds:
        return {"predOk": True, "sectors": [], "fig": None}

    sec_rows = []
    for sp in sec_preds:
        sec_rows.append({
            "Sector": sp["sector"], "Signal": sp["signal"], "Emoji": sp["emoji"],
            "Conf": sp["confidence"], "Score": sp["score"],
            "1D %": sp["predictions"].get(1, {}).get("return_pct", 0),
            "5D %": sp["predictions"].get(5, {}).get("return_pct", 0),
            "10D %": sp["predictions"].get(10, {}).get("return_pct", 0),
        })
    sd = pd.DataFrame(sec_rows).sort_values("Score", ascending=False)

    fig_sec = go.Figure()
    for (col_name, label), bar_color in zip(
        [("1D %", "1-Day"), ("5D %", "1-Week"), ("10D %", "10-Day")], CATEGORY_SEQUENCE
    ):
        fig_sec.add_trace(go.Bar(
            name=label, x=sd["Sector"], y=sd[col_name],
            text=[f"{v:+.2f}%" for v in sd[col_name]], textposition="outside",
            marker_color=bar_color,
        ))
    fig_sec.add_hline(y=0, line_dash="dot", line_color=COLORS["line"])
    apply_chart_theme(fig_sec, height=370)
    fig_sec.update_layout(
        title="Sector Return Forecasts — 1D / 1W / 10D", barmode="group", yaxis_title="Forecast Return (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return {
        "predOk": True,
        "sectors": records(sd),
        "fig": json.loads(fig_sec.to_json()),
        "regime": regime,
    }


# ===========================================================================
# NEW — Sector Predictions map. Reuses SECTOR_COMPANIES (prediction_engine)
# and COMPANY_LOCATIONS (geo_dashboard, real HQ coordinates) — no invented
# geography, only the sector's own already-computed prediction score/return.
# ===========================================================================

@router.get("/predictions/sector-map")
def sector_prediction_map(sentiment: float = 0.0, topic: float = 0.5, regime: str = "Neutral"):
    import plotly.graph_objects as go

    if not _PRED_OK:
        return {"predOk": False, "error": _pred_err_msg}

    _, _, _, sector_returns, regime_df, _ = _signal_defaults()
    sr_json = sector_returns.to_json() if not sector_returns.empty else ""
    rdf_json = regime_df.to_json() if not regime_df.empty else ""
    sec_preds = _cached_sector_predictions(round(sentiment, 3), round(topic, 3), sr_json, rdf_json)

    sec_by_name = {sp["sector"]: sp for sp in sec_preds}

    markers = []
    skipped_sectors = []
    for sector, companies in SECTOR_COMPANIES.items():
        sp = sec_by_name.get(sector)
        if sp is None:
            continue
        # "Broad Market" spans every company (would overlap every other
        # sector's dots) and any sector with zero mapped constituents
        # (e.g. Pharma today) has nothing real to plot -- skip rather than
        # invent a placement, same caveat pattern used elsewhere in Stage 6.
        if sector == "Broad Market" or not companies:
            skipped_sectors.append(sector)
            continue

        ret5 = sp["predictions"].get(5, {}).get("return_pct", 0.0)
        score = sp.get("score", ret5)
        confidence = sp.get("confidence", 50.0)

        for company in companies:
            loc = COMPANY_LOCATIONS.get(company)
            if not loc:
                continue
            markers.append({
                "sector": sector,
                "company": company,
                "city": loc["city"],
                "lat": loc["lat"],
                "lon": loc["lon"],
                "score": score,
                "return5d": ret5,
                "signal": sp["signal"],
                "confidence": confidence,
            })

    if not markers:
        return {"predOk": True, "fig": None, "skippedSectors": skipped_sectors}

    mdf = pd.DataFrame(markers)
    max_abs = float(np.abs(mdf["score"]).max()) or 1.0

    fig = go.Figure(go.Scattergeo(
        lat=mdf["lat"], lon=mdf["lon"],
        mode="markers",
        marker=dict(
            size=mdf["confidence"] / 8 + 6,
            color=mdf["score"],
            colorscale=DIVERGING_SCALE,
            cmin=-max_abs, cmax=max_abs,
            colorbar=dict(
                title=dict(text="Predicted<br>Effect", font=dict(color=COLORS["ink_dim"])),
                tickfont=dict(color=COLORS["ink_dim"]),
            ),
            line=dict(width=1, color="white"),
            opacity=0.9,
        ),
        customdata=mdf[["sector", "company", "city", "return5d", "signal"]],
        hovertemplate=(
            "<b>%{customdata[1]}</b> (%{customdata[0]})<br>"
            "%{customdata[2]}<br>"
            "5D Forecast: %{customdata[3]:+.2f}%<br>"
            "Signal: %{customdata[4]}<extra></extra>"
        ),
    ))
    fig.update_geos(
        projection_type="natural earth",
        showcoastlines=True, showcountries=True,
        countrycolor=COLORS["line"], coastlinecolor=COLORS["line"],
        bgcolor=COLORS["bg"], landcolor=COLORS["surface"], lakecolor=COLORS["bg"],
        center=dict(lat=20, lon=78), projection_scale=3.5,
    )
    fig.update_layout(
        title="Sector Prediction Map — Bullish (green) → Bearish (red)",
        height=460,
        paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"],
        font=dict(family="IBM Plex Sans, sans-serif", color=COLORS["ink"]),
        title_font=dict(family="Fraunces, serif", color=COLORS["ink"], size=18),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    return {"predOk": True, "fig": json.loads(fig.to_json()), "skippedSectors": skipped_sectors}
