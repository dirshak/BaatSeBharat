"""
Stage 7 — Global Influence Map, ported from
src/geo_dashboard.py's `render_global_influence_map()` (4 tabs).
Reuses geo_dashboard's data functions and figure builders directly and
unmodified; only the st.* rendering calls are replaced with JSON responses.
"""
import json
from fastapi import APIRouter
from typing import Optional

from backend.cache import ttl_cache
from backend.jsonsafe import records

router = APIRouter()

try:
    from geo_dashboard import (
        compute_country_risk_scores,
        compute_shock_matrix,
        fetch_wb_data,
        _make_state_choropleth,
        _make_shock_heatmap,
        _make_company_map,
        WB_INDICATORS,
        KEY_COUNTRIES,
    )
    _GEO_OK = True
except Exception as _geo_err:
    _GEO_OK = False
    _geo_err_msg = str(_geo_err)

try:
    from prediction_engine import get_all_company_predictions
    from backend.routers.predictions import (
        _load_avg_sentiment_from_db, _load_topic_strength_from_npy,
        _cached_company_predictions, DB_PATH,
    )
    _PRED_AVAILABLE = True
except Exception:
    _PRED_AVAILABLE = False

# geo_dashboard's own @st.cache_data-decorated functions still work outside
# Streamlit (they just no-op the cache), so we add our own TTL cache around
# them here for real caching in the backend without touching that file.
_compute_country_risk_scores = ttl_cache(ttl_seconds=3600)(compute_country_risk_scores) if _GEO_OK else None
_compute_shock_matrix = ttl_cache(ttl_seconds=3600)(compute_shock_matrix) if _GEO_OK else None
_fetch_wb_data = ttl_cache(ttl_seconds=3600)(fetch_wb_data) if _GEO_OK else None


@router.get("/geo/country-risk")
def country_risk():
    if not _GEO_OK:
        return {"geoOk": False, "error": _geo_err_msg}
    risk_df = _compute_country_risk_scores()
    fig = _make_state_choropleth(risk_df)

    disp = risk_df.sort_values("score", ascending=False).copy()
    disp["Market State"] = disp["state"].map({"Bull": "🟢 Bull", "Neutral": "⚪ Neutral", "Bear": "🔴 Bear"})
    disp = disp.rename(columns={"gdp": "GDP Growth (%)", "inflation": "Inflation (%)", "score": "Score"})[
        ["Country", "Market State", "GDP Growth (%)", "Inflation (%)", "Score"]
    ]

    return {
        "geoOk": True,
        "fig": json.loads(fig.to_json()),
        "counts": {
            "bull": int((risk_df["state"] == "Bull").sum()),
            "neutral": int((risk_df["state"] == "Neutral").sum()),
            "bear": int((risk_df["state"] == "Bear").sum()),
        },
        "table": records(disp),
    }


@router.get("/geo/shocks")
def shocks(shock_type: str = "Inflation"):
    if not _GEO_OK:
        return {"geoOk": False, "error": _geo_err_msg}
    shock_df = _compute_shock_matrix()

    selected_fig = json.loads(_make_shock_heatmap(shock_df, shock_type).to_json())
    all_figs = {
        st_: json.loads(_make_shock_heatmap(shock_df, st_).to_json())
        for st_ in ["Inflation", "Policy", "Banking", "Geopolitical"]
    }

    return {
        "geoOk": True,
        "selectedFig": selected_fig,
        "allFigs": all_figs,
        "table": records(shock_df),
    }


@router.get("/geo/company-map")
def company_map():
    if not _GEO_OK:
        return {"geoOk": False, "error": _geo_err_msg}

    company_predictions = None
    if _PRED_AVAILABLE:
        try:
            sent = _load_avg_sentiment_from_db(DB_PATH)
            topic = _load_topic_strength_from_npy()
            company_predictions = _cached_company_predictions(round(sent, 3), round(topic, 3), "Neutral", 0.0)
        except Exception:
            company_predictions = None

    fig = _make_company_map(company_predictions)
    return {"geoOk": True, "fig": json.loads(fig.to_json())}


@router.get("/geo/indicators")
def list_indicators():
    if not _GEO_OK:
        return {"geoOk": False, "error": _geo_err_msg}
    return {"geoOk": True, "indicators": list(WB_INDICATORS.keys()), "keyCountries": KEY_COUNTRIES}


@router.get("/geo/indicator")
def indicator(name: str, countries: Optional[str] = None):
    import plotly.express as px

    if not _GEO_OK:
        return {"geoOk": False, "error": _geo_err_msg}
    if name not in WB_INDICATORS:
        return {"geoOk": True, "error": f"Unknown indicator '{name}'"}

    indicator_code = WB_INDICATORS[name]
    wb_df = _fetch_wb_data(indicator_code, name)
    simulated = bool(wb_df.attrs.get('simulated'))

    if wb_df.empty:
        return {"geoOk": True, "empty": True, "simulated": simulated}

    country_list = countries.split(",") if countries else [c for c in KEY_COUNTRIES if c in wb_df["Country"].unique()]
    all_countries = wb_df["Country"].unique().tolist()
    filtered = wb_df[wb_df["Country"].isin(country_list)] if country_list else wb_df

    fig_wb = px.choropleth(
        filtered, locations="Country", locationmode="country names", color=name,
        animation_frame=filtered["Year"].astype(str),
        color_continuous_scale=[[0.0, "#0F1727"], [0.5, "#5A4A2E"], [1.0, "#C97A2B"]],
        title=f"{name} (2018–2023)",
    )
    fig_wb.update_geos(projection_type="natural earth", showcoastlines=True, showcountries=True, bgcolor="#0B1220")
    fig_wb.update_layout(
        height=500,
        font=dict(family="IBM Plex Sans, sans-serif", color="#E8E4D9"),
        title_font=dict(family="Fraunces, serif", color="#E8E4D9", size=18),
        paper_bgcolor="#0B1220",
        geo=dict(bgcolor="#0B1220", landcolor="#131B2C"),
    )

    trend = None
    if len(filtered["Country"].unique()) <= 15:
        trend_df = filtered.pivot(index="Year", columns="Country", values=name)
        trend = json.loads(trend_df.reset_index().to_json(orient="records"))

    return {
        "geoOk": True, "empty": False, "simulated": simulated,
        "allCountries": all_countries, "selectedCountries": country_list,
        "fig": json.loads(fig_wb.to_json()),
        "trend": trend,
        "rawData": records(filtered),
    }
