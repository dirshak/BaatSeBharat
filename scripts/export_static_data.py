"""
scripts/export_static_data.py
==============================
One-shot static export for BaatSeBharat's static-first architecture.

Run AFTER scripts/run_prototype.py (which does all the heavy modeling:
FinBERT, BERTopic/NMF, HMM regimes, Granger causality, optional Groq
classification). This script does NOT do any of that modeling itself --
it only imports the same functions backend/routers/ already uses to build
each stage's Plotly figures/tables, calls them once, and writes the result
to versioned JSON files under frontend/public/data/ so the React frontend
can render every stage without a live backend in production.

Design notes (see the Step-1 audit for the full reasoning):
- Filters that don't change which server-side computation runs (Stage 3's
  "source" checkboxes, Stage 7 Macro Indicators' "countries" multiselect,
  Stage 8's source/company dropdowns) are exported UNFILTERED once; the
  frontend does that filtering client-side over the already-fetched JSON.
- Stage 6 exports each company/sector's *baseline inputs* (already
  resolved from DB/CSV/npy + any cached Groq classification) separately
  from a composite score, so the frontend can recompute the composite
  itself when the what-if sliders move -- see frontend/src/predictionMath.js
  (added in Step 4) for the ported formula.
- The World Bank "Macro Indicators" map is exported as a raw
  Country/Year/Value table (not a pre-baked Plotly figure), because its
  country-filter needs to reshape the trace arrays -- the frontend builds
  the choropleth trace client-side from this table (Step 4).
"""
from __future__ import annotations

import os
import re
import sys
import json
import time
import traceback

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'src'))
sys.path.insert(0, os.path.join(_ROOT, 'TradingAgents'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

OUT_DIR = os.path.join(_ROOT, 'frontend', 'public', 'data')

_manifest: dict = {
    "generatedAt": None,
    "stage1_ingestion": {},
    "stage2_nlp": {},
    "stage3_market_impact": {},
    "stage4_regime": {},
    "stage5_company_analytics": {},
    "stage6_predictions": {},
    "stage7_geo": {},
    "stage8_preview": {},
}
_sizes: list[tuple[str, int]] = []
_errors: list[tuple[str, str]] = []


def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r'[^a-z0-9]+', '_', s)
    return s.strip('_') or "_"


def write_json(rel_path: str, data) -> int:
    full = os.path.join(OUT_DIR, rel_path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, 'w', encoding='utf-8') as f:
        json.dump(data, f, separators=(',', ':'), ensure_ascii=False, default=str)
    size = os.path.getsize(full)
    _sizes.append((rel_path, size))
    return size


def step(label):
    def _decorator(fn):
        def _wrapped(*a, **kw):
            t0 = time.time()
            print(f"  -> {label} ...", flush=True)
            try:
                fn(*a, **kw)
                print(f"     done in {time.time()-t0:.1f}s")
            except Exception as exc:
                print(f"     FAILED: {exc}")
                _errors.append((label, "".join(traceback.format_exception(exc))))
        return _wrapped
    return _decorator


# ===========================================================================
# Status / top bar
# ===========================================================================

@step("status")
def export_status():
    from backend.routers import status
    data = status.get_status()
    # logo.png is already a static asset under frontend/public/ -- no need
    # to duplicate it as base64 inside every fetch of status.json.
    data.pop("logoBase64", None)
    write_json("status.json", data)


@step("stage0 overview")
def export_overview():
    from backend.routers import status
    write_json("stage0_overview.json", status.get_overview())


# ===========================================================================
# Stage 1 -- Data Ingestion
# ===========================================================================

@step("stage1 ingestion")
def export_ingestion():
    from backend.routers import ingestion

    index = ingestion.list_speeches(source=None)
    write_json("stage1_ingestion/index.json", index)

    speech_ids = [s["id"] for s in index["speeches"]]
    for sid in speech_ids:
        write_json(f"stage1_ingestion/speech/{sid}.json", ingestion.get_speech_text(sid))

    write_json("stage1_ingestion/market.json", ingestion.get_market_chart())

    _manifest["stage1_ingestion"] = {
        "sources": index["sources"],
        "speechIds": speech_ids,
    }


# ===========================================================================
# Stage 2 -- NLP & Topic Modeling
# ===========================================================================

@step("stage2 nlp")
def export_nlp():
    from backend.routers import nlp

    slugs = {}
    for model_name in nlp.MODEL_OPTIONS:
        data = nlp.get_topics(model=model_name)
        slug = slugify(model_name)
        write_json(f"stage2_nlp/{slug}.json", data)
        slugs[model_name] = slug

    _manifest["stage2_nlp"] = {"models": list(nlp.MODEL_OPTIONS.keys()), "slugs": slugs}


# ===========================================================================
# Stage 3 -- Market Impact
# ===========================================================================

@step("stage3 market impact")
def export_market_impact():
    from backend.routers import market_impact

    tickers = market_impact.list_tickers()["tickers"]
    slugs = {}
    for t in tickers:
        # sources=None -> full unfiltered event table; the frontend's
        # "Filter by Source" checkboxes filter this client-side (see
        # export_static_data.py module docstring).
        data = market_impact.get_market_impact(ticker=t, sources=None)
        slug = slugify(t)
        write_json(f"stage3_market_impact/{slug}.json", data)
        slugs[t] = slug

    _manifest["stage3_market_impact"] = {"tickers": tickers, "slugs": slugs}


# ===========================================================================
# Stage 4 -- Regime Intelligence
# ===========================================================================

@step("stage4 regime")
def export_regime():
    from backend.routers import regime

    tickers = regime.list_tickers()["tickers"]
    slugs = {}
    for t in tickers:
        data = regime.get_regime(ticker=t)
        slug = slugify(t)
        write_json(f"stage4_regime/{slug}.json", data)
        slugs[t] = slug

    _manifest["stage4_regime"] = {"tickers": tickers, "slugs": slugs}


# ===========================================================================
# Stage 5 -- Company Analytics
# ===========================================================================

@step("stage5 company analytics")
def export_company_analytics():
    from backend.routers import company_analytics

    companies = company_analytics.list_companies()["companies"]
    slugs = {}
    for c in companies:
        data = company_analytics.get_company_analytics(company=c)
        slug = slugify(c)
        write_json(f"stage5_company_analytics/{slug}.json", data)
        slugs[c] = slug

    _manifest["stage5_company_analytics"] = {"companies": companies, "slugs": slugs}


# ===========================================================================
# Stage 6 -- AI Predictions
#
# Exports BASELINE INPUTS per company/sector (already resolved from DB/CSV/
# npy + cached Groq classification -- see src/prediction_engine.py's
# _resolve_company_*/_resolve_sector_* helpers, called here unchanged) plus
# the static per-company/sector profile constants and the regime multiplier
# table. The frontend ports _composite_score()/_forecast_return() to JS and
# recomputes them from these baselines whenever a slider moves -- no
# network round-trip. See the Step-1 audit for the exact formula.
#
# NOTE: this deliberately does NOT export a precomputed prediction at
# default slider values -- the composite math is cheap enough (a handful
# of multiplications) that the frontend just computes it on load from the
# baselines below, so there is only ever one source of truth for the
# formula (the ported JS), not two.
# ===========================================================================

@step("stage6 predictions (baselines)")
def export_predictions():
    import numpy as np
    import prediction_engine as pe
    from backend.routers.predictions import _signal_defaults, _load_regime_from_csv, DB_PATH

    live_sentiment, live_topic_str, live_regime_raw, sector_returns, regime_df, live_hist_ret = _signal_defaults()

    # -- per-company baselines --------------------------------------------
    #
    # IMPORTANT: regime and historical-return are NOT slider-driven anywhere
    # in the app -- traced exactly against backend/routers/predictions.py:
    #
    #  - "Company detail" view (company_prediction()) always overrides
    #    regime_label with `_load_regime_from_csv(ticker)` BEFORE calling
    #    get_company_prediction(), so the Regime slider has zero effect
    #    here; historical_return is that company's own row in the
    #    ticker-keyed sector_returns table (or the global live average if
    #    no row matches) -- also not slider-driven.
    #  - "All Companies" bulk table (get_all_company_predictions()) passes
    #    the RAW Regime slider value uniformly to every company (so the
    #    slider DOES matter here) and the GLOBAL live_hist_ret constant
    #    (same value for every company, never slider-driven) for
    #    historical_return.
    #
    # Both paths funnel through get_company_prediction()'s OWN internal
    # blend (`actual_regime = regimeBase if regime_label in (Neutral,
    # Stable) else regime_label`, same for historical_return) -- so rather
    # than re-deriving that blend twice in JS, we precompute the FINAL
    # blended actual_regime/actual_historical_return for each of the two
    # call paths here, once, and export them directly. The frontend only
    # ever blends sentiment/topic client-side (the two genuinely
    # slider-driven inputs) plus, for the bulk table only, regime.
    companies_out = []
    for company, ticker in pe.COMPANY_UNIVERSE.items():
        beta, rhet_sens, base_drift = pe.COMPANY_PROFILES.get(company, (1.0, 0.75, 0.0))

        sent_base = pe._resolve_company_sentiment(ticker, 0.0)
        topic_base = pe._resolve_company_topic_strength(ticker, 0.5)
        regime_base = pe._resolve_company_regime(ticker, "Neutral")
        hist_ret_base = pe._resolve_company_historical_return(ticker)
        llm_strength, llm_sentiment = pe._resolve_company_llm_signal(ticker)
        if llm_strength is not None:
            w = pe._groq_blend_weight()
            topic_base = (1 - w) * topic_base + w * llm_strength
            sent_base = (1 - w) * sent_base + w * llm_sentiment

        m30, m10, m5 = pe._fetch_price_momentum(ticker) if ticker else (0.0, 0.0, 0.0)
        current_price = pe._fetch_current_price(ticker) if ticker else None

        # -- "detail view" regime/historical-return, exactly matching
        #    company_prediction()'s pre-override + get_company_prediction()'s
        #    internal blend, both from prediction_engine.py/predictions.py --
        detail_regime_raw = _load_regime_from_csv(ticker) if ticker else "Neutral"
        actual_regime_detail = (
            regime_base if detail_regime_raw in ("Neutral", "Stable") else detail_regime_raw
        )

        co_hist_ret_detail = live_hist_ret
        if not sector_returns.empty and ticker:
            row = sector_returns[sector_returns["sector"] == ticker]
            if not row.empty:
                co_hist_ret_detail = float(row["return_5d"].iloc[0])
        actual_hist_ret_detail = (
            hist_ret_base if co_hist_ret_detail == 0.0
            else 0.7 * hist_ret_base + 0.3 * co_hist_ret_detail
        )

        # -- "bulk table" historical-return: same for every company (only
        #    regime varies there, and it varies BY the slider, not a
        #    per-company baseline, so nothing to precompute for it) --
        actual_hist_ret_bulk = (
            hist_ret_base if live_hist_ret == 0.0
            else 0.7 * hist_ret_base + 0.3 * live_hist_ret
        )

        companies_out.append({
            "company": company,
            "ticker": ticker,
            "beta": beta,
            "rhetoricSensitivity": rhet_sens,
            "baseDrift": base_drift,
            "sentimentBase": round(float(sent_base), 4),
            "topicBase": round(float(topic_base), 4),
            "regimeBase": regime_base,
            "actualRegimeDetail": actual_regime_detail,
            "actualHistoricalReturnDetail": round(float(actual_hist_ret_detail), 6),
            "actualHistoricalReturnBulk": round(float(actual_hist_ret_bulk), 6),
            "momentum5d": round(float(m5), 6),
            "momentum10d": round(float(m10), 6),
            "currentPrice": current_price,
            "llmStrength": round(llm_strength, 4) if llm_strength is not None else None,
            "llmSentiment": round(llm_sentiment, 4) if llm_sentiment is not None else None,
        })

    missing_market_data = [
        c["company"] for c in companies_out if c["currentPrice"] is None
    ]

    write_json("stage6_predictions/companies.json", {
        "companies": companies_out,
        "missingMarketData": missing_market_data,
    })

    # -- per-sector baselines -----------------------------------------------
    #
    # IMPORTANT: unlike companies, NEITHER regime NOR historical-return is
    # slider-driven for sectors at all -- traced against
    # backend/routers/predictions.py's sector_predictions(): it accepts a
    # `regime` query param but never forwards it into
    # get_all_sector_predictions() (whose signature has no regime_label
    # parameter in the first place); and the sector_returns table it does
    # pass is keyed by TICKER, not sector name, so the sector-name lookup
    # inside get_all_sector_predictions() never matches and always falls
    # through to each sector's own regime_base/ret5_base/ret10_base below.
    # regimeBase/historicalReturn5dBase/historicalReturn10dBase are used
    # AS-IS client-side, with no further blending.
    sectors_out = []
    for sector, constituent_companies in pe.SECTOR_COMPANIES.items():
        beta, rhet_sens, base_drift = pe.SECTOR_PROFILES.get(sector, (1.0, 0.75, 0.0))
        sec_ticker = pe.SECTOR_TICKER_MAP.get(sector, '^NSEI')

        sent_base = pe._resolve_company_sentiment(sec_ticker, 0.0)
        topic_base = pe._resolve_company_topic_strength(sec_ticker, 0.5)
        regime_base = pe._resolve_company_regime(sec_ticker, "Neutral")
        ret5_base, ret10_base = pe._resolve_sector_historical_return(sector)

        llm_strengths, llm_sentiments = [], []
        for co_name in constituent_companies:
            co_ticker = pe.COMPANY_UNIVERSE.get(co_name, "")
            if not co_ticker:
                continue
            s, se = pe._resolve_company_llm_signal(co_ticker)
            if s is not None:
                llm_strengths.append(s)
                llm_sentiments.append(se)
        if llm_strengths:
            w = pe._groq_blend_weight()
            topic_base = (1 - w) * topic_base + w * float(np.mean(llm_strengths))
            sent_base = (1 - w) * sent_base + w * float(np.mean(llm_sentiments))

        all_m5, all_m10, constituent_prices = [], [], {}
        for co_name in constituent_companies:
            co_ticker = pe.COMPANY_UNIVERSE.get(co_name, "")
            if not co_ticker:
                continue
            m30, m10, m5 = pe._fetch_price_momentum(co_ticker)
            all_m5.append(m5)
            all_m10.append(m10)
            price = pe._fetch_current_price(co_ticker)
            if price:
                constituent_prices[co_name] = price

        sectors_out.append({
            "sector": sector,
            "ticker": sec_ticker,
            "constituentCompanies": constituent_companies,
            "beta": beta,
            "rhetoricSensitivity": rhet_sens,
            "baseDrift": base_drift,
            "sentimentBase": round(float(sent_base), 4),
            "topicBase": round(float(topic_base), 4),
            "regimeBase": regime_base,
            "historicalReturn5dBase": round(float(ret5_base), 6),
            "historicalReturn10dBase": round(float(ret10_base), 6),
            "momentum5d": round(float(np.mean(all_m5)), 6) if all_m5 else 0.0,
            "momentum10d": round(float(np.mean(all_m10)), 6) if all_m10 else 0.0,
            "constituentPrices": constituent_prices,
        })

    write_json("stage6_predictions/sectors.json", {"sectors": sectors_out})

    # -- static constants needed by the ported JS formula --------------------
    write_json("stage6_predictions/constants.json", {
        "regimeMultiplier": pe.REGIME_MULTIPLIER,
        "horizons": {str(k): v for k, v in pe.HORIZONS.items()},
        "liveDefaults": {
            "sentiment": round(live_sentiment, 3),
            "topicStrength": round(live_topic_str, 3),
            "regime": live_regime_raw,
            "historicalReturn": live_hist_ret,
        },
    })

    # -- static company HQ locations (reused by the Stage 6 sector map AND
    #    Stage 7's Company Locations tab -- same source of truth) -----------
    from geo_dashboard import COMPANY_LOCATIONS
    write_json("stage6_predictions/company_locations.json", COMPANY_LOCATIONS)

    _manifest["stage6_predictions"] = {
        "companies": list(pe.COMPANY_UNIVERSE.keys()),
        "sectors": list(pe.SECTOR_COMPANIES.keys()),
        "missingMarketData": missing_market_data,
    }


# ===========================================================================
# Stage 7 -- Global Influence Map
# ===========================================================================

@step("stage7 geo (country risk + shocks + company map)")
def export_geo_core():
    from backend.routers import geo

    write_json("stage7_geo/country_risk.json", geo.country_risk())
    write_json("stage7_geo/shocks.json", geo.shocks(shock_type="Inflation"))
    write_json("stage7_geo/company_map.json", geo.company_map())


@step("stage7 geo (World Bank indicators, raw tables)")
def export_geo_indicators():
    from geo_dashboard import fetch_wb_data, WB_INDICATORS, KEY_COUNTRIES

    slugs = {}
    for name, code in WB_INDICATORS.items():
        df = fetch_wb_data(code, name)
        slug = slugify(name)
        write_json(f"stage7_geo/indicator/{slug}.json", {
            "indicator": name,
            "simulated": bool(df.attrs.get("simulated")),
            # Raw table -- the frontend builds the choropleth trace
            # client-side from this so the country-filter can reshape it
            # without needing a backend (see module docstring).
            "rows": json.loads(df.to_json(orient="records")),
        })
        slugs[name] = slug

    write_json("stage7_geo/indicators.json", {
        "indicators": list(WB_INDICATORS.keys()),
        "keyCountries": KEY_COUNTRIES,
        "slugs": slugs,
    })

    _manifest["stage7_geo"] = {
        "indicators": list(WB_INDICATORS.keys()),
        "indicatorSlugs": slugs,
        "keyCountries": KEY_COUNTRIES,
    }


# ===========================================================================
# Stage 8 -- Global Preview
# ===========================================================================

@step("stage8 preview (full unfiltered replay -- this is the slow one)")
def export_preview():
    from backend.routers import preview

    # source=None, company=None -> full unfiltered dataset (~6.3k rows
    # today). The frontend's Source/Company dropdowns filter this
    # client-side (see module docstring).
    data = preview.get_preview(source=None, company=None)
    write_json("stage8_preview/detail.json", data)
    _manifest["stage8_preview"] = {"empty": data.get("empty", True)}


# ===========================================================================

def main():
    print(f"Exporting static data to {OUT_DIR}")
    os.makedirs(OUT_DIR, exist_ok=True)

    export_status()
    export_overview()
    export_ingestion()
    export_nlp()
    export_market_impact()
    export_regime()
    export_company_analytics()
    export_predictions()
    export_geo_core()
    export_geo_indicators()
    export_preview()

    _manifest["generatedAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json("manifest.json", _manifest)

    total_bytes = sum(sz for _, sz in _sizes)
    print(f"\nWrote {len(_sizes)} files, {total_bytes / 1024 / 1024:.2f} MB total.")
    largest = sorted(_sizes, key=lambda x: -x[1])[:10]
    print("Largest files:")
    for path, sz in largest:
        print(f"  {sz / 1024:8.1f} KB  {path}")

    if _errors:
        print(f"\n{len(_errors)} export step(s) FAILED:")
        for label, tb in _errors:
            print(f"--- {label} ---\n{tb}")
        sys.exit(1)


if __name__ == "__main__":
    main()
