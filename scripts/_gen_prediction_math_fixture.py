"""
One-off helper (not part of the pipeline): calls the REAL backend router
functions (backend/routers/predictions.py) -- not prediction_engine.py
directly -- across a grid of slider combinations, exactly reproducing
what a live request would compute for the company-detail, company-bulk,
and sector views. Written next to the exported baselines so
scripts/verify_prediction_math.mjs can diff the JS port against them.
Deleted after verification; not referenced by any other script.
"""
import os
import sys
import json

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'src'))
sys.path.insert(0, _ROOT)

from backend.routers import predictions as P
import prediction_engine as pe

SENTIMENT_GRID = [0.0, 1.0, -1.0, 0.35, -0.6]
TOPIC_GRID = [0.5, 1.0, 0.0, 0.7, 0.2]
REGIME_GRID = ["Neutral", "Bull", "Bear", "Volatile", "Stable"]

cases = list(zip(SENTIMENT_GRID, TOPIC_GRID, REGIME_GRID))

# -- company detail: company_prediction() (regime slider ignored there,
#    but pass it through anyway so the fixture proves that too) --
detail_out = []
for company in pe.COMPANY_UNIVERSE:
    for sentiment, topic, regime in cases:
        result = P.company_prediction(company=company, sentiment=sentiment, topic=topic, regime=regime, use_llm=False)
        detail_out.append({
            "company": company,
            "sliders": {"sentiment": sentiment, "topic": topic, "regime": regime},
            "expected": result["prediction"],
        })

# -- company bulk: all_company_predictions() --
bulk_out = []
for sentiment, topic, regime in cases:
    result = P.all_company_predictions(sentiment=sentiment, topic=topic, regime=regime)
    bulk_out.append({
        "sliders": {"sentiment": sentiment, "topic": topic, "regime": regime},
        "expected": result["predictions"],  # list of {Company, Signal, Confidence, Score, 1D/5D/10D %}
    })

# -- sectors: sector_predictions() --
sector_out = []
for sentiment, topic, regime in cases:
    result = P.sector_predictions(sentiment=sentiment, topic=topic, regime=regime)
    sector_out.append({
        "sliders": {"sentiment": sentiment, "topic": topic, "regime": regime},
        "expected": result["sectors"],  # list of {Sector, Signal, Conf, Score, 1D/5D/10D %}
    })

out_path = os.path.join(_ROOT, "frontend", "public", "data", "_verify_fixture.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({"detail": detail_out, "bulk": bulk_out, "sectors": sector_out}, f)

print(f"Wrote {len(detail_out)} detail + {len(bulk_out)} bulk + {len(sector_out)} sector cases to {out_path}")
