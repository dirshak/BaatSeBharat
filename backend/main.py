"""
FastAPI app for BaatSeBharat. Mirrors App.py's original module setup
(src/ and TradingAgents/ on sys.path) and mounts one router per stage
under /api, then serves the built React app (frontend/dist) for
everything else.
"""
import os
import sys

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.dirname(_BACKEND_DIR)
sys.path.insert(0, os.path.join(_APP_DIR, 'src'))
sys.path.insert(0, os.path.join(_APP_DIR, 'TradingAgents'))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

try:
    from tradingagents.dataflows import yf_cache_patch  # noqa: F401
except Exception:
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.routers import (
    status, pipeline, ingestion, nlp, market_impact,
    regime, company_analytics, predictions, geo, preview,
)

app = FastAPI(title="BaatSeBharat API")

# Allows `npm run dev` (Vite on :5173) to hit this API directly during
# frontend development; harmless once everything is served same-origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

for r in [status, pipeline, ingestion, nlp, market_impact, regime,
          company_analytics, predictions, geo, preview]:
    app.include_router(r.router, prefix="/api")

_FRONTEND_DIST = os.path.join(_APP_DIR, "frontend", "dist")

if os.path.isdir(_FRONTEND_DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(_FRONTEND_DIST, "assets")), name="assets")

    @app.get("/{full_path:path}")
    def spa_catch_all(full_path: str):
        candidate = os.path.join(_FRONTEND_DIST, full_path)
        if full_path and os.path.isfile(candidate):
            return FileResponse(candidate)
        return FileResponse(os.path.join(_FRONTEND_DIST, "index.html"))
