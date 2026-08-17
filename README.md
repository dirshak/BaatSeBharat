# BaatSeBharat — Leadership Rhetoric Driven Market Intelligence

Quantifies how leadership speeches (US Federal Reserve, European Central
Bank, and India's "Mann Ki Baat") move financial markets: scrapes
transcripts, runs FinBERT sentiment + NMF topic modeling + HMM regime
detection, correlates topics with subsequent market returns, and serves
AI-driven company/sector predictions and geo-intelligence maps across a
9-stage dashboard.

## Architecture

The heavy compute and the site users actually load are deliberately
separated:

```
┌─────────────────────────────┐
│ .github/workflows/pipeline.yml │  (weekly cron + manual dispatch)
│                                 │
│  scripts/run_prototype.py       │  scrape → FinBERT → NMF topics →
│         │                       │  HMM regimes → speech-vs-market
│         ▼                       │  impact → Granger causality
│  scripts/classify_speeches_groq.py │ (optional, needs GROQ_API_KEY)
│         │                       │
│         ▼                       │
│  scripts/export_static_data.py  │  snapshots every stage as JSON
│         │                       │
│         ▼                       │
│  commits frontend/public/data/  ─┼──▶ push to main
└─────────────────────────────────┘         │
                                             ▼
                                  Cloudflare Pages auto-deploys
                                  frontend/dist (static, no backend)
```

- **All modeling logic** lives in `src/` and runs **only** inside the
  scheduled GitHub Actions job (or locally via Docker/`python
  scripts/run_prototype.py`) — never on a live user request.
- **`scripts/export_static_data.py`** calls the same functions
  `backend/routers/*.py` use to build each stage's data/Plotly figures,
  and writes the results as versioned JSON under `frontend/public/data/`.
  Stage 6 (AI Predictions) is the one exception: it exports each
  company/sector's *baseline inputs*, not a precomputed prediction — the
  what-if sliders recompute the composite score instantly in the browser
  (`frontend/src/predictionMath.js`, a verified line-for-line port of
  `src/prediction_engine.py`'s formula) with no server round-trip.
- **The production frontend** (`frontend/`, React + Vite) fetches only
  those static JSON files (`frontend/src/dataClient.js`) and renders them
  with `react-plotly.js`. It needs no backend at all in production.
- **`backend/`** (FastAPI) and **`app.py`** still exist and still work —
  they're the local-dev path only (see below), not part of the production
  deploy.

## Running locally for development

Two ways to run the app locally, depending on what you're working on:

**Full live stack** (backend computes on request, like before the static
migration) — use this when iterating on `backend/`/`src/` logic:
```bash
pip install -r requirements.txt
python app.py
```
This builds the frontend if needed, starts FastAPI, and opens your
browser. The frontend still only ever fetches `frontend/public/data/*`
(see Architecture above) — `app.py`/FastAPI are there so you can hit the
API directly (e.g. `http://127.0.0.1:8000/docs`) while developing, not
because the UI depends on them.

**Frontend-only, against static data** (matches production exactly) —
use this when iterating on `frontend/`:
```bash
python scripts/export_static_data.py   # regenerate frontend/public/data/ from current local data
cd frontend
npm install
npm run dev                            # http://localhost:5173
```

## Data refresh

`.github/workflows/pipeline.yml` is the only place the pipeline runs:

- **Schedule**: weekly, Monday 03:00 UTC (`cron: '0 3 * * 1'`). Ask before
  changing this cadence.
- **Manual**: `Actions` tab → "Refresh pipeline data" → *Run workflow* —
  this replaces the old in-app "Run Pipeline" button, which no longer
  exists (it would have meant live compute on a user-facing request,
  which the static architecture explicitly avoids).
- Runs inside the image built from `Dockerfile`, so CI and a local
  `docker build && docker run` use an identical environment.
- `GROQ_API_KEY` (optional) must be set as a GitHub Actions secret —
  **Settings → Secrets and variables → Actions**. Never commit it. The
  workflow verifies no secret value leaks into `frontend/public/data/`
  before committing.
- Commits only `frontend/public/data/` back to `main` using the default
  `GITHUB_TOKEN` — no extra setup needed.

## Deployment (Cloudflare Pages)

Connect this repo once in the Cloudflare dashboard
(**Workers & Pages → Create → Pages → Connect to Git**):

| Setting | Value |
|---|---|
| Root directory | `frontend` |
| Build command | `npm run build` |
| Build output directory | `dist` |
| Environment variables | none required (see below) |

Once connected, Cloudflare rebuilds and redeploys automatically on every
push to `main` — including the commits the pipeline workflow makes after
each run, so new data goes live automatically with no extra step.

`frontend/wrangler.toml` mirrors this config for manual/local deploys
(`npx wrangler pages deploy` from `frontend/` after `npm run build`) and
local preview (`npx wrangler pages dev`).

`frontend/public/data/` ships bundled with the site by default. If the
dataset ever outgrows comfortable repo size, set `VITE_DATA_BASE_URL` to
an external CDN/R2 URL at build time and re-point
`scripts/export_static_data.py`'s output there instead — the frontend
already reads this env var, no code changes needed either side.

## Project structure

```
scripts/run_prototype.py       # the model pipeline (unchanged by the static migration)
scripts/export_static_data.py  # export-only wrapper around backend/routers/* + src/prediction_engine.py
scripts/classify_speeches_groq.py  # optional Groq LLM topic->company classification
src/                            # all modeling logic (FinBERT, topic modeling, HMM, causal validation, prediction engine)
backend/                        # FastAPI — local dev only, mirrors export_static_data.py's endpoints
frontend/                       # React (Vite) — the production site
  src/dataClient.js             # fetches frontend/public/data/*.json
  src/predictionMath.js         # client-side port of prediction_engine.py's composite score
  src/chartBuilders.js          # client-side Plotly figure builders for prediction-derived charts
  public/data/                  # static JSON, regenerated by scripts/export_static_data.py
.github/workflows/pipeline.yml  # the only place the pipeline runs
Dockerfile                      # reproducible pipeline environment (CI + local)
```

## Testing

```bash
pytest
```

## Authors

Disha Kataria — built under the guidance of Prof. Jugal Manek
