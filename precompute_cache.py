# precompute_cache.py
# Run ONCE to generate all small cached CSVs the dashboard reads.
# Re-run whenever the source data changes.
# ===========================================================

import pandas as pd
import os

DATA_DIR   = "content"
CACHE_DIR  = os.path.join(DATA_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

IMPACT_CSV = os.path.join(DATA_DIR, "speech_market_impact_full.csv")
TOPIC_DST  = os.path.join(DATA_DIR, "topic_dataset.csv")
TOPICS_CSV = os.path.join(DATA_DIR, "final_topics.csv")

REGIME_THRESH = 0.005   # >+0.5% = Bull, <-0.5% = Bear

print("Loading raw impact data...")
df = pd.read_csv(IMPACT_CSV, parse_dates=["date"])
df = df.dropna(subset=["return_5d", "date"])

# ──────────────────────────────────────────────────────────
# helper: return-time label from 5d vs 10d capture ratio
# ──────────────────────────────────────────────────────────
def return_time_label(ret5, ret10):
    """Classify how fast returns materialise."""
    if abs(ret10) < 1e-6:
        return "~1 Week"
    capture = abs(ret5) / abs(ret10)
    if capture >= 0.65:
        return "~1 Week"
    elif capture >= 0.35:
        return "1-2 Weeks"
    else:
        return "~2 Weeks"

def add_topic_activity(base_df, period_col):
    """Add speech_count and return_time_label to aggregated DF."""
    activity = (
        df.groupby(["sector", period_col])["speech_id"]
        .nunique()
        .reset_index()
        .rename(columns={"speech_id": "speech_count"})
    )
    merged = base_df.merge(activity, on=["sector", period_col], how="left")
    merged["speech_count"] = merged["speech_count"].fillna(0).astype(int)
    merged["return_time"] = merged.apply(
        lambda r: return_time_label(r["return_5d"], r["return_10d"]), axis=1
    )
    return merged

# ──────────────────────────────────────────────────────────
# 1. Sector × Weekly aggregation
# ──────────────────────────────────────────────────────────
print("Computing weekly sector returns...")
df["week"] = df["date"].dt.to_period("W").astype(str)

weekly_base = (
    df.groupby(["sector", "week"])[["return_5d", "return_10d"]]
    .mean()
    .reset_index()
    .sort_values(["sector", "week"])
)
weekly = add_topic_activity(weekly_base, "week")
weekly.to_csv(os.path.join(CACHE_DIR, "sector_weekly.csv"), index=False)
print(f"  -> sector_weekly.csv  ({len(weekly)} rows, cols: {list(weekly.columns)})")

# ──────────────────────────────────────────────────────────
# 2. Sector × Quarterly aggregation
# ──────────────────────────────────────────────────────────
print("Computing quarterly sector returns...")
df["quarter"] = df["date"].dt.to_period("Q").astype(str)

quarterly_base = (
    df.groupby(["sector", "quarter"])[["return_5d", "return_10d"]]
    .mean()
    .reset_index()
    .sort_values(["sector", "quarter"])
)
quarterly = add_topic_activity(quarterly_base, "quarter")
quarterly.to_csv(os.path.join(CACHE_DIR, "sector_quarterly.csv"), index=False)
print(f"  -> sector_quarterly.csv  ({len(quarterly)} rows, cols: {list(quarterly.columns)})")

# ──────────────────────────────────────────────────────────
# 3. Regime summary table (both granularities)
# ──────────────────────────────────────────────────────────
print("Computing regime summary tables...")

def build_regime_table(agg_df, period_col):
    rows = []
    for sec in agg_df["sector"].unique():
        sdf = agg_df[agg_df["sector"] == sec].copy()
        last_ret    = sdf["return_5d"].iloc[-1]
        last_period = sdf[period_col].iloc[-1]
        avg_ret     = sdf["return_5d"].mean()
        bull_pct    = (sdf["return_5d"] > REGIME_THRESH).mean() * 100
        bear_pct    = (sdf["return_5d"] < -REGIME_THRESH).mean() * 100

        if last_ret > REGIME_THRESH:
            regime = "Bull"
        elif last_ret < -REGIME_THRESH:
            regime = "Bear"
        else:
            regime = "Neutral"

        rows.append({
            "sector":       sec,
            "regime":       regime,
            "last_period":  last_period,
            "last_ret_5d":  round(last_ret * 100, 3),
            "avg_ret_5d":   round(avg_ret * 100, 3),
            "bull_pct":     round(bull_pct, 1),
            "bear_pct":     round(bear_pct, 1),
            "neutral_pct":  round(100 - bull_pct - bear_pct, 1),
        })
    return pd.DataFrame(rows)

regime_w = build_regime_table(weekly,    "week")
regime_q = build_regime_table(quarterly, "quarter")

regime_w.to_csv(os.path.join(CACHE_DIR, "regime_weekly.csv"),    index=False)
regime_q.to_csv(os.path.join(CACHE_DIR, "regime_quarterly.csv"), index=False)
print(f"  -> regime_weekly.csv / regime_quarterly.csv")

# ──────────────────────────────────────────────────────────
# 4. Sector overall averages (for topic-strength overlay)
# ──────────────────────────────────────────────────────────
print("Computing sector overall averages...")
sector_avg = (
    df.groupby("sector")[["return_5d", "return_10d"]]
    .mean()
    .reset_index()
)
sector_avg.to_csv(os.path.join(CACHE_DIR, "sector_avg.csv"), index=False)
print(f"  -> sector_avg.csv  ({len(sector_avg)} rows)")

# ──────────────────────────────────────────────────────────
# 5. Sentiment timeline (from topic_dataset)
# ──────────────────────────────────────────────────────────
print("Computing sentiment timeline...")
try:
    # topic_dataset can be huge - read only needed columns
    ds = pd.read_csv(
        TOPIC_DST,
        usecols=["date", "positive", "negative"],
        parse_dates=["date"],
    )
    ds = ds.dropna(subset=["date"])
    ds["sentiment"] = ds["positive"] - ds["negative"]
    timeline = (
        ds.groupby("date")["sentiment"]
        .mean()
        .reset_index()
        .sort_values("date")
    )
    timeline.to_csv(os.path.join(CACHE_DIR, "sentiment_timeline.csv"), index=False)
    print(f"  -> sentiment_timeline.csv  ({len(timeline)} rows)")
except Exception as e:
    print(f"  WARNING: Could not compute sentiment timeline: {e}")

# ──────────────────────────────────────────────────────────
# 6. Topic enrichment (short labels)
# ──────────────────────────────────────────────────────────
print("Enriching topic data...")
try:
    topics = pd.read_csv(TOPICS_CSV)
    topics["short_kw"] = (
        topics["keywords"]
        .str.split(",")
        .str[:4]
        .str.join(", ")
    )
    topics.to_csv(os.path.join(CACHE_DIR, "topics_enriched.csv"), index=False)
    print(f"  -> topics_enriched.csv  ({len(topics)} rows)")
except Exception as e:
    print(f"  WARNING: Could not enrich topics: {e}")

# ──────────────────────────────────────────────────────────
# 7. Future Regime Forecast
#    Method A: Markov Chain on Bull/Bear/Neutral states
#    Method B: Linear-trend return extrapolation (EWMA slope)
#    Horizons : 4 quarters  AND  8 weeks  forward
# ──────────────────────────────────────────────────────────
print("Computing future regime forecasts...")

import numpy as np

STATES   = ["Bull", "Bear", "Neutral"]
STATE_IDX = {s: i for i, s in enumerate(STATES)}

def label_regime(ret):
    if ret > REGIME_THRESH:  return "Bull"
    if ret < -REGIME_THRESH: return "Bear"
    return "Neutral"

def build_markov(regimes):
    """3×3 transition matrix from a sequence of regime labels."""
    mat = np.zeros((3, 3))
    for a, b in zip(regimes[:-1], regimes[1:]):
        mat[STATE_IDX[a], STATE_IDX[b]] += 1
    # row-normalise (add tiny epsilon to avoid zero rows)
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return mat / row_sums

def next_period_label(current_label, n_ahead):
    """
    Generate the next N quarter/week labels after current_label.
    Supports format:  2026Q2   or   2026-W18/2026-W24
    """
    labels = []
    if "Q" in current_label:
        yr, q = int(current_label[:4]), int(current_label[5])
        for _ in range(n_ahead):
            q += 1
            if q > 4:
                q = 1
                yr += 1
            labels.append(f"{yr}Q{q}")
    else:
        # Week format: take last 4-digit year + ISO week from the string
        import re
        m = re.search(r"(\d{4})-W(\d+)", current_label)
        if m:
            yr, wk = int(m.group(1)), int(m.group(2))
        else:
            yr, wk = 2026, 1
        for _ in range(n_ahead):
            wk += 1
            if wk > 52:
                wk = 1
                yr += 1
            labels.append(f"{yr}-W{wk:02d}")
    return labels

def forecast_sector(sdf, period_col, horizon):
    """
    Returns a DataFrame with columns:
      period, bull_prob, bear_prob, neutral_prob,
      predicted_regime, forecasted_return, ret_lower, ret_upper
    """
    sdf = sdf.sort_values(period_col).copy()
    regimes = sdf["return_5d"].apply(label_regime).tolist()

    # ── Markov chain ──────────────────────────────────────
    T = build_markov(regimes)
    # current state distribution (one-hot of last known state)
    state_vec = np.zeros(3)
    state_vec[STATE_IDX[regimes[-1]]] = 1.0

    future_periods = next_period_label(sdf[period_col].iloc[-1], horizon)
    rows = []
    for fp in future_periods:
        state_vec = state_vec @ T          # propagate one step
        bull_p, bear_p, neut_p = state_vec
        pred_regime = STATES[int(np.argmax(state_vec))]
        rows.append({
            "period":           fp,
            "bull_prob":        round(float(bull_p) * 100, 1),
            "bear_prob":        round(float(bear_p) * 100, 1),
            "neutral_prob":     round(float(neut_p) * 100, 1),
            "predicted_regime": pred_regime,
        })

    # ── Linear return extrapolation ───────────────────────
    y = sdf["return_5d"].values * 100          # in %
    x = np.arange(len(y))
    coeffs = np.polyfit(x, y, 1)              # slope, intercept
    slope, intercept = coeffs

    # rolling std of last 8 periods as uncertainty band
    recent_std = float(np.std(y[-8:])) if len(y) >= 8 else float(np.std(y))

    for i, r in enumerate(rows):
        t = len(y) + i
        fc = slope * t + intercept
        rows[i]["forecasted_return"] = round(fc, 3)
        rows[i]["ret_lower"]         = round(fc - 1.96 * recent_std, 3)
        rows[i]["ret_upper"]         = round(fc + 1.96 * recent_std, 3)

    return pd.DataFrame(rows)


forecast_rows_q, forecast_rows_w = [], []

for sec in quarterly["sector"].unique():
    sdf_q = quarterly[quarterly["sector"] == sec].copy()
    fdf_q = forecast_sector(sdf_q, "quarter", horizon=4)
    fdf_q.insert(0, "sector", sec)
    forecast_rows_q.append(fdf_q)

for sec in weekly["sector"].unique():
    sdf_w = weekly[weekly["sector"] == sec].copy()
    fdf_w = forecast_sector(sdf_w, "week", horizon=8)
    fdf_w.insert(0, "sector", sec)
    forecast_rows_w.append(fdf_w)

forecast_q = pd.concat(forecast_rows_q, ignore_index=True)
forecast_w = pd.concat(forecast_rows_w, ignore_index=True)

forecast_q.to_csv(os.path.join(CACHE_DIR, "forecast_quarterly.csv"), index=False)
forecast_w.to_csv(os.path.join(CACHE_DIR, "forecast_weekly.csv"),    index=False)

print(f"  -> forecast_quarterly.csv  ({len(forecast_q)} rows)")
print(f"  -> forecast_weekly.csv     ({len(forecast_w)} rows)")

print("\nAll cache files written to:", CACHE_DIR)
print("Done!")
