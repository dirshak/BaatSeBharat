"""
BaatSeBharat — Topic Strength vs Sector Returns
================================================
Part A:  Group speeches by week/quarter -> identify dominant sector via BERTopic
         -> predict sector baseline using only past data (no lookahead)

Part B:  Compute topic strength = avg(topic_prob) × (1 + compound_sentiment)
         -> direction: POSITIVE if strength > 0, NEGATIVE if < 0

Plot:    Dual line charts per sector
         • Line 1: Quarterly sector return (actual)
         • Line 2: Quarterly topic strength (from speeches)
         The lag between peaks reveals how speech rhetoric translates to market movement.
"""

import os, sys, warnings, sqlite3
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB_PATH    = "./data/market_rhetoric.db"
OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Sector configuration ──────────────────────────────────────────────────────
SECTOR_KEYWORDS = {
    "Banking":      ["bank", "reserve", "financial", "system", "risk", "credit", "rbi", "monetary", "interest"],
    "IT":           ["digital", "technology", "startup", "innovation", "software", "internet", "data", "ai"],
    "Pharma":       ["health", "medicine", "covid", "hospital", "vaccine", "doctor", "pharma"],
    "Auto":         ["vehicle", "road", "transport", "highway", "electric", "automobile", "maruti"],
    "Energy":       ["energy", "solar", "power", "oil", "gas", "renewable", "electricity"],
    "Agriculture":  ["farmer", "agriculture", "crop", "village", "water", "kisan", "food", "rural"],
    "Broad Market": ["economy", "growth", "gdp", "employment", "trade", "export", "friend", "countryman"],
}

SECTOR_TICKERS = {
    "Banking":      ["HDFCBANK.NS", "ICICIBANK.NS", "^NSEBANK"],
    "IT":           ["TCS.NS", "INFY.NS", "^CNXIT"],
    "Pharma":       ["SUNPHARMA.NS", "^CNXPHARMA"],
    "Auto":         ["MARUTI.NS", "^CNXAUTO"],
    "Energy":       ["RELIANCE.NS", "^CNXENERGY"],
    "Agriculture":  ["^NSEI", "^BSESN"],
    "Broad Market": ["^NSEI", "^BSESN", "^GSPC"],
}

def classify_topic(keywords: list[str]) -> str:
    kw_str = " ".join(str(k) for k in keywords).lower()
    best, best_score = "Broad Market", 0
    for sector, kws in SECTOR_KEYWORDS.items():
        score = sum(1 for kw in kws if kw in kw_str)
        if score > best_score:
            best_score, best = score, sector
    return best


# ─── Load Data ────────────────────────────────────────────────────────────────
def load_data():
    conn = sqlite3.connect(DB_PATH)

    # Speeches with topic distributions
    speeches = pd.read_sql_query("""
        SELECT s.id, s.date, s.source, s.speaker,
               td.topic_id, td.probability,
               sent.compound, sent.positive, sent.negative, sent.neutral
        FROM speeches s
        LEFT JOIN topic_distributions td
               ON s.id = td.speech_id AND td.model_name = 'BERTopic'
        LEFT JOIN sentiment_scores sent
               ON s.id = sent.speech_id
        WHERE s.date IS NOT NULL AND s.date != ''
          AND s.processed_text IS NOT NULL
    """, conn)

    # Market data
    market = pd.read_sql_query("""
        SELECT date, ticker, sector, close, returns
        FROM market_data
        WHERE returns IS NOT NULL
        ORDER BY date
    """, conn)

    # BERTopic topic keywords
    topic_kws = {}
    try:
        ti = pd.read_csv("./data/processed/bertopic_topic_info.csv")
        for _, row in ti.iterrows():
            tid  = int(row.get("Topic", -1))
            name = str(row.get("Name", ""))
            parts = [p for p in name.split("_")[1:] if p]
            topic_kws[tid] = parts
    except Exception:
        pass

    conn.close()
    return speeches, market, topic_kws


# ─── Part A: Quarterly speech grouping -> sector + past baseline ───────────────
def build_quarterly_speech_profile(speeches: pd.DataFrame, topic_kws: dict) -> pd.DataFrame:
    """
    For each quarter, determine:
    - dominant sector (via BERTopic topic keywords)
    - avg topic probability
    - avg sentiment compound
    - topic_strength = avg_prob × (1 + avg_compound)   [−2..+2 range]
    """
    speeches["date"] = pd.to_datetime(speeches["date"], format="mixed", errors="coerce")
    speeches = speeches.dropna(subset=["date"])

    # Quarter period
    speeches["quarter"] = speeches["date"].dt.to_period("Q")

    records = []
    for quarter, grp in speeches.groupby("quarter"):
        # Collect topic keywords across all speeches in this quarter
        all_kws = []
        for tid in grp["topic_id"].dropna().astype(int).unique():
            all_kws.extend(topic_kws.get(tid, []))

        # Classify dominant sector
        sector = classify_topic(all_kws)

        # Average probability (topic confidence)
        avg_prob = float(grp["probability"].dropna().mean()) if grp["probability"].notna().any() else 0.0

        # Average sentiment
        avg_compound = float(grp["compound"].dropna().mean()) if grp["compound"].notna().any() else 0.0

        # Topic strength: confidence × sentiment polarity amplifier
        topic_strength = avg_prob * (1.0 + avg_compound)

        # Source distribution
        sources = grp["source"].value_counts().to_dict()

        records.append({
            "quarter":       str(quarter),
            "quarter_dt":    quarter.start_time,
            "sector":        sector,
            "avg_prob":      avg_prob,
            "avg_compound":  avg_compound,
            "topic_strength": topic_strength,
            "n_speeches":    len(grp),
            "sources":       str(sources),
            "keywords":      ", ".join(all_kws[:6]),
        })

    return pd.DataFrame(records).sort_values("quarter_dt")


# ─── Part A: Past-data baseline prediction ────────────────────────────────────
def compute_past_baseline(market: pd.DataFrame, sector: str, as_of_date: pd.Timestamp) -> float:
    """
    Predict sector return for a quarter using ONLY data prior to that quarter.
    Method: rolling 4-quarter average of the sector's mean quarterly return.
    """
    tickers = SECTOR_TICKERS.get(sector, SECTOR_TICKERS["Broad Market"])
    sector_data = market[market["ticker"].isin(tickers)].copy()
    sector_data["date"] = pd.to_datetime(sector_data["date"], errors="coerce")
    past_data = sector_data[sector_data["date"] < as_of_date]

    if past_data.empty:
        return 0.0

    past_data["quarter"] = past_data["date"].dt.to_period("Q")
    quarterly_returns = past_data.groupby("quarter")["returns"].mean() * 65  # scale to ~quarter return

    # Use last 4 quarters as baseline prediction
    last_4 = quarterly_returns.tail(4)
    return float(last_4.mean()) if not last_4.empty else 0.0


# ─── Build actual quarterly sector returns ────────────────────────────────────
def build_quarterly_sector_returns(market: pd.DataFrame) -> pd.DataFrame:
    """
    For each (quarter, sector) pair, compute the actual mean return.
    """
    market["date"] = pd.to_datetime(market["date"], errors="coerce")
    market = market.dropna(subset=["date"])
    market["quarter"] = market["date"].dt.to_period("Q")

    # Map tickers to sectors
    ticker_sector = {}
    for sector, tickers in SECTOR_TICKERS.items():
        for t in tickers:
            ticker_sector[t] = sector

    market["mapped_sector"] = market["ticker"].map(ticker_sector)
    market = market.dropna(subset=["mapped_sector"])

    qsr = market.groupby(["quarter", "mapped_sector"])["returns"].mean().reset_index()
    qsr.columns = ["quarter", "sector", "mean_daily_return"]
    # Convert mean daily return to approximate quarterly cumulative (65 trading days/quarter)
    qsr["quarterly_return_pct"] = qsr["mean_daily_return"] * 65 * 100
    qsr["quarter_dt"] = qsr["quarter"].apply(lambda p: p.start_time)
    return qsr.sort_values("quarter_dt")


# ─── Plot: Dual time-series per sector ────────────────────────────────────────
def plot_topic_strength_vs_returns(
    speech_profile: pd.DataFrame,
    sector_returns: pd.DataFrame,
    topic_kws: dict,
):
    """
    For each sector: dual-axis chart
      Primary Y  (left,  cyan):   Actual quarterly sector return (%)
      Secondary Y (right, orange): Normalised topic strength score
    """
    active_sectors = speech_profile["sector"].unique()
    n_sectors = len(active_sectors)
    if n_sectors == 0:
        print("No sectors found.")
        return

    fig = make_subplots(
        rows=n_sectors, cols=1,
        shared_xaxes=False,
        subplot_titles=[
            f"<b>{s} Sector</b> — Topic Strength vs Market Returns" for s in active_sectors
        ],
        vertical_spacing=0.12,
        specs=[[{"secondary_y": True}] for _ in range(n_sectors)],
    )

    C_RET      = "#00D4FF"  # cyan   – sector returns
    C_STR      = "#FF6B35"  # orange – topic strength
    C_BASE     = "#7CFC00"  # green  – past-data baseline
    C_POS_BG   = "rgba(0,212,100,0.12)"
    C_NEG_BG   = "rgba(255,80,80,0.12)"

    lag_summary = []

    for row_idx, sector in enumerate(active_sectors, start=1):
        sp = speech_profile[speech_profile["sector"] == sector].copy()
        sr = sector_returns[sector_returns["sector"] == sector].copy()
        if sp.empty or sr.empty:
            continue

        # ── Normalise topic strength to −1 … +1 for right axis ───────────────
        ts  = sp["topic_strength"].values
        rng = max(ts.max() - ts.min(), 1e-6)
        sp["ts_norm"] = (ts - ts.min()) / rng * 2 - 1

        # ── Past-data baseline prediction (Part A) ────────────────────────────
        baselines = []
        for _, srow in sp.iterrows():
            bl = compute_past_baseline(
                sr.rename(columns={"mean_daily_return": "returns",
                                   "quarter_dt":        "date"})
                  .assign(ticker=SECTOR_TICKERS[sector][0]),
                sector,
                srow["quarter_dt"],
            )
            baselines.append(bl)
        sp["baseline_pred"] = baselines

        # ── Background shading: positive / negative quarters ─────────────────
        for _, srow in sp.iterrows():
            fill = C_POS_BG if srow["ts_norm"] >= 0 else C_NEG_BG
            fig.add_vrect(
                x0=str(srow["quarter_dt"])[:7],
                x1=str(srow["quarter_dt"] + pd.DateOffset(months=3))[:7],
                fillcolor=fill, layer="below", line_width=0,
                row=row_idx, col=1,
            )

        show_legend = (row_idx == 1)   # only show legend entries once

        # ── Primary Y: actual sector return ───────────────────────────────────
        fig.add_trace(go.Scatter(
            x=sr["quarter_dt"].dt.strftime("%Y-%m-%d"), y=sr["quarterly_return_pct"],
            mode="lines+markers",
            name="Actual Sector Return (%)",
            line=dict(color=C_RET, width=2.5),
            marker=dict(size=6, symbol="circle"),
            hovertemplate="<b>%{x|%Y}</b><br>Return: %{y:.2f}%<extra></extra>",
            legendgroup="ret", showlegend=show_legend,
        ), row=row_idx, col=1, secondary_y=False)

        # ── Primary Y: past-data baseline ────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=sp["quarter_dt"].dt.strftime("%Y-%m-%d"), y=sp["baseline_pred"],
            mode="lines",
            name="Past-Data Baseline (Part A)",
            line=dict(color=C_BASE, width=1.8, dash="dash"),
            hovertemplate="<b>%{x|%Y}</b><br>Baseline: %{y:.2f}%<extra></extra>",
            legendgroup="base", opacity=0.8, showlegend=show_legend,
        ), row=row_idx, col=1, secondary_y=False)

        # ── Secondary Y: topic strength ───────────────────────────────────────
        hover_kws = sp["keywords"].iloc[0] if len(sp) > 0 else ""
        fig.add_trace(go.Scatter(
            x=sp["quarter_dt"].dt.strftime("%Y-%m-%d"), y=sp["ts_norm"],
            mode="lines+markers",
            name="Topic Strength (Part B)",
            line=dict(color=C_STR, width=2.5, dash="dot"),
            marker=dict(size=9, symbol="diamond",
                        color=C_STR, line=dict(color="white", width=1)),
            hovertemplate=(
                "<b>%{x|%Y}</b><br>"
                "Topic Strength: %{y:.3f}<br>"
                f"Keywords: {hover_kws}"
                "<extra></extra>"
            ),
            legendgroup="str", showlegend=show_legend,
        ), row=row_idx, col=1, secondary_y=True)

        # ── Lag annotation ────────────────────────────────────────────────────
        try:
            merged = pd.merge_asof(
                sr.sort_values("quarter_dt"),
                sp[["quarter_dt", "ts_norm"]].sort_values("quarter_dt"),
                on="quarter_dt", direction="nearest",
            )
            if len(merged) > 4:
                ret_peak_q = merged.loc[merged["quarterly_return_pct"].idxmax(), "quarter_dt"]
                str_peak_q = sp.loc[sp["ts_norm"].idxmax(), "quarter_dt"]
                # Approximate quarters between peaks
                lag_q = round((ret_peak_q - str_peak_q).days / 91)
                lag_summary.append({
                    "sector":              sector,
                    "topic_strength_peak": str(str_peak_q)[:7],
                    "market_return_peak":  str(ret_peak_q)[:7],
                    "lag_quarters":        lag_q,
                })
                # Convert to string for Plotly serialization
                ret_peak_q_str = ret_peak_q.strftime("%Y-%m-%d")
                fig.add_annotation(
                    x=ret_peak_q_str,
                    y=merged["quarterly_return_pct"].max(),
                    text=f"Peak lag: {'+' if lag_q>=0 else ''}{lag_q}Q",
                    showarrow=True, arrowhead=2,
                    arrowcolor="#FFEB3B",
                    font=dict(color="#FFEB3B", size=12),
                    bgcolor="rgba(0,0,0,0.7)",
                    bordercolor="#FFEB3B", borderwidth=1,
                    row=row_idx, col=1, secondary_y=False,
                )
        except Exception:
            pass

        # ── Axis styling ──────────────────────────────────────────────────────
        fig.update_yaxes(
            title_text="<b>Quarterly Return (%)</b>",
            title_font=dict(color=C_RET),
            tickfont=dict(color=C_RET),
            gridcolor="#21262D", zerolinecolor="#555",
            row=row_idx, col=1, secondary_y=False,
        )
        fig.update_yaxes(
            title_text="<b>Topic Strength (norm.)</b>",
            title_font=dict(color=C_STR),
            tickfont=dict(color=C_STR),
            tickformat=".2f", range=[-1.5, 1.5],
            gridcolor="rgba(255,107,53,0.08)",
            zerolinecolor=C_STR, zerolinewidth=1.5,
            row=row_idx, col=1, secondary_y=True,
        )
        fig.update_xaxes(
            gridcolor="#21262D", tickformat="%Y",
            row=row_idx, col=1,
        )

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text="<b>BaatSeBharat: Speech Topic Strength vs Sector Market Returns</b><br>"
                 "<sub>Part A: Past-data baseline | Part B: Topic strength polarity -> Return direction</sub>",
            font=dict(size=20, color="white"),
            x=0.5,
        ),
        paper_bgcolor="#0D1117",
        plot_bgcolor="#161B22",
        font=dict(color="#C9D1D9", family="Inter, sans-serif"),
        height=450 * n_sectors,
        legend=dict(
            orientation="h",
            y=1.02, x=0.5, xanchor="center",
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="#30363D",
            font=dict(size=11),
        ),
        hoverlabel=dict(bgcolor="#21262D", font_color="white", bordercolor="#30363D"),
        margin=dict(t=120, b=60, l=60, r=60),
    )

    # Zero line reference
    fig.add_hline(y=0, line_dash="dash", line_color="#444", line_width=1)

    # Save as HTML
    output_path = os.path.join(OUTPUT_DIR, "topic_strength_vs_sector_returns.html")
    pio.write_html(fig, file=output_path, auto_open=False)
    print(f"\nInteractive chart saved to: {output_path}")

    # Also save as static PNG
    try:
        png_path = os.path.join(OUTPUT_DIR, "topic_strength_vs_sector_returns.png")
        fig.write_image(png_path, width=1600, height=450 * n_sectors, scale=2)
        print(f"Static PNG saved to: {png_path}")
    except Exception as e:
        print(f"  (PNG export skipped: {e} — install kaleido for static export)")

    return lag_summary, fig


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  BaatSeBharat — Topic Strength vs Sector Returns")
    print("=" * 70)

    # 1. Load
    print("\n[1/4] Loading speeches, market data, topic keywords...")
    speeches, market, topic_kws = load_data()
    print(f"  Speeches: {len(speeches)}  |  Market records: {len(market)}")

    # 2. Build quarterly speech profile (Part A + B)
    print("\n[2/4] Building quarterly speech profiles (BERTopic -> Sector -> Strength)...")
    speech_profile = build_quarterly_speech_profile(speeches, topic_kws)
    print(f"  Quarters profiled: {len(speech_profile)}")
    print(speech_profile[["quarter", "sector", "avg_prob", "avg_compound", "topic_strength", "n_speeches"]].to_string(index=False))

    # 3. Build actual quarterly sector returns
    print("\n[3/4] Computing quarterly sector returns from market data...")
    sector_returns = build_quarterly_sector_returns(market)
    print(f"  Sector-quarter pairs: {len(sector_returns)}")

    # 4. Plot
    print("\n[4/4] Generating dual time-series plot...")
    lag_summary, _ = plot_topic_strength_vs_returns(speech_profile, sector_returns, topic_kws)

    # Print lag summary
    print("\n" + "=" * 70)
    print("  LAG ANALYSIS: Quarters between Speech Strength Peak -> Return Peak")
    print("=" * 70)
    if lag_summary:
        for entry in lag_summary:
            sign = "+" if entry["lag_quarters"] >= 0 else ""
            print(f"  {entry['sector']:<15} | Strength peak: {entry['topic_strength_peak']} "
                  f"| Return peak: {entry['market_return_peak']} "
                  f"| Lag: {sign}{entry['lag_quarters']} quarter(s)")
    else:
        print("  Insufficient data for lag analysis.")

    print("\nDone. Open outputs/topic_strength_vs_sector_returns.html in a browser.")


if __name__ == "__main__":
    main()
