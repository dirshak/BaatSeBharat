"""
BaatSeBharat – BERT Topic Modeling + Stock Influence Pipeline
─────────────────────────────────────────────────────────────
Steps:
  1. Fix NULL speech dates from transcript files
  2. Train BERTopic on processed speeches (SBERT embeddings already saved)
  3. Map topics → economic domains → market sectors
  4. Compute speech-market impact (past vs present prices)
  5. Print a structured report
"""

import os
import re
import sys
import json
import glob
import sqlite3
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─── Config ──────────────────────────────────────────────────────────────────
DB_PATH          = "./data/market_rhetoric.db"
EMBEDDINGS_PATH  = "./data/processed/speech_embeddings.npy"
OUTPUT_DIR       = "content"
FINBERT_MODEL_PATH = "./data/processed/finbert_topic_model.pkl"
TRANSCRIPT_DIR   = "./transcripts/mann_ki_baat"

SECTOR_TOPIC_MAP = {
    "Banking":       ["bank", "finance", "credit", "loan", "rbi", "interest", "monetary", "financial", "reserve", "system", "risk"],
    "IT":            ["digital", "technology", "startup", "innovation", "software", "internet", "data", "ai", "electronics", "semiconductor"],
    "Pharma":        ["health", "medicine", "covid", "hospital", "vaccine", "doctor", "ayurveda", "wellness", "pharma", "yoga"],
    "Auto":          ["vehicle", "road", "transport", "highway", "infrastructure", "electric", "ev", "connectivity", "railway"],
    "Energy":        ["energy", "solar", "power", "oil", "gas", "renewable", "electricity", "hydrogen", "green", "carbon"],
    "Agriculture":   ["farmer", "agriculture", "crop", "village", "water", "kisan", "food", "rural", "harvest", "dairy"],
    "Broad Market":  ["economy", "growth", "gdp", "employment", "trade", "export", "inflation", "atmanirbhar", "vocal", "local"],
}

SECTOR_TICKERS = {
    "Banking":     ["HDFCBANK.NS", "ICICIBANK.NS", "^NSEBANK"],
    "IT":          ["TCS.NS", "INFY.NS", "^CNXIT"],
    "Pharma":      ["SUNPHARMA.NS", "^CNXPHARMA"],
    "Auto":        ["MARUTI.NS", "^CNXAUTO"],
    "Energy":      ["RELIANCE.NS", "^CNXENERGY"],
    "Agriculture": ["^NSEI", "^BSESN"],
    "Broad Market":["^NSEI", "^BSESN", "^GSPC"],
}

# ─── 1. Fix NULL dates from transcript files ──────────────────────────────────
def fix_speech_dates():
    print("\n[1/5] Fixing NULL speech dates from transcript files...")
    conn = sqlite3.connect(DB_PATH)

    txt_files = sorted(glob.glob(os.path.join(TRANSCRIPT_DIR, "mann_ki_baat_*.txt")))
    fixed = 0
    for fpath in txt_files:
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                header = f.readline().strip()

            ep_match  = re.match(r"Episode\s+(\d+)", header)
            ep_num    = int(ep_match.group(1)) if ep_match else None

            parsed_date = None
            date_match  = re.search(r"\(([^)]+)\)", header)
            if date_match:
                date_str = date_match.group(1).strip()
                for fmt in ["%d %b, %Y", "%d %B, %Y", "%d %b %Y", "%d %B %Y", "%B %d, %Y"]:
                    try:
                        parsed_date = datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
                        break
                    except ValueError:
                        continue

            if ep_num and parsed_date:
                title = f"Mann Ki Baat - Episode {ep_num}"
                conn.execute(
                    "UPDATE speeches SET date=?, title=? WHERE source='Mann Ki Baat' AND (episode_number=? OR episode_number IS NULL) AND (date IS NULL OR date='')",
                    (parsed_date, title, ep_num),
                )
                fixed += 1
        except Exception:
            pass

    conn.commit()
    conn.close()
    print(f"    Fixed/updated dates for {fixed} episodes.")

# ─── 2. Train Hybrid Topics (LDA + NMF + BERTopic) ───────────────────────────
def train_hybrid_topics():
    print("\n[2/5] Training High-Granularity Hybrid Ensemble (LDA + NMF + BERTopic)...")
    if not os.path.exists(EMBEDDINGS_PATH):
        print("    ERROR: Embeddings not found at", EMBEDDINGS_PATH)
        return None, None, None

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT id, source, processed_text FROM speeches WHERE processed_text IS NOT NULL AND processed_text != ''",
        conn,
    )
    conn.close()

    if df.empty:
        print("    ERROR: No processed speeches found.")
        return None, None, None

    # Match embeddings
    emb_dict   = np.load(EMBEDDINGS_PATH, allow_pickle=True).item()
    valid_mask = df["id"].isin(emb_dict)
    df         = df[valid_mask].reset_index(drop=True)
    embeddings = np.array([emb_dict[sid] for sid in df["id"]])
    docs       = df["processed_text"].tolist()
    
    # Initialize Hybrid Modeler
    from src.models.topic_modeling import HybridTopicModeler
    modeler  = HybridTopicModeler(n_topics=35)
    
    # LOAD PRE-TRAINED FINBERT MODEL
    from bertopic import BERTopic
    print(f"    Loading pre-trained FinBERT model from {FINBERT_MODEL_PATH}...")
    loaded_model = BERTopic.load(FINBERT_MODEL_PATH)
    modeler.bertopic_model = loaded_model
    
    # Get topics/probs from loaded model
    bert_topics, bert_probs = loaded_model.transform(docs, embeddings)
    
    # Size distributions correctly
    discovered_n = len(loaded_model.get_topic_info()) - 1
    bert_dist = np.zeros((len(docs), discovered_n))
    for i, p in enumerate(bert_probs):
        if i < len(docs):
            bert_dist[i] = p

    # Sync LDA and NMF to this model
    modeler.n_topics = discovered_n
    lda_topics = modeler.fit_lda(docs)
    nmf_topics = modeler.fit_nmf(docs)
    
    dists = {
        'lda': lda_topics,
        'nmf': nmf_topics,
        'bertopic': bert_dist
    }
    consensus = modeler.create_consensus(dists)
    print(f"    Hybrid ensemble synced to FinBERT model ({discovered_n} topics).")

    # Save BERTopic Topic Info
    topic_info = modeler.bertopic_model.get_topic_info()
    topic_info.to_csv(f"{OUTPUT_DIR}/bertopic_topic_info.csv", index=False)
    
    # Persist results to DB
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM topic_distributions WHERE model_name IN ('BERTopic', 'LDA', 'NMF', 'Consensus')")
    
    for i, speech_id in enumerate(df["id"]):
        # Save BERTopic
        t_id = int(bert_topics[i]) if bert_topics[i] != -1 else 0
        conn.execute(
            "INSERT INTO topic_distributions (speech_id, topic_id, probability, model_name) VALUES (?,?,?,?)",
            (int(speech_id), t_id, float(dists['bertopic'][i].max()), "BERTopic"),
        )
        # Save LDA
        conn.execute(
            "INSERT INTO topic_distributions (speech_id, topic_id, probability, model_name) VALUES (?,?,?,?)",
            (int(speech_id), int(dists['lda'][i].argmax()), float(dists['lda'][i].max()), "LDA"),
        )
        # Save NMF
        conn.execute(
            "INSERT INTO topic_distributions (speech_id, topic_id, probability, model_name) VALUES (?,?,?,?)",
            (int(speech_id), int(dists['nmf'][i].argmax()), float(dists['nmf'][i].max()), "NMF"),
        )
        # Save Consensus
        conn.execute(
            "INSERT INTO topic_distributions (speech_id, topic_id, probability, model_name) VALUES (?,?,?,?)",
            (int(speech_id), int(consensus[i].argmax()), float(consensus[i].max()), "Consensus"),
        )
        
    conn.commit()
    conn.close()
    print(f"    Saved {len(df)*4} granular topic distribution entries to DB.")

    return modeler.bertopic_model, df, consensus.argmax(axis=1)

# ─── 3. Keyword → Sector mapping ─────────────────────────────────────────────
def classify_topic_to_sector(keywords):
    """Map topic keywords to the best matching market sector."""
    if not keywords:
        return "Broad Market"
    keyword_str = " ".join(keywords).lower()
    best_sector  = "Broad Market"
    best_score   = 0
    for sector, kws in SECTOR_TOPIC_MAP.items():
        score = sum(1 for kw in kws if kw in keyword_str)
        if score > best_score:
            best_score  = score
            best_sector = sector
    return best_sector

# ─── 4. Compute Speech-Market Impact ─────────────────────────────────────────
def compute_impact():
    print("\n[3/5] Computing speech-market impact (past vs present stock prices)...")
    conn = sqlite3.connect(DB_PATH)

    speeches = pd.read_sql_query(
        "SELECT id, date, source, title, speaker FROM speeches WHERE date IS NOT NULL AND date != '' ORDER BY date DESC",
        conn,
    )
    market = pd.read_sql_query(
        "SELECT date, ticker, sector, close, returns FROM market_data WHERE returns IS NOT NULL",
        conn,
    )
    conn.close()

    if speeches.empty or market.empty:
        print("    Not enough data — speeches or market data missing.")
        return pd.DataFrame()

    speeches["date"] = pd.to_datetime(speeches["date"], errors="coerce")
    speeches = speeches.dropna(subset=["date"])

    market["date"] = pd.to_datetime(market["date"], errors="coerce")
    market = market.dropna(subset=["date"])
    market_indexed = market.set_index("date").sort_index()

    records = []
    tickers = market["ticker"].unique()

    for _, sp in speeches.iterrows():
        event_date = sp["date"]
        for ticker in tickers:
            t_data = market_indexed[market_indexed["ticker"] == ticker]["close"].sort_index()
            sector = market[market["ticker"] == ticker]["sector"].iloc[0] if not market[market["ticker"] == ticker].empty else "Unknown"

            # Past price: most recent close ON or BEFORE the speech date
            past = t_data[t_data.index <= event_date]
            past_price = float(past.iloc[-1]) if not past.empty else None

            # Present prices: next 1, 5, 10 trading days
            future = t_data[t_data.index > event_date]
            p1  = float(future.iloc[0])  if len(future) >= 1  else None
            p5  = float(future.iloc[4])  if len(future) >= 5  else None
            p10 = float(future.iloc[9])  if len(future) >= 10 else None

            ret5  = ((p5  / past_price) - 1) if (p5  and past_price and past_price > 0) else None
            ret10 = ((p10 / past_price) - 1) if (p10 and past_price and past_price > 0) else None

            records.append({
                "speech_id":   sp["id"],
                "date":        event_date.strftime("%Y-%m-%d"),
                "source":      sp["source"],
                "title":       sp["title"],
                "speaker":     sp.get("speaker", "Unknown"),
                "ticker":      ticker,
                "sector":      sector,
                "past_price":  past_price,
                "price_t1":    p1,
                "price_t5":    p5,
                "price_t10":   p10,
                "return_5d":   ret5,
                "return_10d":  ret10,
            })

    impact_df = pd.DataFrame(records)
    impact_df.to_csv(f"{OUTPUT_DIR}/speech_market_impact_full.csv", index=False)
    print(f"    Computed {len(impact_df)} speech-ticker impact records.")
    return impact_df

# ─── 5. Generate Report ───────────────────────────────────────────────────────
def generate_report(topic_model, speeches_df, topics, impact_df):
    print("\n[4/5] Generating stock influence report...")

    conn = sqlite3.connect(DB_PATH)
    speeches_with_dates = pd.read_sql_query(
        "SELECT id, date, source, title, speaker FROM speeches WHERE date IS NOT NULL AND date != '' ORDER BY date DESC LIMIT 20",
        conn,
    )
    conn.close()

    if speeches_with_dates.empty:
        print("    No speeches with dates found.")
        return

    # Build topic-keyword map
    topic_keywords = {}
    if topic_model is not None:
        for tid in set(topics if topics is not None else []):
            if tid == -1:
                continue
            topic_res = topic_model.get_topic(tid)
            if topic_res and isinstance(topic_res, list):
                kws = [w for w, _ in topic_res[:8]]
                topic_keywords[tid] = kws

    report_lines = []
    report_lines.append("=" * 90)
    report_lines.append("  BAATSE BHARAT — SPEECH-MARKET INFLUENCE REPORT")
    report_lines.append("  BERTopic + SBERT Embeddings | Past vs Present Stock Prices")
    report_lines.append("=" * 90)

    processed_speeches = set()

    for _, sp in speeches_with_dates.iterrows():
        sp_id      = sp["id"]
        date_str   = str(sp["date"])
        source     = sp["source"]
        title      = sp["title"] if sp["title"] else "Untitled"
        speaker    = sp.get("speaker", "Unknown")

        # Get topic for this speech
        topic_id = None
        if speeches_df is not None and topics is not None:
            match = speeches_df[speeches_df["id"] == sp_id]
            if not match.empty:
                idx      = match.index[0]
                topic_id = topics[idx] if idx < len(topics) else None

        kws     = topic_keywords.get(topic_id, []) if topic_id is not None else []
        sector  = classify_topic_to_sector(kws)
        kw_str  = ", ".join(kws[:6]) if kws else "N/A"

        # Get relevant tickers for this sector
        relevant_tickers = SECTOR_TICKERS.get(sector, SECTOR_TICKERS["Broad Market"])

        report_lines.append(f"\n{'─'*90}")
        report_lines.append(f"  EVENT   : {date_str}  |  {source}  |  {speaker}")
        report_lines.append(f"  TITLE   : {title[:80]}")
        report_lines.append(f"  TOPIC   : {sector} sector  (BERTopic keywords: {kw_str})")
        report_lines.append(f"{'─'*90}")

        # Filter impact data
        sp_impact = impact_df[(impact_df["speech_id"] == sp_id) & (impact_df["ticker"].isin(relevant_tickers))]

        if sp_impact.empty:
            report_lines.append("  [No market data available for this event period]")
            continue

        report_lines.append(f"  {'Ticker':<16} {'Sector':<14} {'Price (T-0)':<14} {'Price (T+5)':<14} {'Price (T+10)':<14} {'5D Ret%':<10} {'10D Ret%':<10} Influence")
        report_lines.append(f"  {'─'*16} {'─'*14} {'─'*14} {'─'*14} {'─'*14} {'─'*10} {'─'*10} {'─'*10}")

        for _, row in sp_impact.iterrows():
            past   = row["past_price"]
            p5     = row["price_t5"]
            p10    = row["price_t10"]
            r5     = row["return_5d"]
            r10    = row["return_10d"]

            if past is None:
                continue

            r5_str  = f"{r5*100:+.2f}%"  if r5  is not None else "N/A"
            r10_str = f"{r10*100:+.2f}%" if r10 is not None else "N/A"
            p5_str  = f"{p5:.2f}"        if p5  is not None else "N/A"
            p10_str = f"{p10:.2f}"       if p10 is not None else "N/A"

            if r5 is not None:
                influence = "POSITIVE" if r5 > 0.01 else ("NEGATIVE" if r5 < -0.01 else "NEUTRAL")
            else:
                influence = "N/A"

            report_lines.append(
                f"  {row['ticker']:<16} {row['sector']:<14} {past:<14.2f} {p5_str:<14} {p10_str:<14} {r5_str:<10} {r10_str:<10} {influence}"
            )

    report_lines.append(f"\n{'='*90}")
    report_lines.append("  SUMMARY: Topics Discovered by BERTopic")
    report_lines.append(f"{'='*90}")

    if topic_model is not None:
        topic_info = topic_model.get_topic_info()
        for _, row in topic_info[topic_info["Topic"] != -1].head(15).iterrows():
            tid = row["Topic"]
            topic_res = topic_model.get_topic(tid)
            if topic_res and isinstance(topic_res, list):
                kws    = [w for w, _ in topic_res[:6]]
                sector = classify_topic_to_sector(kws)
                report_lines.append(f"  Topic {int(tid):>3}  ({row['Count']:>4} docs)  → {sector:<14}  [{', '.join(kws[:5])}]")
    else:
        report_lines.append("  [BERTopic model not available — using keyword-based mapping]")

    report_lines.append(f"\n{'='*90}")
    report_lines.append("  Report complete.")

    report_text = "\n".join(report_lines)
    # Safe print for Windows terminals with limited encoding
    safe_text = report_text.encode("ascii", errors="replace").decode("ascii")
    print(safe_text)

    report_path = f"{OUTPUT_DIR}/final_influence_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n[5/5] Report saved to: {report_path}")

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fix_speech_dates()
    topic_model, speeches_df, topics = train_hybrid_topics()
    impact_df = compute_impact()
    generate_report(topic_model, speeches_df, topics, impact_df)
