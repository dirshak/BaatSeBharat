import sqlite3
import pandas as pd
import numpy as np
import json
import os

DB_PATH = './data/market_rhetoric.db'

def get_report():
    if not os.path.exists(DB_PATH):
        print("Database not found. Please run scripts/run_prototype.py first.")
        return

    conn = sqlite3.connect(DB_PATH)
    
    # 1. Get latest speeches and their top topics
    query = """
    SELECT s.id, s.date, s.source, s.title, td.topic_id, td.probability, s.full_text
    FROM speeches s
    JOIN topic_distributions td ON s.id = td.speech_id
    WHERE td.probability > 0.3
    ORDER BY s.date DESC
    LIMIT 10
    """
    speeches_topics = pd.read_sql_query(query, conn)
    
    if speeches_topics.empty:
        print("No topic distributions found. Training might have failed or not run yet.")
        conn.close()
        return

    # 2. Get market impact for these speeches
    impact_query = """
    SELECT i.speech_id, i.ticker, i.event_date, i.return_t5, i.abnormal_return, i.pwm_shock_score
    FROM speech_market_impact i
    """
    impact_df = pd.read_sql_query(impact_query, conn)
    
    # 3. Load topic labels for domain mapping
    labels_path = './data/processed/topic_labels_combined.json'
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            topic_labels = json.load(f)
    else:
        topic_labels = {}

    print("\n" + "="*80)
    print("      STOCK INFLUENCE REPORT: SPEECHES VS MARKET PERFORMANCE")
    print("="*80)

    for _, speech in speeches_topics.iterrows():
        speech_id = speech['id']
        date = speech['date']
        source = speech['source']
        title = speech['title'] if speech['title'] else "Speech"
        topic_id = speech['topic_id']
        
        # Get topic domain
        t_label = topic_labels.get(f"Topic_{topic_id}", {})
        domain = t_label.get('zero_shot_domain', 'General')
        keywords = ", ".join(t_label.get('keywords', [])[:5])

        print(f"\n[EVENT] {date} | {source} | {title}")
        print(f"  Topic: {domain} (Keywords: {keywords})")
        
        # Get impact for this speech
        speech_impact = impact_df[impact_df['speech_id'] == speech_id]
        
        if speech_impact.empty:
            print("  No market impact data recorded for this event.")
            continue

        print(f"  {'Ticker':<15} | {'Prev Close':<12} | {'Post Close (T+5)':<15} | {'5D Return':<10} | {'Abnormal'}")
        print(f"  {'-'*15}-|-{'-'*12}-|-{'-'*15}-|-{'-'*10}-|-{'-'*8}")
        
        for _, impact in speech_impact.iterrows():
            ticker = impact['ticker']
            ret_5 = impact['return_t5']
            abnormal = impact['abnormal_return']
            
            # Fetch prices from market_data
            # Past price (on event_date)
            past_price_query = f"SELECT close FROM market_data WHERE ticker='{ticker}' AND date <= '{date}' ORDER BY date DESC LIMIT 1"
            past_price = conn.execute(past_price_query).fetchone()
            past_price = past_price[0] if past_price else 0.0
            
            # Present price (approx 5 days after)
            present_price_query = f"SELECT close FROM market_data WHERE ticker='{ticker}' AND date > '{date}' ORDER BY date ASC LIMIT 5"
            present_prices = conn.execute(present_price_query).fetchall()
            present_price = present_prices[-1][0] if present_prices else past_price
            
            ret_str = f"{ret_5*100:+.2f}%" if ret_5 is not None else "N/A"
            ab_str = f"{abnormal*100:+.2f}%" if abnormal is not None else "N/A"
            
            print(f"  {ticker:<15} | {past_price:>12.2f} | {present_price:>15.2f} | {ret_str:>10} | {ab_str}")

    conn.close()
    print("\n" + "="*80)
    print("Report generated successfully.")

if __name__ == "__main__":
    get_report()
