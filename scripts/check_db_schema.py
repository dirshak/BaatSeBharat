import sqlite3
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB_PATH = './data/market_rhetoric.db'
conn = sqlite3.connect(DB_PATH)

tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print("\n=== DATABASE TABLES ===")
for t in tables:
    tname = t[0]
    count = conn.execute(f"SELECT COUNT(*) FROM {tname}").fetchone()[0]
    cols = [c[1] for c in conn.execute(f"PRAGMA table_info({tname})").fetchall()]
    print(f"\n  [{tname}] — {count} rows")
    print(f"    Columns: {cols}")

print("\n=== SPEECHES SAMPLE ===")
df = pd.read_sql_query("SELECT id, source, date, LENGTH(full_text) as text_len, processed_text IS NOT NULL as has_processed FROM speeches LIMIT 5", conn)
print(df.to_string())

print("\n=== SPEECHES BY SOURCE ===")
df2 = pd.read_sql_query("SELECT source, COUNT(*) as count FROM speeches GROUP BY source", conn)
print(df2.to_string())

print("\n=== MARKET DATA SAMPLE ===")
try:
    df3 = pd.read_sql_query("SELECT ticker, date, close, returns FROM market_data ORDER BY date DESC LIMIT 10", conn)
    print(df3.to_string())
except Exception as e:
    print(f"  No market data: {e}")

print("\n=== SPEECH-MARKET IMPACT ===")
try:
    df4 = pd.read_sql_query("SELECT COUNT(*) as count, COUNT(DISTINCT ticker) as tickers FROM speech_market_impact", conn)
    print(df4.to_string())
except Exception as e:
    print(f"  No impact data: {e}")

print("\n=== TOPIC DISTRIBUTIONS ===")
try:
    df5 = pd.read_sql_query("SELECT model_name, COUNT(*) as count FROM topic_distributions GROUP BY model_name", conn)
    print(df5.to_string())
except Exception as e:
    print(f"  No topic distributions: {e}")

conn.close()
