import sqlite3

DB_PATH = "./data/market_rhetoric.db"

def fix_schema():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("Recreating topic_distributions table with correct constraints...")
    
    # 1. Backup old data (optional, but since we're retraining we don't strictly need it)
    cursor.execute("DROP TABLE IF EXISTS topic_distributions")
    
    # 2. Create table with model_name in the PRIMARY KEY
    cursor.execute("""
        CREATE TABLE topic_distributions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            speech_id INTEGER,
            topic_id INTEGER,
            probability REAL,
            model_name TEXT,
            UNIQUE(speech_id, topic_id, model_name),
            FOREIGN KEY (speech_id) REFERENCES speeches (id)
        )
    """)
    
    conn.commit()
    conn.close()
    print("Database schema fixed.")

if __name__ == "__main__":
    fix_schema()
