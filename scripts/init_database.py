import sqlite3
import os

def create_database(db_path='./data/market_rhetoric.db'):
    """Initialize database with all required tables"""
    
    # Create directory if doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Table 1: Speeches
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS speeches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            source TEXT NOT NULL,
            country TEXT,
            speaker TEXT,
            title TEXT,
            full_text TEXT,
            processed_text TEXT,
            url TEXT,
            language TEXT,
            episode_number INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(date, source, speaker, title)
        )
    ''')
    
    # Table 2: Market Data
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS market_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE NOT NULL,
            ticker TEXT NOT NULL,
            sector TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            returns REAL,
            volatility REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(date, ticker)
        )
    ''')
    
    # Table 3: VIX Data
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vix_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE NOT NULL UNIQUE,
            vix_open REAL,
            vix_high REAL,
            vix_low REAL,
            vix_close REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Table 4: Macro Controls
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS macro_controls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE NOT NULL,
            indicator TEXT NOT NULL,
            value REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(date, indicator)
        )
    ''')
    
    # Table 5: Topic Distributions
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS topic_distributions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            speech_id INTEGER NOT NULL,
            segment_type TEXT DEFAULT 'episode', -- 'sentence', 'paragraph', 'episode'
            segment_index INTEGER DEFAULT 0,
            topic_id INTEGER NOT NULL,
            probability REAL NOT NULL,
            model_name TEXT DEFAULT 'combined',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (speech_id) REFERENCES speeches(id),
            UNIQUE(speech_id, segment_type, segment_index, topic_id, model_name)
        )
    ''')
    
    # Table 6: Sentiment Scores (FinBERT style)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sentiment_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            speech_id INTEGER NOT NULL,
            segment_type TEXT DEFAULT 'episode',
            segment_index INTEGER DEFAULT 0,
            optimism_intensity REAL,
            risk_awareness REAL,
            positive REAL,
            negative REAL,
            neutral REAL,
            compound REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (speech_id) REFERENCES speeches(id)
        )
    ''')
    
    # Table 7: Regime Classifications
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS regime_classifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE NOT NULL UNIQUE,
            sector TEXT NOT NULL,
            regime TEXT NOT NULL,
            confidence REAL,
            deviation_magnitude REAL,
            volume_zscore REAL,
            volatility_ratio REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Table 8: Early Warnings
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS early_warnings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE NOT NULL,
            sector TEXT NOT NULL,
            warning_level TEXT NOT NULL,
            warning_score REAL,
            explanation TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Table 9: Speech-Market Impact
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS speech_market_impact (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            speech_id INTEGER,
            ticker TEXT,
            event_date TEXT,
            return_t1 REAL,
            return_t5 REAL,
            return_t10 REAL,
            abnormal_return REAL,
            pwm_shock_score REAL,
            FOREIGN KEY (speech_id) REFERENCES speeches(id)
        )
    ''')
    
    # Create indexes for performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_speeches_date ON speeches(date)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_market_date ON market_data(date)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_market_ticker ON market_data(ticker)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_vix_date ON vix_data(date)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_regime_date ON regime_classifications(date)')
    
    conn.commit()
    conn.close()
    
    print(f"Database created successfully at: {db_path}")
    return db_path

if __name__ == "__main__":
    create_database()
