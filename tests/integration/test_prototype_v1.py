import pytest
import os
import sqlite3
import pandas as pd
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def test_prototype_data_flow():
    """Verify that data flows through the pipeline to the DB."""
    db_path = './data/market_rhetoric.db'
    assert os.path.exists(db_path)
    
    conn = sqlite3.connect(db_path)
    
    # Check Speeches
    df_speeches = pd.read_sql_query("SELECT * FROM speeches", conn)
    assert len(df_speeches) > 0, "No speeches found in DB"
    assert 'processed_text' in df_speeches.columns
    
    # Check Market Data
    df_market = pd.read_sql_query("SELECT * FROM market_data", conn)
    assert len(df_market) > 0, "No market data found in DB"
    
    conn.close()

def test_directory_structure():
    """Verify all required directories for the plan exist."""
    required_dirs = [
        'data/raw', 'data/processed', 'src/data', 'src/models',
        'models/trained', 'logs', 'config'
    ]
    for d in required_dirs:
        assert os.path.isdir(d), f"Directory {d} missing"

def test_config_loading():
    """Verify config.yaml is valid."""
    import yaml
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    assert config['project']['name'] == "Leadership Rhetoric Market Regime Prediction"
