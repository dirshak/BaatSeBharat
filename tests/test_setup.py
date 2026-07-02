import pytest
import os
import sqlite3

def test_environment_setup():
    """Test that basic environment is set up correctly"""
    assert os.path.exists('./data')
    assert os.path.exists('./src')
    assert os.path.exists('./config')

def test_database_exists():
    """Test that database file exists"""
    assert os.path.exists('./data/market_rhetoric.db')

def test_database_tables():
    """Test that all required tables exist"""
    conn = sqlite3.connect('./data/market_rhetoric.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    
    required_tables = [
        'speeches',
        'market_data',
        'vix_data',
        'macro_controls',
        'topic_distributions',
        'sentiment_scores',
        'regime_classifications',
        'early_warnings'
    ]
    
    for table in required_tables:
        assert table in tables, f"Missing table: {table}"
    
    conn.close()

def test_imports():
    """Test that all required packages can be imported"""
    import pandas
    import numpy
    import torch
    import transformers
    import spacy
    import sklearn
    import statsmodels
    assert True
