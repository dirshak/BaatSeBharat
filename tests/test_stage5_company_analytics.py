"""
Regression tests for Stage 5 ("Company Analytics"): topic labels must be
real (not raw IDs), and the topic-impact heatmap must actually
differentiate between companies and topics instead of being flat/identical
(see also tests/test_app_v2_queries.py for the original company-heatmap
collapse bug this builds on).
"""
import json
import os

import pandas as pd
import pytest
import sqlite3

DB_PATH = './data/market_rhetoric.db'
LABELS_PATH = './data/processed/topic_labels_combined.json'


@pytest.fixture
def db_connection():
    conn = sqlite3.connect(DB_PATH)
    yield conn
    conn.close()


def test_topic_labels_file_has_real_labels_not_raw_ids():
    if not os.path.exists(LABELS_PATH):
        pytest.skip("topic_labels_combined.json not generated in this environment")
    with open(LABELS_PATH, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    assert len(labels) > 0
    for topic_id, info in labels.items():
        assert 'label' in info and 'keywords' in info
        assert info['label'], f"Topic {topic_id} has an empty label"
        assert not info['label'].lower().startswith('topic '), (
            f"Topic {topic_id} label is a raw placeholder: {info['label']!r}"
        )


def test_company_heatmap_values_vary_across_topics_and_companies(db_connection):
    query = """
        SELECT s.date, td.topic_id, td.probability, i.return_t5
        FROM topic_distributions td
        JOIN speeches s ON td.speech_id = s.id
        JOIN speech_market_impact i ON i.speech_id = s.id
        WHERE td.model_name = 'Combined' AND i.ticker = ?
    """
    frames = {}
    for ticker in ('HDFCBANK.NS', 'INFY.NS'):
        df = pd.read_sql_query(query, db_connection, params=(ticker,))
        if df.empty:
            pytest.skip(f"No topic-impact data for {ticker}")
        df['weighted'] = df['probability'] * df['return_t5']
        df['month'] = pd.to_datetime(df['date']).dt.to_period('M').dt.to_timestamp()
        frames[ticker] = df.groupby(['month', 'topic_id'])['weighted'].mean().unstack().fillna(0)

    hdfc, infy = frames['HDFCBANK.NS'], frames['INFY.NS']

    # Not flat: each company's heatmap must have real spread, not a
    # constant value across all topics/months.
    assert hdfc.values.std() > 0
    assert infy.values.std() > 0

    # Differentiated across companies: same topic/month grid, different values.
    common_idx = hdfc.index.intersection(infy.index)
    assert len(common_idx) > 0
    assert not hdfc.loc[common_idx].equals(infy.loc[common_idx])
