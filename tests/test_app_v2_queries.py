"""
Regression tests for the two App_v2.py "Stage 3 / Stage 5" bugs where topic-level
output collapsed to identical values regardless of topic or company:

- Stage 3 "Topic-Market Correlation Analysis" (market bigraph): joining the
  long-format topic_distributions table (one row per topic per speech) to
  speech_market_impact without weighting by td.probability caused every
  topic_id to be averaged over the exact same set of (speech, return) pairs,
  so avg_abnormal / avg_ret_t5 were identical across all topics for a ticker.

- Stage 5 "Company Analytics" topic heatmap (topic strength): the query never
  filtered by company/ticker at all, so every company selection produced the
  identical, corpus-wide heatmap.

These tests run the same SQL App_v2.py uses against the real project database
and assert that per-topic / per-company outputs actually differ.
"""
import sqlite3
import pandas as pd
import pytest

DB_PATH = './data/market_rhetoric.db'

TOPIC_MARKET_QUERY = '''
    SELECT
        td.topic_id,
        SUM(td.probability * i.return_t5) / NULLIF(SUM(td.probability), 0) as avg_ret_t5,
        SUM(td.probability * i.abnormal_return) / NULLIF(SUM(td.probability), 0) as avg_abnormal,
        COUNT(DISTINCT i.id) as speech_count
    FROM topic_distributions td
    JOIN speech_market_impact i ON td.speech_id = i.speech_id
    WHERE td.model_name = 'Combined' AND i.ticker = ?
    GROUP BY td.topic_id
    ORDER BY avg_abnormal DESC
'''

COMPANY_TICKER_MAP = {
    "HDFC Bank": "HDFCBANK.NS",
    "Reliance Industries": "RELIANCE.NS",
    "Infosys": "INFY.NS",
    "TCS": "TCS.NS",
    "ICICI Bank": "ICICIBANK.NS",
}

COMPANY_TOPIC_QUERY = '''
    SELECT s.date, td.topic_id, td.probability, i.return_t5
    FROM topic_distributions td
    JOIN speeches s ON td.speech_id = s.id
    JOIN speech_market_impact i ON i.speech_id = s.id
    WHERE td.model_name = 'Combined' AND i.ticker = ?
'''


@pytest.fixture
def db_connection():
    conn = sqlite3.connect(DB_PATH)
    yield conn
    conn.close()


def test_topic_market_correlation_differs_by_topic(db_connection):
    """Stage 3: avg_abnormal must not be identical across all topics for a ticker."""
    df = pd.read_sql_query(TOPIC_MARKET_QUERY, db_connection, params=('HDFCBANK.NS',))
    if df.empty:
        pytest.skip("No topic/impact data available for HDFCBANK.NS")
    assert df['avg_abnormal'].nunique() > 1, (
        "All topics show the identical avg_abnormal return — the topic-probability "
        "weighting that differentiates topics is missing (join fan-out bug)."
    )


def test_topic_market_correlation_differs_by_ticker(db_connection):
    """Sanity: two different tickers should not produce identical topic profiles."""
    df_a = pd.read_sql_query(TOPIC_MARKET_QUERY, db_connection, params=('HDFCBANK.NS',))
    df_b = pd.read_sql_query(TOPIC_MARKET_QUERY, db_connection, params=('INFY.NS',))
    if df_a.empty or df_b.empty:
        pytest.skip("No topic/impact data available for one of the test tickers")
    merged = df_a.merge(df_b, on='topic_id', suffixes=('_a', '_b'))
    assert not (merged['avg_abnormal_a'] == merged['avg_abnormal_b']).all()


def test_company_topic_heatmap_differs_by_company(db_connection):
    """Stage 5: the return-weighted topic heatmap must differ between companies.

    Raw topic probability has no company dimension (every ticker shares the
    same speech/topic rows), so the heatmap must be weighted by each
    company's own return_t5 to actually differentiate companies.
    """
    df_hdfc = pd.read_sql_query(COMPANY_TOPIC_QUERY, db_connection, params=(COMPANY_TICKER_MAP["HDFC Bank"],))
    df_infy = pd.read_sql_query(COMPANY_TOPIC_QUERY, db_connection, params=(COMPANY_TICKER_MAP["Infosys"],))
    if df_hdfc.empty or df_infy.empty:
        pytest.skip("No company-scoped topic data available")

    df_hdfc['weighted_strength'] = df_hdfc['probability'] * df_hdfc['return_t5']
    df_infy['weighted_strength'] = df_infy['probability'] * df_infy['return_t5']
    piv_hdfc = df_hdfc.groupby(['date', 'topic_id'])['weighted_strength'].sum().unstack().fillna(0)
    piv_infy = df_infy.groupby(['date', 'topic_id'])['weighted_strength'].sum().unstack().fillna(0)

    assert not piv_hdfc.equals(piv_infy), (
        "Topic Impact Heatmap is identical for HDFC Bank and Infosys — "
        "company selection has no effect on the chart."
    )


def test_company_topic_heatmap_nonempty_for_known_company(db_connection):
    df = pd.read_sql_query(COMPANY_TOPIC_QUERY, db_connection, params=(COMPANY_TICKER_MAP["HDFC Bank"],))
    assert not df.empty, "Expected topic-impact rows for HDFC Bank given seeded speech_market_impact data"
