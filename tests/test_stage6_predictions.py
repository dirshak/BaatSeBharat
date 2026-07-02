"""
Regression tests for Stage 6 ("AI Predictions").

Bug found while investigating Task 6 ("Stage 6 should be better"): the
time-decay/importance weighting added to prediction_engine.py
(_load_ticker_speech_signals) early-returned an empty DataFrame WITHOUT the
'compound'/'weight' columns whenever a ticker had zero
topic_distributions/speech_market_impact rows. 10 of the 15 companies in
COMPANY_UNIVERSE (Wipro, Bajaj Finance, SBI, Bharti Airtel, HCL Tech, Asian
Paints, Axis Bank, Kotak Mahindra, Titan Company) have no market_data
downloaded at all, so this path crashed the entire "All Companies" table
with KeyError: ['compound'] on every single page load -- a full-page
failure, not just missing data for those companies.

Fixed by having the empty-result path return a frame with the expected
columns, and by having both resolver functions bail out to their fallback
value on an empty frame before touching any column.
"""
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import prediction_engine as pe


def test_all_companies_predictions_do_not_crash():
    """get_all_company_predictions() must return a result for every company
    in COMPANY_UNIVERSE, including ones with no market_data downloaded."""
    results = pe.get_all_company_predictions()
    companies_returned = {r['company'] for r in results}
    assert companies_returned == set(pe.COMPANY_UNIVERSE.keys())


def test_resolvers_handle_ticker_with_no_market_data():
    """A ticker with zero topic_distributions/speech_market_impact rows
    must fall back gracefully, not raise KeyError: ['compound']."""
    missing_ticker = "NOT_A_REAL_TICKER.NS"
    sentiment = pe._resolve_company_sentiment(missing_ticker, fallback_sentiment=0.0)
    topic_strength = pe._resolve_company_topic_strength(missing_ticker, fallback_topic=0.5)
    assert sentiment == 0.0
    assert topic_strength == 0.5


def test_load_ticker_speech_signals_empty_result_has_expected_columns():
    df = pe._load_ticker_speech_signals("NOT_A_REAL_TICKER.NS")
    assert 'compound' in df.columns
    assert 'weight' in df.columns
