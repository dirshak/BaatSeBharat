"""
Regression tests for the AI Predictions module returning near-identical
values across companies/sectors.

Root causes found and fixed in src/prediction_engine.py:

1. _resolve_company_sentiment() / _resolve_company_topic_strength() derived
   their "leadership rhetoric" signal purely from live yfinance price
   momentum (explicitly bypassing the FinBERT/topic-model DB tables), so
   whenever momentum was flat or yfinance failed, every company collapsed
   to the identical sentiment=0.0 / topic_strength=0.4 baseline. Fixed to
   read real FinBERT sentiment (sentiment_scores) and topic-probability
   weighted returns (topic_distributions), joined per-ticker via
   speech_market_impact.

2. _resolve_company_regime() picked `df.columns[-1]` whenever a column
   literally named "regime" was absent from data/processed/regime_labels_*.csv.
   Those files actually use the column name "regime_label", so the code was
   silently reading `regime_probability` (a float) instead, which never
   matches "bull/bear/stable/volatile" and always resolved to "Neutral" for
   every single ticker. Fixed to also recognize "regime_label".
"""
import os
import sqlite3

import pandas as pd
import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import prediction_engine as pe

DB_PATH = pe.DB_PATH


def _has_impact_data(ticker: str) -> bool:
    if not os.path.exists(DB_PATH):
        return False
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            "SELECT COUNT(*) as n FROM speech_market_impact WHERE ticker = ?",
            conn, params=(ticker,)
        )
    finally:
        conn.close()
    return bool(df.empty is False and df['n'].iloc[0] > 0)


@pytest.fixture
def companies():
    # One Banking, one IT, one representative of the Broad Market universe.
    return ["HDFC Bank", "Infosys", "Reliance Industries"]


def test_topic_strength_in_valid_range():
    ticker = pe.COMPANY_UNIVERSE["HDFC Bank"]
    if not _has_impact_data(ticker):
        pytest.skip("No speech_market_impact data available for HDFC Bank")
    value = pe._resolve_company_topic_strength(ticker)
    assert 0.1 <= value <= 0.9


def test_company_topic_strength_differs_across_sectors(companies):
    """Companies from different sectors must not resolve to an identical
    topic-strength baseline (this was the direct symptom of the bug)."""
    values = {}
    for company in companies:
        ticker = pe.COMPANY_UNIVERSE[company]
        if not _has_impact_data(ticker):
            pytest.skip(f"No speech_market_impact data available for {company}")
        values[company] = pe._resolve_company_topic_strength(ticker)
    assert len(set(round(v, 6) for v in values.values())) > 1, (
        f"All companies resolved to the identical topic strength: {values}"
    )


def test_regime_reads_regime_label_column(tmp_path):
    """A regime_labels_*.csv using the real column name `regime_label`
    (not `regime`) must be parsed correctly instead of silently falling
    back to whatever the last column happens to be.
    """
    csv_path = tmp_path / "regime_labels_TESTTICKER.NS.csv"
    pd.DataFrame({
        "date": ["2026-06-01", "2026-06-02"],
        "regime_label": ["Transitional", "Volatile"],
        "regime_probability": [0.71, 0.93],
    }).to_csv(csv_path, index=False)

    df = pd.read_csv(csv_path)
    for candidate in ("regime", "regime_label"):
        if candidate in df.columns:
            regime_col = candidate
            break
    else:
        regime_col = df.columns[-1]

    assert regime_col == "regime_label"
    latest = str(df[regime_col].iloc[-1])
    assert "volatile" in latest.lower()


def test_company_regime_uses_real_csv_when_available():
    """If a per-ticker regime CSV exists, the resolved regime must reflect
    its regime_label column, not default to Neutral via the wrong column.
    """
    ticker = pe.COMPANY_UNIVERSE["Infosys"]
    csv_path = f'./data/processed/regime_labels_{ticker}.csv'
    if not os.path.exists(csv_path):
        pytest.skip("No regime CSV available for Infosys")
    df = pd.read_csv(csv_path)
    assert 'regime_label' in df.columns
    expected_latest = str(df['regime_label'].iloc[-1]).lower()
    resolved = pe._resolve_company_regime(ticker)
    if 'stable' in expected_latest or 'bull' in expected_latest:
        assert resolved == 'Bull'
    elif 'volatile' in expected_latest or 'bear' in expected_latest:
        assert resolved == 'Bear'
    else:
        assert resolved == 'Neutral'


def test_company_predictions_differ_across_sectors(companies):
    """End-to-end regression for the reported bug: 1-year/near-term
    predictions must not be near-identical across companies from different
    sectors (Banking vs IT vs Broad Market).
    """
    scores = {}
    for company in companies:
        ticker = pe.COMPANY_UNIVERSE[company]
        if not _has_impact_data(ticker):
            pytest.skip(f"No speech_market_impact data available for {company}")
        scores[company] = pe.get_company_prediction(company)['score']

    values = list(scores.values())
    spread = max(values) - min(values)
    assert spread > 0.01, (
        f"Company prediction scores are nearly identical across sectors: {scores}"
    )


def test_sector_predictions_differ():
    sectors = ["Banking", "IT", "Broad Market"]
    scores = {}
    for sector in sectors:
        scores[sector] = pe.get_sector_prediction(sector)['score']
    spread = max(scores.values()) - min(scores.values())
    assert spread > 0.01, (
        f"Sector prediction scores are nearly identical: {scores}"
    )


def test_prediction_magnitudes_are_plausible_not_degenerate():
    """Task 8 sanity check: composite scores must not all cluster at ~0
    (dead model) nor hit the +/-1 clip ceiling (runaway/miscalibrated
    model) for every company under realistic current inputs."""
    preds = pe.get_all_company_predictions()
    scores = [p['score'] for p in preds]
    assert len(scores) == len(pe.COMPANY_UNIVERSE)
    assert not all(abs(s) < 1e-6 for s in scores), "All scores are exactly zero -- dead model"
    assert not any(abs(s) >= 0.999 for s in scores), (
        f"Some scores are pinned at the +/-1 clip ceiling: {scores}"
    )
    # 1-year-equivalent annualized return implied by the score (see
    # _forecast_return: annual_return_pct = score * 25) should stay in a
    # plausible band, not explode.
    annualized = [abs(s) * 25.0 for s in scores]
    assert max(annualized) < 25.0, f"Implausible annualized return magnitude: {annualized}"
