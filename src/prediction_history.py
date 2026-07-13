"""
prediction_history.py
======================
"Global Preview" backend: for every past speech with a known market outcome
(speech_market_impact.return_t1/return_t5), replay what
prediction_engine.get_company_prediction() would have produced using only
signals available as of that speech -- no live yfinance momentum, no
"latest" regime label, no historical-return average that includes future
events -- and compare it against what actually happened.

Reuses prediction_engine's own scoring primitives (_composite_score,
_forecast_return, _score_to_signal) rather than reimplementing them, so this
is a genuine replay of the live pipeline's math, not a second, drifting
implementation of it.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.append(_SRC_DIR)

from prediction_engine import (  # noqa: E402
    COMPANY_PROFILES,
    COMPANY_UNIVERSE,
    DB_PATH,
    SECTOR_TICKER_MAP,
    _composite_score,
    _forecast_return,
    _groq_blend_weight,
    _score_to_signal,
    get_company_sector_by_ticker,
)

_TICKER_TO_COMPANY: Dict[str, str] = {t: n for n, t in COMPANY_UNIVERSE.items()}


def _ticker_to_company(ticker: str) -> str:
    return _TICKER_TO_COMPANY.get(ticker, ticker)


_REGIME_CSV_CACHE: Dict[str, Optional[pd.DataFrame]] = {}


def _load_regime_csv(ticker: str) -> Optional[pd.DataFrame]:
    if ticker in _REGIME_CSV_CACHE:
        return _REGIME_CSV_CACHE[ticker]
    path = f'./data/processed/regime_labels_{ticker}.csv'
    df = None
    if os.path.exists(path):
        try:
            raw = pd.read_csv(path)
            if 'regime_label' in raw.columns:
                raw['date'] = pd.to_datetime(raw['date'], errors='coerce')
                df = raw.dropna(subset=['date']).sort_values('date')
        except Exception:
            df = None
    _REGIME_CSV_CACHE[ticker] = df
    return df


def _regime_asof(ticker: str, as_of_date: pd.Timestamp) -> str:
    """Regime label as of `as_of_date` (strictly historical -- the last row
    on or before that date), falling back to the sector index ticker, then
    'Neutral'. Deliberately NOT the same as
    prediction_engine._resolve_company_regime(), which reads the *latest*
    row regardless of date and would leak future regime info into a
    backtest of a past speech.
    """
    df = _load_regime_csv(ticker)
    if df is None or df.empty:
        sector = get_company_sector_by_ticker(ticker)
        sector_ticker = SECTOR_TICKER_MAP.get(sector, '^NSEI')
        if sector_ticker != ticker:
            df = _load_regime_csv(sector_ticker)
    if df is None or df.empty:
        return "Neutral"
    prior = df[df['date'] <= as_of_date]
    if prior.empty:
        return "Neutral"
    latest = str(prior.iloc[-1]['regime_label']).lower()
    if 'volatile' in latest or 'bear' in latest:
        return 'Bear'
    if 'stable' in latest or 'bull' in latest:
        return 'Bull'
    return 'Neutral'


def _momentum_asof(market_by_ticker: Dict[str, pd.DataFrame], ticker: str,
                    as_of_date: pd.Timestamp) -> Tuple[float, float]:
    """5-day/10-day cumulative return strictly BEFORE `as_of_date`, from the
    market_data table (not live yfinance) so this never sees data that
    wasn't available yet at the time of the speech."""
    series = market_by_ticker.get(ticker)
    if series is None:
        return 0.0, 0.0
    prior = series[series['date'] < as_of_date]['returns']
    if prior.empty:
        return 0.0, 0.0
    m5 = float(np.prod(1 + prior.tail(5).values) - 1)
    m10 = float(np.prod(1 + prior.tail(10).values) - 1)
    return m5, m10


def compute_prediction_vs_actual(source: Optional[str] = None,
                                  company: Optional[str] = None,
                                  db_path: str = DB_PATH) -> pd.DataFrame:
    """One row per (speech, tracked company) that has a realized outcome.

    Columns: date, source, company, ticker, predicted_signal,
    predicted_return_1d, predicted_return_5d, actual_return_1d,
    actual_return_5d, hit (bool or None if actual was exactly zero).
    """
    if not os.path.exists(db_path):
        return pd.DataFrame()

    tickers = [COMPANY_UNIVERSE[company]] if company else list(COMPANY_UNIVERSE.values())
    conn = sqlite3.connect(db_path)
    try:
        placeholders = ",".join("?" for _ in tickers)
        query = f"""
            SELECT i.speech_id, i.ticker, i.event_date, i.return_t1, i.return_t5,
                   s.source, s.date as speech_date
            FROM speech_market_impact i
            JOIN speeches s ON s.id = i.speech_id
            WHERE i.ticker IN ({placeholders})
              AND i.return_t1 IS NOT NULL AND i.return_t5 IS NOT NULL
        """
        params = list(tickers)
        if source:
            query += " AND s.source = ?"
            params.append(source)
        query += " ORDER BY i.event_date"
        events = pd.read_sql_query(query, conn, params=params)
        if events.empty:
            return pd.DataFrame()

        sent_df = pd.read_sql_query(
            "SELECT speech_id, AVG(compound) as compound FROM sentiment_scores GROUP BY speech_id", conn
        )
        topic_df = pd.read_sql_query(
            "SELECT speech_id, MAX(probability) as max_topic_prob FROM topic_distributions "
            "WHERE model_name = 'Combined' GROUP BY speech_id", conn
        )
        llm_df = pd.read_sql_query(
            "SELECT speech_id, ticker, strength_score, sentiment_score as llm_sentiment_score "
            "FROM llm_company_signals", conn
        )
        market_df = pd.read_sql_query(
            "SELECT date, ticker, returns FROM market_data WHERE returns IS NOT NULL", conn
        )
    finally:
        conn.close()

    market_df['date'] = pd.to_datetime(market_df['date'], errors='coerce')
    market_df = market_df.dropna(subset=['date'])
    market_by_ticker = {t: g.sort_values('date') for t, g in market_df.groupby('ticker')}

    events['event_date'] = pd.to_datetime(events['event_date'], errors='coerce')
    events = events.dropna(subset=['event_date']).sort_values('event_date')
    events = events.merge(sent_df, on='speech_id', how='left')
    events = events.merge(topic_df, on='speech_id', how='left')
    events = events.merge(llm_df, on=['speech_id', 'ticker'], how='left')

    blend_w = _groq_blend_weight()
    ticker_hist_returns: Dict[str, list] = {}

    rows = []
    for _, row in events.iterrows():
        ticker = row['ticker']
        as_of = row['event_date']

        sentiment = float(row['compound']) if pd.notna(row['compound']) else 0.0
        topic_strength = float(row['max_topic_prob']) if pd.notna(row['max_topic_prob']) else 0.5

        if pd.notna(row.get('strength_score')):
            topic_strength = (1 - blend_w) * topic_strength + blend_w * float(row['strength_score'])
            sentiment = (1 - blend_w) * sentiment + blend_w * float(row['llm_sentiment_score'])

        beta, rhet_sens, base_drift = COMPANY_PROFILES.get(_ticker_to_company(ticker), (1.0, 0.75, 0.0))
        regime = _regime_asof(ticker, as_of)
        m5, m10 = _momentum_asof(market_by_ticker, ticker, as_of)

        past = ticker_hist_returns.get(ticker, [])
        hist_ret = float(np.mean(past)) if past else 0.0

        score = _composite_score(
            sentiment, topic_strength, regime, m5, m10, hist_ret,
            beta=beta, rhetoric_sensitivity=rhet_sens, base_drift=base_drift,
        )
        signal, emoji = _score_to_signal(score)
        f1 = _forecast_return(score, 1, hist_ret, None)
        f5 = _forecast_return(score, 5, hist_ret, None)

        actual_1d = float(row['return_t1']) * 100.0
        actual_5d = float(row['return_t5']) * 100.0
        pred_5d = f5['return_pct']

        rows.append({
            'date': row['speech_date'],
            'source': row['source'],
            'company': _ticker_to_company(ticker),
            'ticker': ticker,
            'predicted_signal': signal,
            'predicted_return_1d': f1['return_pct'],
            'predicted_return_5d': pred_5d,
            'actual_return_1d': actual_1d,
            'actual_return_5d': actual_5d,
            'hit': (np.sign(pred_5d) == np.sign(actual_5d)) if actual_5d != 0 else None,
        })

        # Only extend history AFTER using it, so this event's own outcome
        # never leaks into its own (or earlier events') prediction.
        ticker_hist_returns.setdefault(ticker, []).append(float(row['return_t5']))

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> Dict:
    """Overall + per-company + per-source hit-rate and mean absolute error."""
    if df is None or df.empty:
        return {}

    hit_df = df.dropna(subset=['hit'])
    overall_hit_rate = float(hit_df['hit'].mean()) if not hit_df.empty else None
    mae_1d = float((df['predicted_return_1d'] - df['actual_return_1d']).abs().mean())
    mae_5d = float((df['predicted_return_5d'] - df['actual_return_5d']).abs().mean())

    def _group_stats(g: pd.DataFrame) -> pd.Series:
        g_hits = g['hit'].dropna()
        return pd.Series({
            'n': len(g),
            'hit_rate': float(g_hits.mean()) if not g_hits.empty else None,
            'mae_5d': float((g['predicted_return_5d'] - g['actual_return_5d']).abs().mean()),
        })

    per_company = df.groupby('company', group_keys=True).apply(_group_stats, include_groups=False).reset_index()
    per_source = df.groupby('source', group_keys=True).apply(_group_stats, include_groups=False).reset_index()

    return {
        'overall_hit_rate': overall_hit_rate,
        'mean_abs_error_1d': mae_1d,
        'mean_abs_error_5d': mae_5d,
        'n_events': int(len(df)),
        'per_company': per_company,
        'per_source': per_source,
    }


if __name__ == "__main__":
    result = compute_prediction_vs_actual()
    print(f"{len(result)} prediction-vs-actual rows")
    if not result.empty:
        print(summarize(result))
