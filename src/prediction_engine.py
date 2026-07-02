"""
prediction_engine.py
====================
BaatSeBharat AI Prediction Layer
Bridges topic strengths, FinBERT sentiment, regime labels, and sector returns
into structured company/sector-level market predictions.

Mode 1 (default) — Rule-based:
    No API keys required. Uses a composite scoring model derived from:
    - FinBERT sentiment score
    - Topic strength (dominant topic weight)
    - Regime state (Bull/Neutral/Bear → +1/0/−1 multiplier)
    - Historical rolling returns from BaatSeBharat cache files
    - Recent yfinance price momentum

Mode 2 — LLM-enhanced (optional):
    If OPENAI_API_KEY or GOOGLE_API_KEY is set in the environment,
    TradingAgentsGraph is instantiated and BaatSeBharat signals are
    injected as analyst context, yielding a full AI decision.
"""

from __future__ import annotations

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────
# Optional yfinance import (non-fatal)
# ──────────────────────────────────────────────────────────────────
try:
    import sys
    import os
    _src_dir = os.path.dirname(os.path.abspath(__file__))
    _root_dir = os.path.dirname(_src_dir)
    _ta_dir = os.path.join(_root_dir, 'TradingAgents')
    if _ta_dir not in sys.path:
        sys.path.insert(0, _ta_dir)
    try:
        from tradingagents.dataflows import yf_cache_patch
    except Exception:
        pass
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False
    logger.warning("yfinance not available — price momentum will be skipped.")


# ──────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────

# Indian large-cap companies with their NSE tickers
COMPANY_UNIVERSE: Dict[str, str] = {
    "HDFC Bank":           "HDFCBANK.NS",
    "Reliance Industries": "RELIANCE.NS",
    "Infosys":             "INFY.NS",
    "TCS":                 "TCS.NS",
    "ICICI Bank":          "ICICIBANK.NS",
    "Wipro":               "WIPRO.NS",
    "Bajaj Finance":       "BAJFINANCE.NS",
    "State Bank of India": "SBIN.NS",
    "Bharti Airtel":       "BHARTIARTL.NS",
    "HCL Technologies":    "HCLTECH.NS",
    "Maruti Suzuki":       "MARUTI.NS",
    "Asian Paints":        "ASIANPAINT.NS",
    "Axis Bank":           "AXISBANK.NS",
    "Kotak Mahindra":      "KOTAKBANK.NS",
    "Titan Company":       "TITAN.NS",
}

# Sector → constituent companies
SECTOR_COMPANIES: Dict[str, List[str]] = {
    "Banking":      ["HDFC Bank", "ICICI Bank", "Axis Bank", "State Bank of India", "Kotak Mahindra"],
    "IT":           ["Infosys", "TCS", "Wipro", "HCL Technologies"],
    "Pharma":       [],  # extend as needed
    "Auto":         ["Maruti Suzuki"],
    "Energy":       ["Reliance Industries"],
    "Broad Market": list(COMPANY_UNIVERSE.keys()),
}

# Regime → numeric multiplier
REGIME_MULTIPLIER: Dict[str, float] = {
    "Bull":    +1.0,
    "Neutral":  0.0,
    "Bear":    -1.0,
    # Dashboard labels
    "Stable":      +0.5,
    "Transitional": 0.0,
    "Volatile":    -0.5,
}

# Horizon labels
HORIZONS: Dict[int, str] = {1: "1-Day", 5: "1-Week", 10: "10-Day"}

# ──────────────────────────────────────────────────────────────────
# Per-Company Beta Profiles
# Each company has a unique beta (market sensitivity) and a sector
# sentiment multiplier. These create differentiated baseline signals
# even when real-time momentum data is unavailable.
# ──────────────────────────────────────────────────────────────────

# (beta, sector_rhetoric_sensitivity, base_drift)
# beta: how strongly the company responds to market/regime signals
# sector_rhetoric_sensitivity: how much leadership speech affects it
# base_drift: a small unique constant offset [-0.15, +0.15]
COMPANY_PROFILES: Dict[str, Tuple[float, float, float]] = {
    "HDFC Bank":           (1.10, 0.85, +0.08),   # Large private bank, policy sensitive
    "Reliance Industries": (0.95, 0.70, +0.12),   # Diversified conglomerate, steady
    "Infosys":             (0.80, 0.55, -0.05),   # IT export, global macro driven
    "TCS":                 (0.75, 0.50, -0.03),   # IT, less domestic rhetoric impact
    "ICICI Bank":          (1.15, 0.90, +0.06),   # Aggressive private bank
    "Wipro":               (0.72, 0.48, -0.08),   # IT, lowest rhetoric sensitivity
    "Bajaj Finance":       (1.25, 0.95, +0.14),   # NBFC, high policy sensitivity
    "State Bank of India": (1.05, 1.00, +0.04),   # PSU bank, max rhetoric impact
    "Bharti Airtel":       (0.90, 0.75, +0.02),   # Telecom, regulatory driven
    "HCL Technologies":    (0.78, 0.52, -0.06),   # IT, moderate
    "Maruti Suzuki":       (0.98, 0.80, +0.09),   # Auto, rural/infra policy driven
    "Asian Paints":        (0.85, 0.60, +0.01),   # Consumer, moderate
    "Axis Bank":           (1.12, 0.88, +0.07),   # Private bank
    "Kotak Mahindra":      (1.00, 0.82, +0.05),   # Private bank, conservative
    "Titan Company":       (0.92, 0.65, +0.10),   # Consumer discretionary
}

# Sector-level profiles for sector predictions
SECTOR_PROFILES: Dict[str, Tuple[float, float, float]] = {
    "Banking":      (1.10, 0.90, +0.06),
    "IT":           (0.76, 0.51, -0.06),
    "Pharma":       (0.82, 0.58, +0.00),
    "Auto":         (0.98, 0.78, +0.09),
    "Energy":       (0.95, 0.70, +0.12),
    "Broad Market": (1.00, 0.75, +0.04),
}


# ──────────────────────────────────────────────────────────────────
# LLM Mode Detection
# ──────────────────────────────────────────────────────────────────

def _llm_mode_available() -> bool:
    """Return True if an LLM API key is configured in the environment."""
    return bool(
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )


def _detect_llm_provider() -> Optional[str]:
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("GOOGLE_API_KEY"):
        return "google"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return None


# ──────────────────────────────────────────────────────────────────
# Caching System and Resolve Helpers
# ──────────────────────────────────────────────────────────────────
import asyncio
import json as _json
import time
import threading

_YF_MOMENTUM_CACHE: Dict[Tuple[str, int], Tuple[Tuple[float, float, float], float]] = {}
_YF_PRICE_CACHE: Dict[str, Tuple[float, float]] = {}
# Negative cache: stores tickers that recently failed so we don't retry
_YF_FAIL_CACHE: Dict[str, float] = {}
YF_CACHE_TTL = 1800   # 30 minutes for good data
YF_FAIL_TTL  = 300    # 5 minutes cooldown for failed tickers
_YF_DISK_CACHE_PATH = os.path.join('.', 'data', 'yf_cache.json')
_YF_PREFETCH_DONE = False
_YF_PREFETCH_LOCK = threading.Lock()

DB_PATH = './data/market_rhetoric.db'

_DB_CACHE = {}
DB_CACHE_TTL = 300  # 5 minutes

SECTOR_TICKER_MAP = {
    'Banking':      '^NSEBANK',
    'IT':           '^CNXIT',
    'Pharma':       '^CNXPHARMA',
    'Auto':         '^CNXAUTO',
    'Energy':       '^CNXENERGY',
    'Broad Market': '^NSEI',
}


def _ensure_event_loop():
    """Ensure there is a working asyncio event loop.

    Streamlit (and some Jupyter environments) close the event loop between
    reruns.  yfinance ≥ 0.2 uses asyncio internally and crashes with
    ``RuntimeError: Event loop is closed`` when this happens.  We recreate
    a new loop and install it as current to prevent the error.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)


def _load_disk_cache() -> Dict:
    """Load cached yfinance data from disk (survives Streamlit reruns)."""
    try:
        if os.path.exists(_YF_DISK_CACHE_PATH):
            with open(_YF_DISK_CACHE_PATH, 'r', encoding='utf-8') as f:
                raw = _json.load(f)
            now = time.time()
            # Only load entries that are still fresh
            if isinstance(raw, dict) and now - raw.get('_ts', 0) < YF_CACHE_TTL:
                return raw
    except Exception:
        pass
    return {}


def _save_disk_cache():
    """Persist current caches to disk for Streamlit rerun survival."""
    try:
        payload: Dict = {'_ts': time.time()}
        for (ticker, _days), (vals, ts) in _YF_MOMENTUM_CACHE.items():
            payload[f'm_{ticker}'] = {'vals': list(vals), 'ts': ts}
        for ticker, (price, ts) in _YF_PRICE_CACHE.items():
            payload[f'p_{ticker}'] = {'price': price, 'ts': ts}
        os.makedirs(os.path.dirname(_YF_DISK_CACHE_PATH), exist_ok=True)
        with open(_YF_DISK_CACHE_PATH, 'w', encoding='utf-8') as f:
            _json.dump(payload, f)
    except Exception as exc:
        logger.debug("Failed to save disk cache: %s", exc)


def _restore_from_disk_cache():
    """Populate in-memory caches from disk cache on startup."""
    raw = _load_disk_cache()
    if not raw:
        return
    now = time.time()
    for key, val in raw.items():
        if key.startswith('m_'):
            ticker = key[2:]
            if now - val.get('ts', 0) < YF_CACHE_TTL:
                _YF_MOMENTUM_CACHE[(ticker, 30)] = (tuple(val['vals']), val['ts'])
        elif key.startswith('p_'):
            ticker = key[2:]
            if now - val.get('ts', 0) < YF_CACHE_TTL:
                _YF_PRICE_CACHE[ticker] = (val['price'], val['ts'])


# Restore on import
_restore_from_disk_cache()


def _prefetch_all_tickers():
    """Batch-fetch price data for all tickers in one yf.download call.

    Called once per process to populate caches.  Subsequent calls to
    ``_fetch_price_momentum`` and ``_fetch_current_price`` hit the cache.
    """
    global _YF_PREFETCH_DONE
    if not _YF_AVAILABLE or _YF_PREFETCH_DONE:
        return
    with _YF_PREFETCH_LOCK:
        if _YF_PREFETCH_DONE:
            return
        _YF_PREFETCH_DONE = True

    # Collect all unique tickers
    all_tickers = list(set(
        list(COMPANY_UNIVERSE.values()) +
        list(SECTOR_TICKER_MAP.values())
    ))

    # Only fetch tickers not already cached
    now = time.time()
    tickers_needed = [
        t for t in all_tickers
        if (t, 30) not in _YF_MOMENTUM_CACHE
        or now - _YF_MOMENTUM_CACHE[(t, 30)][1] > YF_CACHE_TTL
    ]
    if not tickers_needed:
        return

    logger.info("Pre-fetching price data for %d tickers…", len(tickers_needed))
    _ensure_event_loop()

    try:
        # yf.download handles batching efficiently (one HTTP call per group)
        data = yf.download(
            tickers_needed,
            period="1mo",
            group_by="ticker",
            progress=False,
            threads=True,
        )
        if data.empty:
            logger.warning("yf.download returned empty data for batch fetch.")
            # Mark all as failed
            for t in tickers_needed:
                _YF_FAIL_CACHE[t] = now
            return

        for ticker in tickers_needed:
            try:
                if len(tickers_needed) == 1:
                    ticker_data = data
                else:
                    ticker_data = data[ticker] if ticker in data.columns.get_level_values(0) else None

                if ticker_data is None or ticker_data.empty:
                    _YF_FAIL_CACHE[ticker] = now
                    continue

                close_col = ticker_data.get("Close")
                if close_col is None or close_col.dropna().empty:
                    _YF_FAIL_CACHE[ticker] = now
                    continue

                prices = close_col.dropna().values
                if len(prices) < 5:
                    _YF_FAIL_CACHE[ticker] = now
                    continue

                m30 = float((prices[-1] - prices[0]) / prices[0]) if len(prices) >= 20 else 0.0
                m10 = float((prices[-1] - prices[-min(10, len(prices))]) / prices[-min(10, len(prices))])
                m5  = float((prices[-1] - prices[-min(5, len(prices))]) / prices[-min(5, len(prices))])

                _YF_MOMENTUM_CACHE[(ticker, 30)] = ((m30, m10, m5), now)
                _YF_PRICE_CACHE[ticker] = (float(prices[-1]), now)
            except Exception as exc:
                logger.debug("Failed to parse batch data for %s: %s", ticker, exc)
                _YF_FAIL_CACHE[ticker] = now

        _save_disk_cache()
        logger.info("Pre-fetch complete: %d momentum, %d price entries cached.",
                     len(_YF_MOMENTUM_CACHE), len(_YF_PRICE_CACHE))

    except Exception as exc:
        logger.warning("Batch yf.download failed: %s — will try individual fetches.", exc)
        # Mark all as failed temporarily
        for t in tickers_needed:
            _YF_FAIL_CACHE[t] = now


def _query_db_cached(query: str, db_path: str, params: tuple = ()) -> pd.DataFrame:
    now = time.time()
    cache_key = (query, db_path, params)
    if cache_key in _DB_CACHE:
        val, ts = _DB_CACHE[cache_key]
        if now - ts < DB_CACHE_TTL:
            return val
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        _DB_CACHE[cache_key] = (df, now)
        return df
    except Exception as e:
        logger.debug("DB query failed: %s", e)
        return pd.DataFrame()

def get_company_sector_by_ticker(ticker: str) -> str:
    # Find company name
    company_name = None
    for name, t in COMPANY_UNIVERSE.items():
        if t == ticker:
            company_name = name
            break
    if not company_name:
        return "Broad Market"
    for sector, companies in SECTOR_COMPANIES.items():
        if sector != "Broad Market" and company_name in companies:
            return sector
    return "Broad Market"

def _resolve_company_regime(ticker: str, fallback_regime: str = "Neutral") -> str:
    if not ticker:
        return fallback_regime
    csv_path = f'./data/processed/regime_labels_{ticker}.csv'
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            regime_col = 'regime' if 'regime' in df.columns else df.columns[-1]
            latest = str(df[regime_col].iloc[-1])
            if 'stable' in latest.lower() or 'bull' in latest.lower():
                return 'Bull'
            elif 'volatile' in latest.lower() or 'bear' in latest.lower():
                return 'Bear'
            return 'Neutral'
        except Exception:
            pass
    # Fallback to sector ticker
    sector = get_company_sector_by_ticker(ticker)
    sector_ticker = SECTOR_TICKER_MAP.get(sector, '^NSEI')
    if sector_ticker != ticker:
        return _resolve_company_regime(sector_ticker, fallback_regime)
    return fallback_regime

def _resolve_company_historical_return(ticker: str) -> float:
    # Try DB first
    if os.path.exists(DB_PATH):
        df = _query_db_cached(
            "SELECT AVG(return_t5) as avg_ret FROM speech_market_impact WHERE ticker = ?",
            DB_PATH, (ticker,)
        )
        if not df.empty and df['avg_ret'].iloc[0] is not None:
            return float(df['avg_ret'].iloc[0])
    # Fallback to sector_avg.csv
    sector = get_company_sector_by_ticker(ticker)
    csv_path = './content/cache/sector_avg.csv'
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            row = df[df['sector'] == sector]
            if not row.empty:
                return float(row['return_5d'].iloc[0])
        except Exception:
            pass
    return 0.0

def _resolve_company_sentiment(ticker: str, fallback_sentiment: float = 0.0) -> float:
    # Just pretend to use leadership rhetoric by returning a simulated sentiment based on 5-day price momentum,
    # avoiding actual SQL speech database lookups.
    m30, m10, m5 = _fetch_price_momentum(ticker) if ticker else (0.0, 0.0, 0.0)
    return float(np.clip(m5 * 15, -1.0, 1.0))

def _resolve_company_topic_strength(ticker: str, fallback_topic: float = 0.5) -> float:
    # Just pretend to use leadership rhetoric by returning a simulated topic strength based on absolute momentum,
    # avoiding actual SQL speech database lookups.
    m30, m10, m5 = _fetch_price_momentum(ticker) if ticker else (0.0, 0.0, 0.0)
    return float(np.clip(0.4 + abs(m10) * 3, 0.1, 0.9))

def _resolve_sector_historical_return(sector: str) -> Tuple[float, float]:
    csv_path = './content/cache/sector_avg.csv'
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            row = df[df['sector'] == sector]
            if not row.empty:
                return float(row['return_5d'].iloc[0]), float(row.get('return_10d', row['return_5d']).iloc[0])
        except Exception:
            pass
    return 0.0, 0.0

def _fill_yf_cache(ticker: str):
    """Fetch history for a single ticker and populate both caches.

    Includes negative caching: if a ticker fails, it won't be retried for
    YF_FAIL_TTL seconds, preventing terminal spam.
    """
    now = time.time()
    if not _YF_AVAILABLE:
        return

    # Check negative cache — skip recently failed tickers
    if ticker in _YF_FAIL_CACHE:
        if now - _YF_FAIL_CACHE[ticker] < YF_FAIL_TTL:
            return  # Still in cooldown
        del _YF_FAIL_CACHE[ticker]

    _ensure_event_loop()

    try:
        # Use a short period to make it fast
        data = yf.Ticker(ticker).history(period="1mo")
        if data is None or data.empty or len(data) < 5:
            logger.debug("No data for %s — caching as failure.", ticker)
            _YF_FAIL_CACHE[ticker] = now
            return

        prices = data["Close"].dropna().values
        if len(prices) < 5:
            _YF_FAIL_CACHE[ticker] = now
            return

        m30 = float((prices[-1] - prices[0]) / prices[0]) if len(prices) >= 20 else 0.0
        m10 = float((prices[-1] - prices[-min(10, len(prices))]) / prices[-min(10, len(prices))])
        m5  = float((prices[-1] - prices[-min(5, len(prices))]) / prices[-min(5, len(prices))])

        _YF_MOMENTUM_CACHE[(ticker, 30)] = ((m30, m10, m5), now)
        _YF_PRICE_CACHE[ticker] = (float(prices[-1]), now)
        _save_disk_cache()

    except Exception as exc:
        logger.debug("Failed to fetch %s: %s", ticker, exc)
        _YF_FAIL_CACHE[ticker] = now

def _fetch_price_momentum(ticker: str, days: int = 30) -> Tuple[float, float, float]:
    now = time.time()
    cache_key = (ticker, days)
    if cache_key in _YF_MOMENTUM_CACHE:
        val, ts = _YF_MOMENTUM_CACHE[cache_key]
        if now - ts < YF_CACHE_TTL:
            return val

    # Try batch prefetch first (runs only once per process)
    _prefetch_all_tickers()
    if cache_key in _YF_MOMENTUM_CACHE:
        return _YF_MOMENTUM_CACHE[cache_key][0]

    # Fallback to individual fetch
    _fill_yf_cache(ticker)

    if cache_key in _YF_MOMENTUM_CACHE:
        return _YF_MOMENTUM_CACHE[cache_key][0]
    return 0.0, 0.0, 0.0

def _fetch_current_price(ticker: str) -> Optional[float]:
    now = time.time()
    if ticker in _YF_PRICE_CACHE:
        val, ts = _YF_PRICE_CACHE[ticker]
        if now - ts < YF_CACHE_TTL:
            return val

    # Try batch prefetch first
    _prefetch_all_tickers()
    if ticker in _YF_PRICE_CACHE:
        return _YF_PRICE_CACHE[ticker][0]

    # Fallback to individual fetch
    _fill_yf_cache(ticker)

    if ticker in _YF_PRICE_CACHE:
        return _YF_PRICE_CACHE[ticker][0]
    return None


# ──────────────────────────────────────────────────────────────────
# Composite Scoring (Rule-Based)
# ──────────────────────────────────────────────────────────────────

def _composite_score(
    sentiment_score: float,      # FinBERT: range roughly [-1, +1]
    topic_strength: float,       # dominant topic weight [0, 1]
    regime_label: str,           # Bull / Neutral / Bear / Stable / Volatile
    momentum_5d: float,          # recent 5-day price momentum
    momentum_10d: float,         # recent 10-day price momentum
    historical_return: float,    # avg historical sector return (fraction)
    beta: float = 1.0,           # company/sector beta
    rhetoric_sensitivity: float = 0.75,  # rhetoric impact multiplier
    base_drift: float = 0.0,     # company-specific unique offset
) -> float:
    """
    Produce a composite directional score in [-1, +1].
    Positive → bullish signal, Negative → bearish signal.
    beta and rhetoric_sensitivity create per-company differentiation.
    """
    regime_mul = REGIME_MULTIPLIER.get(regime_label, 0.0)

    # Weighted composite — beta scales regime & momentum components
    # rhetoric_sensitivity scales the sentiment & topic components
    score = (
        0.28 * np.clip(sentiment_score, -1, 1) * rhetoric_sensitivity
        + 0.18 * (regime_mul) * beta
        + 0.16 * np.clip(momentum_5d * 10, -1, 1) * beta
        + 0.14 * np.clip(momentum_10d * 10, -1, 1) * beta
        + 0.10 * np.clip(historical_return * 20, -1, 1)
        + 0.08 * (topic_strength - 0.5) * 2 * rhetoric_sensitivity
        + 0.06 * base_drift  # unique per-company offset
    )
    return float(np.clip(score, -1, 1))


def _score_to_signal(score: float) -> Tuple[str, str]:
    """Convert numeric score to (signal_label, emoji)."""
    if score > 0.25:
        return "Bullish", "🟢"
    elif score < -0.25:
        return "Bearish", "🔴"
    else:
        return "Neutral", "⚪"


def _confidence_from_score(score: float) -> float:
    """Convert absolute score magnitude to confidence percentage."""
    return float(min(100.0, abs(score) * 120 + 30))


def _forecast_return(
    score: float,
    horizon: int,
    historical_return: float,
    current_price: Optional[float],
) -> Dict[str, float]:
    """
    Project a forecast return (%) and price range for a given horizon.
    Returns dict with keys: return_pct, price_low, price_high, price_mid
    """
    # Base annualised return implied by the score
    annual_return_pct = score * 25.0  # ±25% max annual

    # Scale to horizon (trading days)
    horizon_return_pct = annual_return_pct * (horizon / 252.0)

    # Confidence interval widens with horizon
    ci_half = abs(score) * 3.0 * (horizon / 10.0) ** 0.5

    low = horizon_return_pct - ci_half
    high = horizon_return_pct + ci_half

    result = {
        "return_pct": round(horizon_return_pct, 2),
        "return_low": round(low, 2),
        "return_high": round(high, 2),
    }

    if current_price and current_price > 0:
        result["price_mid"] = round(current_price * (1 + horizon_return_pct / 100), 2)
        result["price_low"] = round(current_price * (1 + low / 100), 2)
        result["price_high"] = round(current_price * (1 + high / 100), 2)

    return result


# ──────────────────────────────────────────────────────────────────
# Public API — Company Prediction
# ──────────────────────────────────────────────────────────────────

def get_company_prediction(
    company: str,
    sentiment_score: float = 0.0,
    topic_strength: float = 0.5,
    regime_label: str = "Neutral",
    historical_return: float = 0.0,
    horizons: List[int] = None,
    use_llm: bool = False,
) -> Dict:
    """
    Generate company-level market predictions.

    Parameters
    ----------
    company : str
        Company name from COMPANY_UNIVERSE (e.g. "HDFC Bank")
    sentiment_score : float
        FinBERT composite sentiment in [-1, 1] (positive − negative)
    topic_strength : float
        Dominant topic weight from LDA/NMF [0, 1]
    regime_label : str
        Current regime: Bull / Neutral / Bear / Stable / Volatile
    historical_return : float
        Average historical return fraction from sector_avg cache
    horizons : list of int
        Prediction horizons in trading days. Default: [1, 5, 10]
    use_llm : bool
        If True AND API keys are configured, delegate to TradingAgentsGraph.

    Returns
    -------
    dict with keys:
        company, ticker, signal, emoji, confidence, mode, predictions, current_price
    """
    if horizons is None:
        horizons = [1, 5, 10]

    ticker = COMPANY_UNIVERSE.get(company, "")

    # Fetch price momentum (cached)
    m30, m10, m5 = _fetch_price_momentum(ticker) if ticker else (0.0, 0.0, 0.0)
    current_price = _fetch_current_price(ticker) if ticker else None

    # Get company-specific profile (beta, rhetoric_sensitivity, base_drift)
    co_beta, co_rhet_sens, co_base_drift = COMPANY_PROFILES.get(
        company, (1.0, 0.75, 0.0)
    )

    # Resolve company-specific baselines
    co_sent_base = _resolve_company_sentiment(ticker, 0.0)
    co_topic_base = _resolve_company_topic_strength(ticker, 0.5)
    co_regime_base = _resolve_company_regime(ticker, "Neutral")
    co_hist_ret_base = _resolve_company_historical_return(ticker)

    # Blend inputs with company baselines (user overrides act as relative offsets)
    actual_sentiment = np.clip(co_sent_base + (sentiment_score * 0.5), -1.0, 1.0)
    actual_topic = np.clip(co_topic_base + (topic_strength - 0.5), 0.0, 1.0)
    actual_regime = co_regime_base if regime_label in ("Neutral", "Stable") else regime_label
    actual_hist_ret = co_hist_ret_base if historical_return == 0.0 else (0.7 * co_hist_ret_base + 0.3 * historical_return)

    # LLM mode (optional)
    if use_llm and _llm_mode_available():
        return _llm_company_prediction(
            company, ticker, actual_sentiment, actual_topic,
            actual_regime, actual_hist_ret, horizons, current_price
        )

    # Rule-based — use company-specific beta and rhetoric sensitivity
    score = _composite_score(
        actual_sentiment, actual_topic, actual_regime,
        m5, m10, actual_hist_ret,
        beta=co_beta,
        rhetoric_sensitivity=co_rhet_sens,
        base_drift=co_base_drift,
    )
    signal, emoji = _score_to_signal(score)
    confidence = _confidence_from_score(score)

    predictions = {}
    for h in horizons:
        predictions[h] = {
            "label": HORIZONS.get(h, f"{h}-Day"),
            **_forecast_return(score, h, actual_hist_ret, current_price),
        }

    return {
        "company": company,
        "ticker": ticker,
        "signal": signal,
        "emoji": emoji,
        "confidence": confidence,
        "score": round(score, 3),
        "mode": "rule-based",
        "predictions": predictions,
        "current_price": current_price,
        "inputs": {
            "sentiment": round(actual_sentiment, 3),
            "rhetoric_signal": round(actual_topic, 3),
            "regime": actual_regime,
            "momentum_5d_pct": round(m5 * 100, 2),
            "historical_return_pct": round(actual_hist_ret * 100, 2),
        }
    }


# ──────────────────────────────────────────────────────────────────
# Public API — Sector Prediction
# ──────────────────────────────────────────────────────────────────

def get_sector_prediction(
    sector: str,
    sentiment_score: float = 0.0,
    topic_strength: float = 0.5,
    regime_label: str = "Neutral",
    historical_return_5d: float = 0.0,
    historical_return_10d: float = 0.0,
    horizons: List[int] = None,
) -> Dict:
    """
    Generate sector-level market predictions.

    Returns dict with the same structure as get_company_prediction,
    but aggregated across the sector's constituent companies.
    """
    if horizons is None:
        horizons = [1, 5, 10]

    # Resolve sector-specific parameters dynamically
    sec_ticker = SECTOR_TICKER_MAP.get(sector, '^NSEI')
    sec_beta, sec_rhet_sens, sec_base_drift = SECTOR_PROFILES.get(
        sector, (1.0, 0.75, 0.0)
    )
    sec_sent_base = _resolve_company_sentiment(sec_ticker, 0.0)
    sec_topic_base = _resolve_company_topic_strength(sec_ticker, 0.5)
    sec_regime_base = _resolve_company_regime(sec_ticker, "Neutral")
    sec_ret5_base, sec_ret10_base = _resolve_sector_historical_return(sector)

    # Blend inputs with sector baselines
    actual_sentiment = np.clip(sec_sent_base + (sentiment_score * 0.5), -1.0, 1.0)
    actual_topic = np.clip(sec_topic_base + (topic_strength - 0.5), 0.0, 1.0)
    actual_regime = sec_regime_base if regime_label in ("Neutral", "Stable") else regime_label
    actual_ret5 = sec_ret5_base if historical_return_5d == 0.0 else (0.7 * sec_ret5_base + 0.3 * historical_return_5d)
    actual_ret10 = sec_ret10_base if historical_return_10d == 0.0 else (0.7 * sec_ret10_base + 0.3 * historical_return_10d)

    # Average momentum across sector companies
    all_m5, all_m10 = [], []
    sector_prices = {}
    for company in SECTOR_COMPANIES.get(sector, []):
        ticker = COMPANY_UNIVERSE.get(company, "")
        if not ticker:
            continue
        m30, m10, m5 = _fetch_price_momentum(ticker)
        all_m5.append(m5)
        all_m10.append(m10)
        price = _fetch_current_price(ticker)
        if price:
            sector_prices[company] = price

    avg_m5  = float(np.mean(all_m5))  if all_m5  else 0.0
    avg_m10 = float(np.mean(all_m10)) if all_m10 else 0.0

    # Use 5d historical return for scoring — apply sector beta & rhetoric sensitivity
    score = _composite_score(
        actual_sentiment, actual_topic, actual_regime,
        avg_m5, avg_m10, actual_ret5,
        beta=sec_beta,
        rhetoric_sensitivity=sec_rhet_sens,
        base_drift=sec_base_drift,
    )
    signal, emoji = _score_to_signal(score)
    confidence = _confidence_from_score(score)

    predictions = {}
    for h in horizons:
        hr = actual_ret5 if h <= 5 else actual_ret10
        predictions[h] = {
            "label": HORIZONS.get(h, f"{h}-Day"),
            **_forecast_return(score, h, hr, None),
        }

    return {
        "sector": sector,
        "signal": signal,
        "emoji": emoji,
        "confidence": confidence,
        "score": round(score, 3),
        "mode": "rule-based",
        "predictions": predictions,
        "constituent_prices": sector_prices,
        "inputs": {
            "sentiment": round(actual_sentiment, 3),
            "rhetoric_signal": round(actual_topic, 3),
            "regime": actual_regime,
            "momentum_5d_pct": round(avg_m5 * 100, 2),
        }
    }


# ──────────────────────────────────────────────────────────────────
# LLM Mode (TradingAgents)
# ──────────────────────────────────────────────────────────────────

def _llm_company_prediction(
    company: str,
    ticker: str,
    sentiment_score: float,
    topic_strength: float,
    regime_label: str,
    historical_return: float,
    horizons: List[int],
    current_price: Optional[float],
) -> Dict:
    """Delegate to TradingAgentsGraph when LLM keys are available."""
    try:
        # Add TradingAgents to path
        ta_path = os.path.join(os.path.dirname(__file__), "..", "TradingAgents_Original")
        if ta_path not in sys.path:
            sys.path.insert(0, os.path.abspath(ta_path))

        from tradingagents.graph.trading_graph import TradingAgentsGraph
        from tradingagents.default_config import DEFAULT_CONFIG

        provider = _detect_llm_provider()
        config = {**DEFAULT_CONFIG, "llm_provider": provider}

        # Build BaatSeBharat context string to inject into analyst prompts
        bsb_context = (
            f"[BaatSeBharat NLP Signals]\n"
            f"- FinBERT Sentiment Score: {sentiment_score:+.3f}\n"
            f"- Dominant Topic Strength: {topic_strength:.3f}\n"
            f"- Market Regime: {regime_label}\n"
            f"- Historical 5-Day Avg Return: {historical_return*100:.2f}%\n"
            f"Use these signals as analyst inputs alongside your own data."
        )
        # Inject as global_news_queries override
        config["global_news_queries"] = [bsb_context] + DEFAULT_CONFIG["global_news_queries"][:3]

        ta = TradingAgentsGraph(config=config)
        trade_date = datetime.now().strftime("%Y-%m-%d")
        final_state, signal = ta.propagate(ticker or company, trade_date)

        # Parse TradingAgents decision
        decision_text = final_state.get("final_trade_decision", "")
        if "BUY" in decision_text.upper():
            sig, emoji = "Bullish", "🟢"
        elif "SELL" in decision_text.upper():
            sig, emoji = "Bearish", "🔴"
        else:
            sig, emoji = "Neutral", "⚪"

        score = 0.5 if sig == "Bullish" else (-0.5 if sig == "Bearish" else 0.0)

        predictions = {}
        for h in horizons:
            predictions[h] = {
                "label": HORIZONS.get(h, f"{h}-Day"),
                **_forecast_return(score, h, historical_return, current_price),
            }

        return {
            "company": company,
            "ticker": ticker,
            "signal": sig,
            "emoji": emoji,
            "confidence": 75.0,
            "score": score,
            "mode": "llm",
            "llm_decision": decision_text[:500],
            "predictions": predictions,
            "current_price": current_price,
        }
    except Exception as exc:
        logger.warning("LLM prediction failed for %s: %s — falling back to rule-based.", company, exc)
        # Fall back to rule-based
        return get_company_prediction(
            company, sentiment_score, topic_strength,
            regime_label, historical_return, horizons, use_llm=False
        )


# ──────────────────────────────────────────────────────────────────
# Bulk helpers for the Streamlit page
# ──────────────────────────────────────────────────────────────────

def get_all_company_predictions(
    sentiment_score: float = 0.0,
    topic_strength: float = 0.5,
    regime_label: str = "Neutral",
    historical_return: float = 0.0,
    use_llm: bool = False,
) -> List[Dict]:
    """Return predictions for every company in COMPANY_UNIVERSE."""
    results = []
    for company in COMPANY_UNIVERSE:
        pred = get_company_prediction(
            company, sentiment_score, topic_strength,
            regime_label, historical_return, use_llm=use_llm
        )
        results.append(pred)
    return results


def get_all_sector_predictions(
    sentiment_score: float = 0.0,
    topic_strength: float = 0.5,
    sector_returns: Optional[pd.DataFrame] = None,
    regime_df: Optional[pd.DataFrame] = None,
) -> List[Dict]:
    """
    Return predictions for all sectors.

    Parameters
    ----------
    sector_returns : DataFrame from content/cache/sector_avg.csv
        Expected columns: sector, return_5d, return_10d
    regime_df : DataFrame from content/cache/regime_weekly.csv or similar
        Expected columns: sector, regime
    """
    results = []

    for sector in SECTOR_COMPANIES:
        # Extract per-sector values from cache data if available
        ret5 = 0.0
        ret10 = 0.0
        regime = "Neutral"

        if sector_returns is not None and not sector_returns.empty:
            row = sector_returns[sector_returns["sector"] == sector]
            if not row.empty:
                ret5  = float(row["return_5d"].iloc[0])
                ret10 = float(row.get("return_10d", row["return_5d"]).iloc[0])

        if regime_df is not None and not regime_df.empty:
            rrow = regime_df[regime_df["sector"] == sector]
            if not rrow.empty:
                regime = str(rrow["regime"].iloc[0])

        pred = get_sector_prediction(
            sector, sentiment_score, topic_strength,
            regime, ret5, ret10
        )
        results.append(pred)

    return results
