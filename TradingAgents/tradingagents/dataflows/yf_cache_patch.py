import yfinance as yf
import pandas as pd
import sqlite3
import time
import json
import os
import hashlib
import logging
from yfinance.exceptions import YFRateLimitError

logger = logging.getLogger("yf_cache_patch")

# Cache configuration
USER_HOME = os.path.expanduser("~")
DB_DIR = os.path.join(USER_HOME, ".tradingagents")
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "yf_global_cache.db")

def init_db():
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT,
                is_error INTEGER,
                exception_type TEXT,
                timestamp REAL
            )
        """)
        conn.commit()
    except Exception as e:
        logger.debug(f"Failed to initialize SQLite cache db: {e}")
    finally:
        conn.close()

# Initialize on import
init_db()

def get_cached(key: str, ttl: float, fail_ttl: float = 600.0):
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT value, is_error, exception_type, timestamp FROM cache WHERE key = ?", (key,))
        row = cursor.fetchone()
        if row:
            val, is_error, exc_type, ts = row
            now = time.time()
            current_ttl = fail_ttl if is_error else ttl
            if now - ts < current_ttl:
                if is_error:
                    return {"status": "error", "exception_type": exc_type, "message": val}
                return {"status": "success", "value": val}
            else:
                cursor.execute("DELETE FROM cache WHERE key = ?", (key,))
                conn.commit()
    except Exception as e:
        logger.debug(f"Cache get failed for {key}: {e}")
    finally:
        conn.close()
    return None

def set_cached(key: str, val: str, is_error: int = 0, exc_type: str = None):
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(
            "INSERT OR REPLACE INTO cache (key, value, is_error, exception_type, timestamp) VALUES (?, ?, ?, ?, ?)",
            (key, val, is_error, exc_type, time.time())
        )
        conn.commit()
    except Exception as e:
        logger.debug(f"Cache set failed for {key}: {e}")
    finally:
        conn.close()

def serialize_df(df):
    tz_str = str(df.index.tz) if (isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None) else None
    return json.dumps({
        'data': df.to_json(orient='split', date_format='iso'),
        'index_is_datetime': isinstance(df.index, pd.DatetimeIndex),
        'index_name': df.index.name,
        'tz': tz_str
    })

def deserialize_df(serialized_str):
    dct = json.loads(serialized_str)
    df = pd.read_json(dct['data'], orient='split')
    if dct['index_is_datetime']:
        df.index = pd.to_datetime(df.index)
        if dct.get('tz') and dct['tz'] != 'None':
            try:
                df.index = df.index.tz_convert(dct['tz'])
            except TypeError:
                try:
                    df.index = df.index.tz_localize(dct['tz'])
                except Exception:
                    pass
    if dct.get('index_name'):
        df.index.name = dct['index_name']
    return df

def raise_cached_error(exc_type, message):
    if exc_type == 'YFRateLimitError':
        raise YFRateLimitError(message)
    else:
        raise RuntimeError(message)

# Save reference to original functions
real_download = yf.download
real_Ticker = yf.Ticker

def cached_download(*args, **kwargs):
    key_parts = {
        'method': 'download',
        'args': args,
        'kwargs': kwargs
    }
    key_str = json.dumps(key_parts, sort_keys=True, default=str)
    key_hash = hashlib.md5(key_str.encode('utf-8')).hexdigest()
    key = f"download_{key_hash}"

    # Use 1 day TTL for price history
    cached = get_cached(key, ttl=86400.0)
    if cached:
        if cached['status'] == 'error':
            raise_cached_error(cached['exception_type'], cached['message'])
        return deserialize_df(cached['value'])

    try:
        df = real_download(*args, **kwargs)
        set_cached(key, serialize_df(df))
        return df
    except Exception as e:
        exc_type = type(e).__name__
        set_cached(key, str(e), is_error=1, exc_type=exc_type)
        raise

class CachedTicker:
    def __init__(self, ticker_name):
        self._ticker_name = ticker_name
        self._real_ticker = real_Ticker(ticker_name)

    def _get_cached_value(self, attribute_name, fetch_func, ttl=86400.0):
        key = f"ticker_{self._ticker_name}_{attribute_name}"
        cached = get_cached(key, ttl=ttl)
        if cached:
            if cached['status'] == 'error':
                raise_cached_error(cached['exception_type'], cached['message'])
            return json.loads(cached['value'])

        try:
            val = fetch_func()
            set_cached(key, json.dumps(val))
            return val
        except Exception as e:
            exc_type = type(e).__name__
            set_cached(key, str(e), is_error=1, exc_type=exc_type)
            raise

    def _get_cached_df(self, attribute_name, fetch_func, ttl=86400.0, key_suffix=""):
        key = f"ticker_{self._ticker_name}_{attribute_name}{key_suffix}"
        cached = get_cached(key, ttl=ttl)
        if cached:
            if cached['status'] == 'error':
                raise_cached_error(cached['exception_type'], cached['message'])
            return deserialize_df(cached['value'])

        try:
            df = fetch_func()
            set_cached(key, serialize_df(df))
            return df
        except Exception as e:
            exc_type = type(e).__name__
            set_cached(key, str(e), is_error=1, exc_type=exc_type)
            raise

    @property
    def info(self):
        return self._get_cached_value('info', lambda: self._real_ticker.info)

    @property
    def quarterly_balance_sheet(self):
        return self._get_cached_df('quarterly_balance_sheet', lambda: self._real_ticker.quarterly_balance_sheet)

    @property
    def balance_sheet(self):
        return self._get_cached_df('balance_sheet', lambda: self._real_ticker.balance_sheet)

    @property
    def quarterly_cashflow(self):
        return self._get_cached_df('quarterly_cashflow', lambda: self._real_ticker.quarterly_cashflow)

    @property
    def cashflow(self):
        return self._get_cached_df('cashflow', lambda: self._real_ticker.cashflow)

    @property
    def quarterly_income_stmt(self):
        return self._get_cached_df('quarterly_income_stmt', lambda: self._real_ticker.quarterly_income_stmt)

    @property
    def income_stmt(self):
        return self._get_cached_df('income_stmt', lambda: self._real_ticker.income_stmt)

    @property
    def insider_transactions(self):
        return self._get_cached_df('insider_transactions', lambda: self._real_ticker.insider_transactions)

    def history(self, *args, **kwargs):
        key_parts = {
            'args': args,
            'kwargs': kwargs
        }
        key_str = json.dumps(key_parts, sort_keys=True, default=str)
        key_hash = hashlib.md5(key_str.encode('utf-8')).hexdigest()
        return self._get_cached_df('history', lambda: self._real_ticker.history(*args, **kwargs), key_suffix=f"_{key_hash}")

    def __getattr__(self, name):
        return getattr(self._real_ticker, name)

# Apply global monkey patching on import
yf.download = cached_download
yf.Ticker = CachedTicker
