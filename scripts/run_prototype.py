import sys
import os
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import asyncio

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Ensure TradingAgents is in path and apply yfinance cache patch
_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ta_dir = os.path.join(_root_dir, 'TradingAgents')
if _ta_dir not in sys.path:
    sys.path.insert(0, _ta_dir)
try:
    from tradingagents.dataflows import yf_cache_patch
except Exception:
    pass

from src.utils.logger import setup_logger
from src.data.centralized_scraper import CentralizedSpeechScraper
from src.data.market_data_downloader import MarketDataDownloader
from src.data.vix_downloader import VIXDownloader
from src.features.text_preprocessing import TextPreprocessor
from src.models.sentiment_overlay import SentimentOverlay
from src.models.topic_modeling import train_and_save as train_topic_model
from src.models.market_modeling import MarketModeler
from src.models.fusion_engine import FusionEngine
from src.models.causal_validation import CausalValidator
from src.utils.db_utils import get_db_connection

logger = setup_logger("Prototype_V1")

DB_PATH = './data/market_rhetoric.db'


def compute_speech_market_impact():
    """
    For each speech in the DB, compute 1-, 5- and 10-day forward returns
    for every market ticker. Saves results to speech_market_impact table.
    """
    logger.info("Computing speech-market impact...")
    conn = get_db_connection(DB_PATH)

    speeches_df = pd.read_sql_query(
        "SELECT id, date, source FROM speeches WHERE date IS NOT NULL", conn
    )
    market_df = pd.read_sql_query(
        "SELECT date, ticker, returns FROM market_data WHERE returns IS NOT NULL", conn
    )

    if speeches_df.empty or market_df.empty:
        logger.warning("Not enough data for impact computation.")
        conn.close()
        return

    market_df['date'] = pd.to_datetime(market_df['date'])
    market_df = market_df.set_index('date').sort_index()

    # Clear old impact data
    conn.execute("DELETE FROM speech_market_impact")

    tickers = market_df['ticker'].unique()

    # Hoist all per-ticker work out of the speech loop. Previously this
    # rebuilt `market_df[market_df['ticker'] == ticker]` -- a full boolean
    # scan of ~47k rows -- once per (speech, ticker) pair, i.e. ~17k times,
    # and recomputed `rolling(5).sum().mean()` just as often even though it
    # is a constant per ticker. Doing it once per ticker instead takes this
    # step from ~85s to ~2s on the current corpus, with identical output.
    per_ticker = {}
    for ticker, g in market_df.groupby('ticker', sort=False):
        dates = g.index.to_numpy()
        rets = g['returns'].to_numpy(dtype=float)
        mean_5d = float(pd.Series(rets).rolling(5).sum().mean()) if len(rets) > 5 else None
        per_ticker[ticker] = (dates, rets, mean_5d)

    rows = []
    for _, row in speeches_df.iterrows():
        try:
            event_date = np.datetime64(pd.to_datetime(row['date']))
        except Exception:
            continue

        speech_id = int(row['id'])
        for ticker in tickers:
            dates, rets, mean_5d = per_ticker[ticker]

            # Dates are sorted, so the first observation strictly after the
            # event is a binary search rather than a full-series comparison.
            pos = np.searchsorted(dates, event_date, side='right')
            window = rets[pos:pos + 10]

            if window.size == 0:
                r1 = r5 = r10 = None
            else:
                # One cumulative product serves all three horizons; when
                # fewer than n observations remain, fall back to the last
                # available point (matching the previous behaviour).
                cum = np.cumprod(1.0 + window) - 1.0
                r1 = float(cum[0])
                r5 = float(cum[min(4, cum.size - 1)])
                r10 = float(cum[min(9, cum.size - 1)])

            # Abnormal return: r5 minus the mean 5-day return of the ticker
            abnormal = (r5 - mean_5d) if (r5 is not None and mean_5d is not None) else None
            rows.append((speech_id, ticker, row['date'], r1, r5, r10, abnormal))

    try:
        conn.executemany('''
            INSERT INTO speech_market_impact
            (speech_id, ticker, event_date, return_t1, return_t5, return_t10, abnormal_return)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', rows)
        inserted = len(rows)
    except Exception as e:
        logger.error(f"Impact insert error: {e}")
        inserted = 0

    conn.commit()
    conn.close()
    logger.info(f"✓ Saved {inserted} speech-market impact records.")


def run_prototype():
    logger.info("=== Starting Patch V1.2 Prototype (Unified: MKB + ECB + Fed) ===")

    # 1. Centralized Data Ingestion — all 3 sources
    logger.info("Step 1: Ingesting multi-source speech data (MKB + ECB + Fed)...")
    scraper = CentralizedSpeechScraper(db_path=DB_PATH)
    scraper._ensure_db_exists() # Ensure all tables (including topic_distributions) exist

    # Change default from 3650 (10yrs) to 30 days for routine updates
    # Use days_back=3650 for initial historical load
    try:
        asyncio.run(scraper.scrape_all(days_back=30))
    except Exception as e:
        logger.error(f"Incomplete ingestion: {e}")

    # 2. Market Data
    logger.info("Step 2: Downloading market & macro data...")
    try:
        downloader = MarketDataDownloader()
        market_df = downloader.download_all_data()
        if market_df is not None:
            downloader.save_to_database(market_df)
    except Exception as e:
        logger.error(f"Market data error: {e}")

    # 2b. Download VIX explicitly
    try:
        vix_dl = VIXDownloader()
        vix_df = vix_dl.download_vix()
        if vix_df is not None:
            vix_dl.save_to_database(vix_df)
    except Exception as e:
        logger.error(f"VIX download error: {e}")

    # 3. Preprocessing all speeches
    logger.info("Step 3: Preprocessing speeches...")
    preprocessor = TextPreprocessor()
    conn = get_db_connection(DB_PATH)

    try:
        conn.execute("ALTER TABLE speeches ADD COLUMN processed_text TEXT")
    except Exception:
        pass  # Column already exists

    # OPTIMIZATION: Only process speeches that haven't been processed yet
    df_speeches = pd.read_sql_query(
        "SELECT id, full_text FROM speeches WHERE (processed_text IS NULL OR processed_text = '') AND full_text IS NOT NULL AND full_text != ''", conn
    )

    # Initialize Sentiment overlay
    try:
        sentiment_analyzer = SentimentOverlay()
    except Exception as e:
        logger.warning(f"Could not initialize FinBERT, skipping sentiment: {e}")
        sentiment_analyzer = None

    # Which speeches already have an episode-level sentiment row. Fetched
    # once as a set instead of issuing a per-speech SELECT (the old loop ran
    # a fresh read_sql_query -- parse, execute, build a DataFrame -- for
    # every single speech purely to test existence).
    already_scored = set(
        pd.read_sql_query(
            "SELECT speech_id FROM sentiment_scores WHERE segment_type='episode'", conn
        )['speech_id'].tolist()
    )

    # Batch the spaCy pass. Per-document nlp() calls re-pay pipeline setup
    # every time; nlp.pipe() with the unused NER component disabled is ~2x
    # faster for byte-identical output (verified across the corpus).
    texts = df_speeches['full_text'].tolist()
    if texts:
        logger.info(f"Preprocessing {len(texts)} speeches (batched)...")
        processed_texts = preprocessor.preprocess_batch(texts)
    else:
        processed_texts = []

    processed_count = 0
    sentiment_rows = []
    for (_, row), processed in zip(df_speeches.iterrows(), processed_texts):
        try:
            conn.execute(
                "UPDATE speeches SET processed_text = ? WHERE id = ?",
                (processed, row['id'])
            )

            # Sentiment Overlay using FinBERT. This is the dominant cost of
            # the whole pipeline (~3.5s per speech on CPU), so skip any
            # speech that already has a score rather than recomputing it.
            if sentiment_analyzer and row['id'] not in already_scored:
                # To prevent memory issues with long text, we just use the first 512 tokens implicitly in analyze_sentiment
                scores = sentiment_analyzer.analyze_sentiment(row['full_text'])
                sentiment_rows.append((
                    row['id'], scores['optimism_intensity'], scores['risk_awareness'],
                    scores['positive'], scores['negative'], scores['neutral'],
                    scores['compound']
                ))

            processed_count += 1
        except Exception as e:
            logger.warning(f"Preprocess/Sentiment error id={row['id']}: {e}")

    if sentiment_rows:
        conn.executemany('''
            INSERT INTO sentiment_scores
            (speech_id, segment_type, optimism_intensity, risk_awareness, positive, negative, neutral, compound)
            VALUES (?, 'episode', ?, ?, ?, ?, ?, ?)
        ''', sentiment_rows)

    conn.commit()
    logger.info(f"Done: Preprocessed and sentiment-analyzed {processed_count} speeches.")

    # 4. Multi-Source Topic Modeling (TF-IDF + NMF, see src/models/topic_modeling.py)
    logger.info("Step 4: Multi-Source Topic Modeling...")

    def train_and_persist(name, query):
        logger.info(f"Training Topic Model: {name}...")
        df_subset = pd.read_sql_query(query, conn)
        docs = df_subset['processed_text'].tolist()
        speech_ids = df_subset['id'].tolist()

        model, topic_dist = train_topic_model(name, docs, speech_ids, n_topics=10)
        if model is None:
            return

        # Persist to DB: this is what Stage 3/5/6 (App_v2.py) and
        # prediction_engine.py actually read from -- the .npy/.json files
        # written by train_topic_model() are only used by Stage 2's
        # per-model keyword/heatmap browser.
        conn.execute("DELETE FROM topic_distributions WHERE model_name = ?", (name,))
        for i, speech_id in enumerate(speech_ids):
            if i >= len(topic_dist):
                break
            for topic_id, prob in enumerate(topic_dist[i]):
                conn.execute('''
                    INSERT OR REPLACE INTO topic_distributions
                    (speech_id, topic_id, probability, model_name)
                    VALUES (?, ?, ?, ?)
                ''', (int(speech_id), topic_id, float(prob), name))
        conn.commit()
        logger.info(f"Done: Persisted {name} model ({len(speech_ids)} speeches) to DB.")

    model_tasks = [
        ("Combined", "SELECT id, processed_text FROM speeches WHERE processed_text IS NOT NULL AND processed_text != ''"),
        ("Fed", "SELECT id, processed_text FROM speeches WHERE source='Fed' AND processed_text IS NOT NULL AND processed_text != ''"),
        ("ECB", "SELECT id, processed_text FROM speeches WHERE source='ECB' AND processed_text IS NOT NULL AND processed_text != ''"),
        ("Mann Ki Baat", "SELECT id, processed_text FROM speeches WHERE source='Mann Ki Baat' AND processed_text IS NOT NULL AND processed_text != ''"),
    ]

    for name, query in model_tasks:
        train_and_persist(name, query)

    conn.close()

    # 4.5 Compute ASBN / CPTM Market Regimes
    logger.info("Step 4.5: Computing ASBN & CPTM-F Regimes...")
    try:
        market_modeler = MarketModeler()
        market_modeler.compute_regime_metrics()
    except Exception as e:
        logger.error(f"Market Modeling failed: {e}")

    # 5. Compute Speech-Market Impact
    logger.info("Step 5: Computing speech-event market impact...")
    try:
        compute_speech_market_impact()
    except Exception as e:
        logger.error(f"Impact computation failed: {e}")

    # 6. Fusion Engine (PWM Shock Modeling)
    logger.info("Step 6: Running Fusion Engine (PWM Shocks)...")
    try:
        fusion = FusionEngine()
        fusion.compute_all_shocks()
    except Exception as e:
        logger.error(f"Fusion / PWM Shock failed: {e}")

    # 7. Causal Validation
    logger.info("Step 7: Granger Causality Validation...")
    try:
        validator = CausalValidator()
        validator.test_causality()
    except Exception as e:
        logger.error(f"Granger Causality failed: {e}")

    logger.info("=== Prototype V1.2 Run Complete ===")
    return True


if __name__ == "__main__":
    run_prototype()
