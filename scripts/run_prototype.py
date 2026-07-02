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
from src.models.topic_modeling import HybridTopicModeler, ZeroShotLabeler
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
    inserted = 0

    for _, row in speeches_df.iterrows():
        try:
            event_date = pd.to_datetime(row['date'])
        except Exception:
            continue

        for ticker in tickers:
            ticker_data = market_df[market_df['ticker'] == ticker]['returns']

            def forward_return(n_days):
                future = ticker_data[ticker_data.index > event_date]
                future = future.iloc[:n_days] if len(future) >= n_days else future
                if future.empty:
                    return None
                # Cumulative return: (1+r1)(1+r2)... - 1
                return float(np.prod(1 + future.values) - 1)

            r1 = forward_return(1)
            r5 = forward_return(5)
            r10 = forward_return(10)

            # Abnormal return: r5 minus the mean 5-day return of the ticker
            mean_5d = float(ticker_data.rolling(5).sum().mean()) if len(ticker_data) > 5 else None
            abnormal = (r5 - mean_5d) if (r5 is not None and mean_5d is not None) else None

            try:
                conn.execute('''
                    INSERT INTO speech_market_impact
                    (speech_id, ticker, event_date, return_t1, return_t5, return_t10, abnormal_return)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (int(row['id']), ticker, row['date'], r1, r5, r10, abnormal))
                inserted += 1
            except Exception as e:
                logger.error(f"Impact insert error: {e}")

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

    processed_count = 0
    for _, row in df_speeches.iterrows():
        try:
            processed = preprocessor.preprocess(row['full_text'])
            conn.execute(
                "UPDATE speeches SET processed_text = ? WHERE id = ?",
                (processed, row['id'])
            )
            
            # Sentiment Overlay using FinBERT
            if sentiment_analyzer:
                # To prevent memory issues with long text, we just use the first 512 tokens implicitly in analyze_sentiment
                scores = sentiment_analyzer.analyze_sentiment(row['full_text'])
                
                # Check if entry already exists
                existing = pd.read_sql_query(f"SELECT id FROM sentiment_scores WHERE speech_id={row['id']} AND segment_type='episode'", conn)
                if existing.empty:
                    conn.execute('''
                        INSERT INTO sentiment_scores 
                        (speech_id, segment_type, optimism_intensity, risk_awareness, positive, negative, neutral, compound)
                        VALUES (?, 'episode', ?, ?, ?, ?, ?, ?)
                    ''', (
                        row['id'], scores['optimism_intensity'], scores['risk_awareness'],
                        scores['positive'], scores['negative'], scores['neutral'], scores['compound']
                    ))

            processed_count += 1
        except Exception as e:
            logger.warning(f"Preprocess/Sentiment error id={row['id']}: {e}")

    conn.commit()
    logger.info(f"Done: Preprocessed and sentiment-analyzed {processed_count} speeches.")

    # 4. Multi-Source Topic Modeling
    logger.info("Step 4: Multi-Source Topic Modeling...")
    
    def train_and_save_model(name, query):
        logger.info(f"Training Topic Model: {name}...")
        df_subset = pd.read_sql_query(query, conn)
        docs = df_subset['processed_text'].tolist()
        speech_ids = df_subset['id'].tolist()
        
        if len(docs) < 2:
            logger.warning(f"Not enough data for model '{name}' ({len(docs)} documents).")
            return
            
        n_topics = min(10, len(docs))
        modeler = HybridTopicModeler(n_topics=n_topics)
        # Load real embeddings
        embeddings_dict = np.load('./data/processed/speech_embeddings.npy', allow_pickle=True).item()
        embeddings = np.array([embeddings_dict[sid] for sid in speech_ids if sid in embeddings_dict])
        
        if len(embeddings) == 0:
            logger.warning(f"No embeddings found for model '{name}'. Skipping.")
            return
        
        # Ensure docs match embeddings if some were missing
        valid_indices = [i for i, sid in enumerate(speech_ids) if sid in embeddings_dict]
        docs = [docs[i] for i in valid_indices]
        speech_ids = [speech_ids[i] for i in valid_indices]
        
        try:
            os.makedirs('./data/processed', exist_ok=True)
            consensus, dists = modeler.fit_ensemble(docs, embeddings)
            
            # Save to npy
            filename = f'topic_distributions_{name.lower().replace(" ", "_")}.npy'
            np.save(f'./data/processed/{filename}', consensus)
            logger.info(f"✓ Saved {filename}")
            
            # Persist to DB
            conn.execute("DELETE FROM topic_distributions WHERE model_name = ?", (name,))
            for i, speech_id in enumerate(speech_ids):
                if i >= len(consensus): break
                for topic_id, prob in enumerate(consensus[i]):
                    conn.execute('''
                        INSERT OR REPLACE INTO topic_distributions 
                        (speech_id, topic_id, probability, model_name)
                        VALUES (?, ?, ?, ?)
                    ''', (int(speech_id), topic_id, float(prob), name))
            conn.commit()
            logger.info(f"Done: Persisted {name} model to DB.")
            
            # Save keywords for UI
            topic_labels = modeler.get_topic_labels()
            import json
            labels_file = f'topic_labels_{name.lower().replace(" ", "_")}.json'
            with open(f'./data/processed/{labels_file}', 'w') as f:
                json.dump(topic_labels, f)
            logger.info(f"Done: Saved keywords to {labels_file}")
        except Exception as e:
            logger.error(f"Failed training {name}: {e}")

    # Define tasks
    model_tasks = [
        ("Combined", "SELECT id, processed_text FROM speeches WHERE processed_text IS NOT NULL AND processed_text != ''"),
        ("Fed", "SELECT id, processed_text FROM speeches WHERE source='Fed' AND processed_text IS NOT NULL AND processed_text != ''"),
        ("ECB", "SELECT id, processed_text FROM speeches WHERE source='ECB' AND processed_text IS NOT NULL AND processed_text != ''"),
        ("Mann Ki Baat", "SELECT id, processed_text FROM speeches WHERE source='Mann Ki Baat' AND processed_text IS NOT NULL AND processed_text != ''"),
        ("Prototype", "SELECT id, processed_text FROM speeches WHERE processed_text IS NOT NULL AND processed_text != '' LIMIT 10") # For backward compatibility with App_v2
    ]
    
    for name, query in model_tasks:
        train_and_save_model(name, query)

    # Label topics using ZeroShot
    logger.info("Applying Zero-Shot Classification to Topics...")
    try:
        labeler = ZeroShotLabeler()
        for name, _ in model_tasks:
            # We would technically load labels and map them. Simple check ensures it doesn't crash if model misses
            labels_file = f'topic_labels_{name.lower().replace(" ", "_")}.json'
            if os.path.exists(f'./data/processed/{labels_file}'):
                import json
                with open(f'./data/processed/{labels_file}', 'r') as f:
                    topic_labels = json.load(f)
                
                for t_key, t_val in topic_labels.items():
                    if 'keywords' in t_val:
                        top_label, score = labeler.classify_keywords(t_val['keywords'])
                        t_val['zero_shot_domain'] = top_label
                        t_val['zero_shot_score'] = score
                
                with open(f'./data/processed/{labels_file}', 'w') as f:
                    json.dump(topic_labels, f, indent=2)
        logger.info("Zero-Shot Classification complete.")
    except Exception as e:
        logger.warning(f"ZeroShot labeling failed: {e}")

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
