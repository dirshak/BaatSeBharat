"""
classify_speeches_groq.py
==========================
Batch-classifies speeches (Mann Ki Baat, ECB, Fed) against the tracked
company universe using src/models/llm_topic_classifier.py.

For each speech that already has a topic model result (topic_distributions,
model_name='Combined' -- see src/models/topic_modeling.py), this pulls the
speech's dominant topic label/keywords from data/processed/topic_labels_combined.json,
sends the (topic + excerpt) to Groq, and stores the per-company
strength/sentiment verdicts in llm_company_signals.

Idempotent: speeches already classified under the configured llm_model are
skipped, so re-running only processes new/unclassified speeches (or all of
them again if the model name in config/config.yaml changes).

Usage:
    python scripts/classify_speeches_groq.py [--limit N] [--source "Mann Ki Baat"]
"""

import argparse
import json
import os
import sqlite3
import sys
import time

import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.llm_topic_classifier import GroqTopicClassifier
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

DB_PATH = './data/market_rhetoric.db'
TOPIC_LABELS_PATH = './data/processed/topic_labels_combined.json'


def _load_topic_labels(path=TOPIC_LABELS_PATH):
    if not os.path.exists(path):
        logger.warning(f"Topic labels file not found: {path} -- run topic_modeling.py first.")
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _fetch_pending_speeches(conn, llm_model, source=None, limit=None):
    """Speeches with a dominant topic already computed that haven't been
    classified yet under `llm_model`."""
    query = """
        SELECT s.id, s.source, s.title, s.full_text,
               td.topic_id, td.probability
        FROM speeches s
        JOIN (
            SELECT speech_id, topic_id, probability,
                   ROW_NUMBER() OVER (PARTITION BY speech_id ORDER BY probability DESC) as rn
            FROM topic_distributions
            WHERE model_name = 'Combined'
        ) td ON td.speech_id = s.id AND td.rn = 1
        LEFT JOIN llm_company_signals lcs
            ON lcs.speech_id = s.id AND lcs.llm_model = ?
        WHERE lcs.id IS NULL AND s.full_text IS NOT NULL AND s.full_text != ''
    """
    params = [llm_model]
    if source:
        query += " AND s.source = ?"
        params.append(source)
    # NULLs sort first in SQLite's default ascending order, which would
    # otherwise push undated (often corrupted/legacy-ingested) rows to the
    # front of every batch run ahead of genuinely dated speeches.
    query += " ORDER BY s.date IS NULL, s.date"
    if limit:
        query += " LIMIT ?"
        params.append(int(limit))

    return pd.read_sql_query(query, conn, params=params)


def run(limit=None, source=None, sleep_between=0.5):
    classifier = GroqTopicClassifier()
    topic_labels = _load_topic_labels()

    conn = sqlite3.connect(DB_PATH)
    pending = _fetch_pending_speeches(conn, classifier.model, source=source, limit=limit)

    if pending.empty:
        logger.info("No pending speeches to classify (all done, or no topic model output yet).")
        conn.close()
        return 0

    logger.info(f"Classifying {len(pending)} speeches with model '{classifier.model}'...")

    total_signals = 0
    for _, row in tqdm(pending.iterrows(), total=len(pending)):
        topic_info = topic_labels.get(str(row['topic_id']), {})
        topic_label = topic_info.get('label')
        topic_keywords = topic_info.get('keywords', [])

        try:
            n = classifier.classify_and_store(
                conn, row['id'], row['full_text'],
                topic_label=topic_label, topic_keywords=topic_keywords,
            )
            conn.commit()
            total_signals += n
        except Exception as exc:
            logger.error(f"Failed to classify speech {row['id']} ({row['title']}): {exc}")

        time.sleep(sleep_between)

    conn.close()
    logger.info(f"Done. Classified {len(pending)} speeches -> {total_signals} company signals stored.")
    return total_signals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify speeches against the company universe via Groq.")
    parser.add_argument("--limit", type=int, default=None, help="Max number of speeches to classify this run.")
    parser.add_argument("--source", type=str, default=None,
                         help="Restrict to one source, e.g. 'Mann Ki Baat', 'ECB', 'Fed'.")
    parser.add_argument("--sleep", type=float, default=0.5, help="Seconds to sleep between API calls.")
    args = parser.parse_args()

    run(limit=args.limit, source=args.source, sleep_between=args.sleep)
