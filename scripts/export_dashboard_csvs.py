"""
export_dashboard_csvs.py
=========================
Generates the "source" CSVs dashboard.py and precompute_cache.py expect
under content/ (topic_dataset.csv, final_topics.csv, nmf_topics.csv) from
the live database -- these three were previously produced by an older,
no-longer-present pipeline stage and had gone stale/missing, breaking the
Speech Audit, Topic Explorer, and Market Dynamics "Topic Strength" pages.

Run this after the topic model / sentiment pipeline (scripts/run_prototype.py
Steps 3-4) so it reflects current data, then re-run precompute_cache.py to
refresh the derived cache/topics_enriched.csv and cache/sentiment_timeline.csv.
"""

import json
import os
import sqlite3

import pandas as pd

DB_PATH = './data/market_rhetoric.db'
DATA_DIR = 'content'
TOPIC_LABELS_PATH = './data/processed/topic_labels_combined.json'

TOPIC_DATASET_CSV = os.path.join(DATA_DIR, 'topic_dataset.csv')
FINAL_TOPICS_CSV = os.path.join(DATA_DIR, 'final_topics.csv')
NMF_TOPICS_CSV = os.path.join(DATA_DIR, 'nmf_topics.csv')


def export_topic_dataset(conn):
    """One row per speech with sentiment -- feeds Speech Audit's
    load_speeches() and precompute_cache.py's sentiment_timeline.csv."""
    df = pd.read_sql_query('''
        SELECT s.id, s.date, s.source, s.title, s.full_text as text,
               ss.positive, ss.negative, ss.neutral
        FROM speeches s
        LEFT JOIN sentiment_scores ss
            ON ss.speech_id = s.id AND ss.segment_type = 'episode'
        WHERE s.full_text IS NOT NULL AND s.full_text != ''
    ''', conn)
    df['filename'] = df['title'].fillna('untitled').astype(str) + '.txt'
    df[['positive', 'negative', 'neutral']] = df[['positive', 'negative', 'neutral']].fillna(0.0)
    df[['date', 'source', 'filename', 'text', 'positive', 'negative', 'neutral']].to_csv(
        TOPIC_DATASET_CSV, index=False
    )
    print(f"  -> {TOPIC_DATASET_CSV} ({len(df)} rows)")


def export_topic_tables(conn):
    """Per-topic label/keywords/score for the 'Combined' model -- feeds
    Topic Explorer, Market Dynamics' 'Topic Strength vs Sector Returns',
    and precompute_cache.py's topics_enriched.csv."""
    if not os.path.exists(TOPIC_LABELS_PATH):
        print(f"  WARNING: {TOPIC_LABELS_PATH} not found -- run topic modeling first.")
        return

    with open(TOPIC_LABELS_PATH, 'r', encoding='utf-8') as f:
        labels = json.load(f)

    avg_prob = pd.read_sql_query('''
        SELECT topic_id, AVG(probability) as score
        FROM topic_distributions
        WHERE model_name = 'Combined'
        GROUP BY topic_id
    ''', conn).set_index('topic_id')['score'].to_dict()

    rows = []
    for topic_id_str, info in labels.items():
        topic_id = int(topic_id_str)
        rows.append({
            'topic_id': topic_id,
            'label': info.get('label', f'Topic {topic_id}'),
            'keywords': ', '.join(info.get('keywords', [])),
            'score': round(float(avg_prob.get(topic_id, 0.0)), 4),
        })

    df = pd.DataFrame(rows).sort_values('score', ascending=False)
    df[['topic_id', 'keywords', 'score']].to_csv(FINAL_TOPICS_CSV, index=False)
    print(f"  -> {FINAL_TOPICS_CSV} ({len(df)} rows)")
    df[['topic_id', 'label', 'keywords', 'score']].to_csv(NMF_TOPICS_CSV, index=False)
    print(f"  -> {NMF_TOPICS_CSV} ({len(df)} rows)")


if __name__ == '__main__':
    conn = sqlite3.connect(DB_PATH)
    print("Exporting dashboard source CSVs from the database...")
    export_topic_dataset(conn)
    export_topic_tables(conn)
    conn.close()
    print("Done. Now re-run: python precompute_cache.py")
