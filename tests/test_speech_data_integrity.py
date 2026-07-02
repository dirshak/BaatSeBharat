"""
Regression tests for the Executive Summary speech-count bug.

Root cause: the `speeches` table itself contained duplicate rows, not a bad
COUNT(*) computation. Two independent sources of duplication:

1. `CentralizedSpeechScraper.scrape_mann_ki_baat()` globbed
   `mann_ki_baat_*.txt` unfiltered. The transcripts/mann_ki_baat/ directory
   also accumulates corrupted legacy files from older buggy runs whose
   header line is a previously-written header re-ingested as a title and
   re-saved (growing longer each round trip, e.g.
   "Episode 1 | 2019-06-30 | PM Modi | Speech | None | PM Modi | Speech...").
   Every pipeline run re-ingested this corruption as "new" episodes,
   inflating Mann Ki Baat from 70 real episodes to 2723 DB rows.
2. Fed/ECB had a smaller number of genuine re-scrape duplicates (same
   source/date/title/content inserted more than once across separate runs).

Fixed by: restricting the MKB glob to `mann_ki_baat_\\d+\\.txt` (the
canonical clean filename pattern), re-ingesting Mann Ki Baat from the clean
local files, deduplicating Fed/ECB by content hash, and adding a
UNIQUE(source, date, title) index so future duplicate inserts are rejected
at the DB level rather than relying solely on the app-level pre-check.
"""
import glob
import os
import re
import sqlite3

import pandas as pd
import pytest

DB_PATH = './data/market_rhetoric.db'
TRANSCRIPTS_DIR = './transcripts/mann_ki_baat'


@pytest.fixture
def db_connection():
    conn = sqlite3.connect(DB_PATH)
    yield conn
    conn.close()


def test_no_duplicate_speech_content(db_connection):
    """No two speeches from the same source should share identical full_text."""
    df = pd.read_sql_query(
        "SELECT source, full_text, COUNT(*) as n FROM speeches "
        "GROUP BY source, full_text HAVING n > 1",
        db_connection
    )
    assert df.empty, f"Found duplicate-content speech rows: {df.to_dict('records')}"


def test_no_duplicate_source_date_title(db_connection):
    """No two speeches should share (source, date, title) -- this is also
    enforced by a UNIQUE index at the DB level."""
    df = pd.read_sql_query(
        "SELECT source, date, title, COUNT(*) as n FROM speeches "
        "GROUP BY source, date, title HAVING n > 1",
        db_connection
    )
    assert df.empty, f"Found duplicate (source, date, title) rows: {df.to_dict('records')}"


def test_unique_index_exists(db_connection):
    cur = db_connection.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='speeches'")
    indexes = {row[0] for row in cur.fetchall()}
    assert 'idx_speeches_source_date_title' in indexes


def test_mkb_glob_excludes_legacy_corrupted_files():
    """The MKB transcript loader must only pick up canonical
    `mann_ki_baat_<episode_number>.txt` files, not legacy/corrupted ones
    left in the same directory by older buggy runs.
    """
    if not os.path.isdir(TRANSCRIPTS_DIR):
        pytest.skip("Transcript directory not present in this environment")
    all_files = glob.glob(os.path.join(TRANSCRIPTS_DIR, 'mann_ki_baat_*.txt'))
    canonical = [
        f for f in all_files
        if re.fullmatch(r'mann_ki_baat_\d+\.txt', os.path.basename(f))
    ]
    # Directory should contain non-canonical legacy files that must NOT be
    # ingested; if this ever becomes false the corruption may have been
    # cleaned up, which is fine, but the canonical set must never be empty.
    assert len(canonical) > 0

    from src.data.centralized_scraper import CentralizedSpeechScraper
    import inspect
    src = inspect.getsource(CentralizedSpeechScraper.scrape_mann_ki_baat)
    assert r'mann_ki_baat_\d+\.txt' in src, (
        "scrape_mann_ki_baat() must filter to canonical numeric filenames only"
    )


def test_mann_ki_baat_speech_count_matches_canonical_files(db_connection):
    """DB row count for Mann Ki Baat must match the number of canonical
    (non-corrupted) local transcript files, not exceed it."""
    if not os.path.isdir(TRANSCRIPTS_DIR):
        pytest.skip("Transcript directory not present in this environment")
    canonical = [
        f for f in glob.glob(os.path.join(TRANSCRIPTS_DIR, 'mann_ki_baat_*.txt'))
        if re.fullmatch(r'mann_ki_baat_\d+\.txt', os.path.basename(f))
    ]
    df = pd.read_sql_query(
        "SELECT COUNT(*) as n FROM speeches WHERE source = 'Mann Ki Baat'",
        db_connection
    )
    assert df['n'].iloc[0] <= len(canonical)
