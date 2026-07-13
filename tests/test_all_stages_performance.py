"""
Task 9: all pages must load quickly. This is the whole-app companion to
tests/test_stage3_performance.py and tests/test_stage4_performance.py.

Measured before fixing (via streamlit.testing.v1.AppTest, see conversation
history / PR description for the full before/after table):
  Executive Summary : 4.5s cold  / 0.1s warm  (cold = one-time heavy imports)
  1. Data Ingestion  : uncached, pulled full_text for ~1300 speeches just to
                        populate a title dropdown -> now cached + lazy text fetch
  3. Market Impact   : 180s+ (timeout) -> 1.16s -> further cached
  4. Regime Intelligence: 2.21s (60% in Plotly add_vrect) -> 0.26s -> cached
  5. Company Analytics: uncached -> cached
  6. AI Predictions  : ~5-6s cold (yfinance/World Bank network I/O, not
                        reducible without changing product behavior) -> <0.3s warm
  7. Global Influence Map: ~2s cold (World Bank network I/O) -> <0.4s warm

This test asserts every stage renders without exceptions and that a SECOND
visit (i.e. everything should be cache-warm) is fast -- this is the
regression guard: if someone removes an @st.cache_data decorator, this
test catches the resulting slow re-query on revisit even if the first-load
time is legitimately network-bound and hard to bound tightly.
"""
import time

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest

from conftest import goto_stage

STAGES = [
    "Executive Summary",
    "1. Data Ingestion",
    "2. NLP Intelligence",
    "3. Market Impact",
    "4. Regime Intelligence",
    "5. Company Analytics",
    "6. AI Predictions",
    "7. Global Influence Map",
]

MAX_WARM_SECONDS = 3.0


def test_every_stage_loads_without_exceptions_and_is_fast_when_warm():
    at = AppTest.from_file('App_v2.py', default_timeout=120)
    at.run()
    assert not at.exception, f"Initial load raised: {list(at.exception)}"

    # First pass: visit every stage once (may include legitimate one-time
    # network I/O for Stage 6/7 -- not bounded here).
    for stage in STAGES[1:]:
        goto_stage(at, stage)
        assert not at.exception, f"{stage} raised on first visit: {list(at.exception)}"

    # Second pass: everything should now be cache-warm.
    for stage in STAGES:
        start = time.time()
        goto_stage(at, stage)
        elapsed = time.time() - start
        assert not at.exception, f"{stage} raised on warm revisit: {list(at.exception)}"
        assert elapsed < MAX_WARM_SECONDS, (
            f"{stage} took {elapsed:.2f}s on a cache-warm revisit "
            f"(must be < {MAX_WARM_SECONDS}s) -- likely a missing/broken @st.cache_data."
        )
