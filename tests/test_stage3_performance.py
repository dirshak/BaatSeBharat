"""
Regression test for "Stage 3 isn't loading".

Root cause (reproduced with streamlit.testing.v1.AppTest, which timed out
at 180s): App_v2.py's Stage 3 ("Market Impact") called fig.add_vline() once
per speech-market-impact event to draw a vertical marker line -- for a
heavily-covered ticker that's ~1000+ individual add_vline() calls. Plotly
recomputes the whole figure layout on every add_vline/add_shape call, which
is effectively O(n^2) in the number of events and made the page hang for
minutes instead of rendering.

Fixed by drawing all of a source's vertical lines as a single Scatter
trace (x/y interleaved with None separators), which is O(n).

This test loads the real app end-to-end (real DB, real pipeline output)
and asserts Stage 3 renders well under the old 180s timeout and without
exceptions.
"""
import time

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest

from conftest import goto_stage

# Generous vs. the old failure mode (180s timeout) but tight enough to catch
# a regression back to O(n^2) chart-building.
MAX_STAGE3_SECONDS = 30


def test_stage3_market_impact_loads_quickly():
    at = AppTest.from_file('App_v2.py', default_timeout=120)
    at.run()
    assert not at.exception, f"Executive Summary raised: {list(at.exception)}"

    start = time.time()
    goto_stage(at, '3. Market Impact')
    elapsed = time.time() - start

    assert not at.exception, f"Stage 3 raised: {list(at.exception)}"
    assert elapsed < MAX_STAGE3_SECONDS, (
        f"Stage 3 took {elapsed:.1f}s to render (must be < {MAX_STAGE3_SECONDS}s). "
        "This is the exact symptom of the add_vline() O(n^2) regression."
    )
