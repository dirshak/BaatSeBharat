"""Shared test helpers for streamlit.testing.v1.AppTest-based tests.

The stage navigation moved from a left sidebar (st.sidebar.radio, then a
sidebar button stepper) to a horizontal top nav bar (st.container(key=...,
horizontal=True) + st.button, see App.py's top-bar section) with SHORT
display labels distinct from the full STAGES routing values. goto_stage()
centralizes the STAGES-name -> short-nav-label mapping and the button
lookup so individual tests don't each hardcode it.
"""

import os

# Absolute path to the Streamlit entrypoint, resolved from this file's
# location. Tests must use APP_PATH rather than passing a bare filename to
# AppTest.from_file(): a bare name resolves relative to this tests/
# directory, not the project root, so it can never match the real app. That
# is exactly how the App_v2.py -> App.py rename went unnoticed -- all three
# app-loading tests silently became "File not found" assertion errors
# instead of real coverage, which in turn hid a startup crash in App.py.
# Centralising the path here means a future rename breaks one line, loudly.
APP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'App.py'
)


def _is_streamlit_app(path):
    """True only if APP_PATH is still a Streamlit script.

    Upstream replaced the Streamlit App.py with a 58-line launcher that
    builds the React frontend and starts FastAPI. AppTest-based tests then
    fail with a confusing '[WinError 2] file not found' from the launcher's
    `npm install` subprocess -- which looks like a broken test rather than
    a removed feature. Detect the architecture explicitly instead.
    """
    if not os.path.isfile(path):
        return False
    try:
        with open(path, encoding='utf-8') as f:
            head = f.read(4000)
    except OSError:
        return False
    return 'import streamlit' in head or 'streamlit as st' in head


STREAMLIT_APP_AVAILABLE = _is_streamlit_app(APP_PATH)

SKIP_NO_STREAMLIT = (
    "App.py is no longer a Streamlit script -- the UI moved to React + "
    "FastAPI (backend/routers/). These stage-render performance guards "
    "need reimplementing against the backend endpoints; see "
    "backend/routers/status.py for the equivalent surface."
)


def test_app_entrypoint_exists():
    """Guard: fail loudly here if the app is renamed, rather than letting
    every AppTest-based test degrade into a confusing 'File not found'."""
    assert os.path.isfile(APP_PATH), (
        f"App entrypoint not found at {APP_PATH}. If the app was "
        "renamed, update APP_PATH in tests/conftest.py."
    )


# Must mirror App.py's STAGES / NAV_LABELS lists exactly.
_STAGES = [
    "Executive Summary",
    "1. Data Ingestion",
    "2. NLP Intelligence",
    "3. Market Impact",
    "4. Regime Intelligence",
    "5. Company Analytics",
    "6. AI Predictions",
    "7. Global Influence Map",
]
_NAV_LABELS = [
    "Overview",
    "01 Ingestion",
    "02 NLP",
    "03 Impact",
    "04 Regime",
    "05 Company",
    "06 Predictions",
    "07 Global",
]
_STAGE_TO_NAV_LABEL = dict(zip(_STAGES, _NAV_LABELS))


def goto_stage(at, stage_name):
    """Click the top-nav button for `stage_name` (a STAGES value, e.g.
    '3. Market Impact') and rerun the app."""
    nav_label = _STAGE_TO_NAV_LABEL.get(stage_name, stage_name)
    for b in at.button:
        if b.label == nav_label:
            b.click()
            break
    else:
        raise AssertionError(f"No top-nav button found for stage {stage_name!r} (label {nav_label!r})")
    at.run()
