"""Shared test helpers for streamlit.testing.v1.AppTest-based tests.

The stage navigation moved from a left sidebar (st.sidebar.radio, then a
sidebar button stepper) to a horizontal top nav bar (st.container(key=...,
horizontal=True) + st.button, see App_v2.py's top-bar section) with SHORT
display labels distinct from the full STAGES routing values. goto_stage()
centralizes the STAGES-name -> short-nav-label mapping and the button
lookup so individual tests don't each hardcode it.
"""

# Must mirror App_v2.py's STAGES / NAV_LABELS lists exactly.
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
