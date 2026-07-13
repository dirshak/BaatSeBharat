import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import sys
import subprocess
from datetime import datetime
import json

# ── Add src to path ────────────────────────────────────────────────────────
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_APP_DIR, 'src'))
sys.path.insert(0, os.path.join(_APP_DIR, 'TradingAgents'))

try:
    from tradingagents.dataflows import yf_cache_patch
except Exception:
    pass

from utils.logger import setup_logger
from utils.db_utils import get_db_connection

# ── Integration imports (non-fatal: app still works without them) ──────────
try:
    from prediction_engine import (
        get_all_company_predictions,
        get_all_sector_predictions,
        get_company_prediction,
        COMPANY_UNIVERSE,
        SECTOR_COMPANIES,
        _llm_mode_available,
    )
    _PRED_OK = True
except Exception as _pred_err:
    _PRED_OK = False
    _pred_err_msg = str(_pred_err)

try:
    from geo_dashboard import render_global_influence_map
    _GEO_OK = True
except Exception as _geo_err:
    _GEO_OK = False
    _geo_err_msg = str(_geo_err)

try:
    from prediction_history import compute_prediction_vs_actual, summarize as summarize_prediction_history
    _PREDHIST_OK = True
except Exception as _predhist_err:
    _PREDHIST_OK = False
    _predhist_err_msg = str(_predhist_err)

from pathlib import Path

_LOGO_PATH = Path(__file__).parent / "logo.png"

_LOGO_EXISTS = _LOGO_PATH.exists()
st.set_page_config(
    page_title="Leadership Rhetoric Driven Market Intelligence",
    page_icon=str(_LOGO_PATH) if _LOGO_EXISTS else "🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)
if not _LOGO_EXISTS:
    # Flag missing logo rather than silently falling back to the emoji
    # favicon -- st.set_page_config() must run first (it's the very first
    # Streamlit call), so this warning has to come after it.
    st.warning(f"⚠️ logo.png not found at `{_LOGO_PATH}` — using a fallback emoji favicon instead.")

# ===========================================================================
# DESIGN SYSTEM — palette, type, and component overrides
# ===========================================================================
# Color roles (see design plan): Ink Navy (bg), Ledger (surface), Parchment
# (text), Signal Saffron (rhetoric/accent), Market Green (positive),
# Rust Alert (negative). Exposed as CSS vars so every stage's chart code can
# reference the same palette via COLORS/PLOTLY_TEMPLATE below.
COLORS = {
    "bg":        "#0B1220",
    "surface":   "#131B2C",
    "surface2":  "#0F1727",
    "ink":       "#E8E4D9",
    "ink_dim":   "#9AA3B5",
    "line":      "#26324A",
    "saffron":   "#C97A2B",
    "saffron_dim": "#8A5A24",
    "green":     "#2F6F4E",
    "green_dim": "#1F4A34",
    "rust":      "#A6503A",
    "navy":      "#1B2A4A",
}

# Shared Plotly chart chrome so every figure across all 7 stages reads as
# the same product instead of default Plotly styling with swapped colors.
PLOTLY_TEMPLATE = dict(
    paper_bgcolor=COLORS["surface"],
    plot_bgcolor=COLORS["surface"],
    font=dict(family="IBM Plex Sans, sans-serif", color=COLORS["ink"], size=12),
    title_font=dict(family="Fraunces, serif", color=COLORS["ink"], size=18),
    xaxis=dict(gridcolor=COLORS["line"], zerolinecolor=COLORS["line"], linecolor=COLORS["line"],
                tickfont=dict(family="IBM Plex Mono, monospace", color=COLORS["ink_dim"], size=11)),
    yaxis=dict(gridcolor=COLORS["line"], zerolinecolor=COLORS["line"], linecolor=COLORS["line"],
                tickfont=dict(family="IBM Plex Mono, monospace", color=COLORS["ink_dim"], size=11)),
    legend=dict(font=dict(family="IBM Plex Sans, sans-serif", color=COLORS["ink_dim"], size=11),
                bgcolor="rgba(0,0,0,0)"),
    margin=dict(t=60, l=10, r=10, b=10),
)

# Categorical sequence for multi-source charts (pie/bar by source, etc.)
# derived from the same three logo hues instead of default Plotly colors.
CATEGORY_SEQUENCE = [COLORS["saffron"], COLORS["green"], COLORS["navy"], COLORS["rust"], "#5A7A9A", "#7A5A3A"]

# Sequential scale for heatmaps/intensity: surface (low) -> saffron (high),
# replacing Plotly's default "Blues" so heatmaps read as this product's
# palette instead of generic Plotly output.
SEQUENTIAL_SCALE = [[0.0, COLORS["surface2"]], [0.5, "#5A4A2E"], [1.0, COLORS["saffron"]]]
# Diverging scale for signed values (returns, correlation): rust (neg) ->
# surface (~0) -> green (pos).
DIVERGING_SCALE = [[0.0, COLORS["rust"]], [0.5, COLORS["surface2"]], [1.0, COLORS["green"]]]

def apply_chart_theme(fig, height=None):
    """Apply the shared design-system chrome to a Plotly figure in place."""
    fig.update_layout(**PLOTLY_TEMPLATE)
    if height:
        fig.update_layout(height=height)
    return fig

def metric_row(items):
    """Render a horizontal 'ledger row' of stat cells — replaces boxed
    st.metric() cards with a hairline-separated strip (label above,
    Plex Mono number below), consistent with the design plan.
    items: list of (label, value, sublabel_or_None) or
           (label, value, sublabel_or_None, tooltip_or_None) tuples.
    """
    # NOTE: this HTML must stay on a single line per cell -- st.markdown runs
    # unsafe_allow_html content through a CommonMark parser first, and any
    # line indented 4+ spaces is treated as an indented code block (which
    # silently rendered the 2nd-4th cells as literal escaped text the first
    # time this was written with pretty-printed multi-line f-strings).
    cells = []
    for item in items:
        label, value, sub = item[0], item[1], item[2]
        tooltip = item[3] if len(item) > 3 else None
        title_attr = f' title="{tooltip}"' if tooltip else ""
        sub_html = f'<div class="ledger-sub">{sub}</div>' if sub else ""
        cells.append(
            f'<div class="ledger-cell"{title_attr}><div class="ledger-label">{label}</div>'
            f'<div class="ledger-value">{value}</div>{sub_html}</div>'
        )
    st.markdown(f'<div class="ledger-row">{"".join(cells)}</div>', unsafe_allow_html=True)

def stage_header(number, title, subtitle=None):
    """Consistent stage header: small-caps mono eyebrow (STAGE NN) + a
    Fraunces headline, replacing ad hoc st.title("<emoji> Stage N: ...")
    calls with one shared, on-system pattern across all 7 stages."""
    st.markdown(f'<div class="stage-eyebrow">Stage {number}</div>', unsafe_allow_html=True)
    st.title(title)
    if subtitle:
        st.markdown(f'<p style="color:{COLORS["ink_dim"]};margin-top:-0.6rem">{subtitle}</p>', unsafe_allow_html=True)

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    /* ── Hide native Streamlit chrome so this reads as a standalone app ──
       Selectors verified against the actual rendered DOM for the installed
       Streamlit version (1.55) rather than assumed from older docs: this
       version uses data-testid="stHeader"/"stToolbar", not the classic
       bare <header>/#MainMenu/<footer> alone (those still exist as aliases
       but stToolbar/stHeader are what's actually populated with content
       here). The sidebar no longer has any content rendered into it (see
       the top nav below), but its expand/collapse controls are hidden too
       in case an empty <section> still gets a toggle affordance. */
    header[data-testid="stHeader"], #MainMenu, footer,
    [data-testid="stToolbar"], [data-testid="stDecoration"],
    [data-testid="stStatusWidget"] {{
        visibility: hidden; height: 0; display: none;
    }}
    section[data-testid="stSidebar"],
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stExpandSidebarButton"] {{
        display: none !important;
    }}

    :root {{
        --bg: {COLORS["bg"]}; --surface: {COLORS["surface"]}; --surface2: {COLORS["surface2"]};
        --ink: {COLORS["ink"]}; --ink-dim: {COLORS["ink_dim"]}; --line: {COLORS["line"]};
        --saffron: {COLORS["saffron"]}; --green: {COLORS["green"]}; --rust: {COLORS["rust"]}; --navy: {COLORS["navy"]};
    }}

    html, body, [class*="css"] {{ font-family: 'IBM Plex Sans', sans-serif; }}
    .stApp {{ background-color: var(--bg); color: var(--ink); }}
    .main .block-container {{ padding-top: 2rem; max-width: 1200px; }}

    h1, h2, h3 {{ font-family: 'Fraunces', serif; font-weight: 500; color: var(--ink); letter-spacing: -0.01em; }}
    h1 {{ font-size: 2.1rem; }}
    h2 {{ font-size: 1.5rem; }}
    h3 {{ font-size: 1.15rem; }}
    p, li, span, label, div {{ color: var(--ink); }}

    /* Stage eyebrow label — small-caps letter-spaced tag above headline */
    .stage-eyebrow {{
        font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; letter-spacing: 0.14em;
        text-transform: uppercase; color: var(--saffron); margin-bottom: 0.2rem;
    }}

    /* Numbers everywhere use the mono face */
    .stMetric [data-testid="stMetricValue"], code, .ledger-value {{ font-family: 'IBM Plex Mono', monospace; }}

    /* ── Ledger metric row (replaces boxed st.metric cards) ────────────── */
    .ledger-row {{
        display: flex; flex-wrap: wrap; gap: 0; border-top: 1px solid var(--line);
        border-bottom: 1px solid var(--line); margin: 0.5rem 0 1.25rem 0;
    }}
    .ledger-cell {{
        flex: 1 1 160px; padding: 0.85rem 1.1rem; border-right: 1px solid var(--line);
    }}
    .ledger-cell:last-child {{ border-right: none; }}
    .ledger-label {{
        font-family: 'IBM Plex Sans', sans-serif; font-size: 0.68rem; letter-spacing: 0.1em;
        text-transform: uppercase; color: var(--ink-dim); margin-bottom: 0.3rem;
    }}
    .ledger-value {{ font-size: 1.6rem; font-weight: 500; color: var(--ink); line-height: 1.1; }}
    .ledger-sub {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: var(--ink-dim); margin-top: 0.2rem; }}

    /* ── Top header row: brand mark + Run Pipeline button ──────────────── */
    .topbar-brand {{ display: flex; align-items: center; gap: 0.6rem; }}
    .topbar-brand img {{ width: 34px; height: 34px; border-radius: 50%; }}
    .topbar-brand-name {{ font-family: 'Fraunces', serif; font-size: 1.15rem; color: var(--ink); line-height: 1.1; }}
    .topbar-brand-sub {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.62rem; letter-spacing: 0.12em;
                          color: var(--ink-dim); text-transform: uppercase; }}
    .st-key-runpipeline button {{ float: right; }}

    /* ── Horizontal stage nav (replaces the old sidebar stepper) ───────
       Row of content-sized buttons inside st.container(key="topnav",
       horizontal=True) -- a real flexbox row (verified against the
       rendered DOM), not stretched st.columns().

       position:sticky does NOT hold here despite computing correctly
       (verified with getBoundingClientRect before/after a programmatic
       scroll, and even with an inline !important override) -- Streamlit's
       block containers carry a `data-test-scroll-behavior` attribute and
       evidently run their own JS layout/scroll management on
       stHorizontalBlock/stVerticalBlock elements that fights native
       sticky. A plain injected <div> at the same DOM depth stuck
       correctly, isolating this to Streamlit's own container elements,
       not a CSS mistake. position:fixed (full viewport width, edge to
       edge -- common for app chrome even when content below is
       constrained to --content-max-width) works reliably instead; see the
       .topnav-spacer element right after this container in the Python
       code, which reserves the vertical space fixed positioning removes
       from the normal document flow.

       overflow-x:auto is the graceful-degradation path on viewports too
       narrow to fit all 8 short labels; nowrap keeps it a single
       scrollable row instead of Streamlit's default wrap (which would
       stack pills unevenly). */
    .st-key-topnav {{
        position: fixed !important; top: 0; left: 0; right: 0; width: 100%; z-index: 999;
        background-color: var(--surface2); border-bottom: 1px solid var(--line);
        padding: 0.4rem 1.5rem;
        overflow-x: auto; flex-wrap: nowrap !important; gap: 0.3rem !important;
    }}
    .topnav-spacer {{ height: 52px; }}
    /* Each pill's wrapper must not shrink below its content width, or
       Streamlit's default flex-shrink lets buttons compress and collide
       instead of the row scrolling -- confirmed by screenshot at 800px
       viewport width where labels ran together with no gap. Direct flex
       children of the horizontal container are stLayoutWrapper divs (each
       wrapping one st.container(key=f"nav_{{i}}")), verified against the
       rendered DOM rather than assumed. */
    .st-key-topnav > div[data-testid="stLayoutWrapper"] {{
        flex-shrink: 0 !important;
    }}
    .st-key-topnav div[data-testid="stButton"] button {{
        white-space: nowrap; background-color: transparent; border: none;
        border-bottom: 2px solid transparent; border-radius: 0;
        padding: 0.45rem 0.75rem; margin: 0;
        font-family: 'IBM Plex Sans', sans-serif; font-size: 0.84rem; font-weight: 400;
        color: var(--ink-dim); transition: background-color 0.15s, color 0.15s, border-color 0.15s;
    }}
    .st-key-topnav div[data-testid="stButton"] button:hover {{
        background-color: rgba(201,122,43,0.08); color: var(--ink);
    }}
    .st-key-topnav div[data-testid="stButton"] button:focus-visible {{
        outline: 2px solid var(--saffron); outline-offset: -2px;
    }}
    /* Active stage: primary-typed button gets a solid underline, standing
       in for the old sidebar's colored-dot progress signal. */
    .st-key-topnav div[data-testid="stButton"] button[kind="primary"],
    .st-key-topnav div[data-testid="stButton"] button[kind="primaryFormSubmit"] {{
        background-color: rgba(201,122,43,0.14) !important; color: var(--ink) !important;
        border-bottom-color: var(--saffron) !important; font-weight: 500 !important;
        box-shadow: none !important;
    }}
    /* Per-pill accent sequencing saffron -> navy -> green across the 8
       stages (each button carries its own stable st.button(key=...)
       class, so this doesn't rely on sibling position/nth-of-type). */
    .st-key-nav_0 div[data-testid="stButton"] button {{ border-bottom-color: rgba(201,122,43,0.35); }}
    .st-key-nav_1 div[data-testid="stButton"] button {{ border-bottom-color: rgba(176,106,46,0.35); }}
    .st-key-nav_2 div[data-testid="stButton"] button {{ border-bottom-color: rgba(150,97,64,0.35); }}
    .st-key-nav_3 div[data-testid="stButton"] button {{ border-bottom-color: rgba(27,42,74,0.5); }}
    .st-key-nav_4 div[data-testid="stButton"] button {{ border-bottom-color: rgba(35,74,69,0.5); }}
    .st-key-nav_5 div[data-testid="stButton"] button {{ border-bottom-color: rgba(40,90,70,0.5); }}
    .st-key-nav_6 div[data-testid="stButton"] button {{ border-bottom-color: rgba(47,111,78,0.5); }}
    .st-key-nav_7 div[data-testid="stButton"] button {{ border-bottom-color: rgba(47,111,78,0.5); }}
    .st-key-nav_8 div[data-testid="stButton"] button {{ border-bottom-color: rgba(47,111,78,0.5); }}

    /* Generic buttons elsewhere (Run Pipeline, etc.) */
    .stButton button, .main div[data-testid="stButton"] button {{
        background-color: var(--surface); border: 1px solid var(--line); color: var(--ink);
        border-radius: 4px; font-family: 'IBM Plex Sans', sans-serif;
    }}
    .main div[data-testid="stButton"] button:hover {{ border-color: var(--saffron); color: var(--saffron); }}
    .st-key-topnav div[data-testid="stButton"] button:hover {{ border-color: transparent; border-bottom-color: var(--saffron); }}

    /* Status strip (relocated from the old sidebar footer) */
    .status-strip {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: var(--ink-dim); text-align: right; }}
    .status-strip .dot {{ display: inline-block; width: 6px; height: 6px; border-radius: 50%; margin-right: 0.4rem; }}
    .status-strip .sep {{ margin: 0 0.7rem; color: var(--line); }}

    /* ── Tabs ────────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid var(--line); }}
    .stTabs [data-baseweb="tab"] {{
        height: 42px; background-color: transparent; border-radius: 0; color: var(--ink-dim);
        font-family: 'IBM Plex Sans', sans-serif; font-size: 0.9rem;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: transparent !important; color: var(--saffron) !important;
        border-bottom: 2px solid var(--saffron) !important;
    }}

    /* ── Misc components ────────────────────────────────────────────── */
    div[data-testid="stExpander"] {{ background-color: var(--surface); border: 1px solid var(--line); border-radius: 4px; }}
    div[data-testid="stDataFrame"] {{ border: 1px solid var(--line); border-radius: 4px; }}
    .stAlert {{ border-radius: 4px; font-family: 'IBM Plex Sans', sans-serif; }}
    div[data-testid="stAlertContainer"] {{ border: 1px solid var(--line); background-color: var(--surface) !important; }}
    div[data-testid="stAlertContainer"]:has(div[data-testid="stAlertContentInfo"]) {{ border-left: 3px solid var(--saffron); }}
    div[data-testid="stAlertContainer"]:has(div[data-testid="stAlertContentSuccess"]) {{ border-left: 3px solid var(--green); }}
    div[data-testid="stAlertContainer"]:has(div[data-testid="stAlertContentWarning"]) {{ border-left: 3px solid var(--saffron); }}
    div[data-testid="stAlertContainer"]:has(div[data-testid="stAlertContentError"]) {{ border-left: 3px solid var(--rust); }}
    div[data-testid="stAlertContainer"] p, div[data-testid="stAlertContainer"] li {{ color: var(--ink) !important; }}
    hr {{ border-color: var(--line); }}
    ::selection {{ background-color: rgba(201,122,43,0.35); }}
    a {{ color: var(--saffron); }}
    </style>
    """, unsafe_allow_html=True)

# --- Required File Verification ---
DB_PATH = './data/market_rhetoric.db'

SOURCE_COLORS = {
    'Mann Ki Baat': COLORS["saffron"],
    'ECB':          "#5A7A9A",   # muted slate-blue — legible against the navy surface, unlike --navy itself
    'Fed':          COLORS["green"],
}

REQUIRED_FILES = [
    DB_PATH,
    './data/processed/topic_distributions_combined.npy',
    './data/processed/topic_labels_combined.json'
]

missing_reqs = [f for f in REQUIRED_FILES if not os.path.exists(f)]
if missing_reqs:
    st.error(f"### ❌ CRITICAL: Missing Required Pipeline Files\n\nThe following files are missing. Please click **'🚀 Run Pipeline'** in the sidebar to generate them.\n\n" + "\n".join([f"- `{f}`" for f in missing_reqs]))
    # We don't st.stop() here because we want the user to be able to click the button in the sidebar

@st.cache_data
def load_db_stats():
    try:
        conn = get_db_connection(DB_PATH)
        speech_count = pd.read_sql_query("SELECT COUNT(*) as count FROM speeches", conn)['count'][0]
        market_count = pd.read_sql_query("SELECT COUNT(*) as count FROM market_data", conn)['count'][0]
        conn.close()
        return speech_count, market_count
    except Exception:
        return 0, 0

@st.cache_data
def load_source_breakdown():
    try:
        conn = get_db_connection(DB_PATH)
        df = pd.read_sql_query("SELECT source, COUNT(*) as count FROM speeches GROUP BY source", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

# --- Top bar: brand mark + Run Pipeline button ------------------------------
import base64

@st.cache_data
def _logo_b64(path="logo.png"):
    if os.path.exists(path):
        return base64.b64encode(open(path, "rb").read()).decode()
    return ""

_logo64 = _logo_b64()
_logo_img = f'<img src="data:image/png;base64,{_logo64}" />' if _logo64 else ""

_hdr_col1, _hdr_col2 = st.columns([5, 1])
with _hdr_col1:
    st.markdown(
        f"""<div class="topbar-brand">{_logo_img}
            <div><div class="topbar-brand-name">BaatSeBharat</div>
            <div class="topbar-brand-sub">Rhetoric &amp; Markets Intel.</div></div>
            </div>""",
        unsafe_allow_html=True,
    )

# Check if models exist and get timestamp
model_path = "./data/processed/topic_distributions_combined.npy"
models_exist = os.path.exists(model_path)
btn_label = "Run Pipeline Again" if models_exist else "Run Pipeline"

with _hdr_col2:
    with st.container(key="runpipeline"):
        if st.button(btn_label, use_container_width=True):
            with st.spinner("Executing End-to-End Prototype (MKB + ECB + Fed)..."):
                result = subprocess.run([sys.executable, "scripts/run_prototype.py"], capture_output=True, text=True)
                if result.returncode == 0:
                    st.success("Pipeline executed successfully.")
                    st.rerun()
                else:
                    st.error("Execution failed. Check data consistency.")
                    with open("logs/pipeline_error.log", "w") as f:
                        f.write(result.stderr)

STAGES = [
    "Executive Summary",
    "1. Data Ingestion",
    "2. NLP Intelligence",
    "3. Market Impact",
    "4. Regime Intelligence",
    "5. Company Analytics",
    "6. AI Predictions",
    "7. Global Influence Map",
    "8. Global Preview",
]
# Short labels for the horizontal nav pills only -- STAGES itself (the
# values every `elif stage == "..."` block below matches on) is unchanged,
# so this is purely a display-layer relabeling, not a routing change.
NAV_LABELS = [
    "Overview",
    "01 Ingestion",
    "02 NLP",
    "03 Impact",
    "04 Regime",
    "05 Company",
    "06 Predictions",
    "07 Global",
    "08 Preview",
]
if "active_stage" not in st.session_state:
    st.session_state.active_stage = STAGES[0]

# Horizontal top nav: a sticky flexbox row (st.container(horizontal=True))
# of content-sized buttons tracked via session_state, replacing the old
# vertical sidebar stepper. Each button keeps a stable per-index CSS hook
# (.st-key-nav_0 .. nav_8) for the saffron -> navy -> green accent
# sequence, and the active pill gets a solid underline (the horizontal
# equivalent of the old stepper's colored-dot progress marker).
with st.container(key="topnav", horizontal=True):
    for _i, (_label, _s) in enumerate(zip(NAV_LABELS, STAGES)):
        is_active = st.session_state.active_stage == _s
        with st.container(key=f"nav_{_i}"):
            if st.button(_label, key=f"navbtn_{_i}",
                         type="primary" if is_active else "secondary"):
                st.session_state.active_stage = _s
                st.rerun()
# position:fixed takes .st-key-topnav out of normal document flow, so this
# spacer reserves the vertical space it would otherwise occupy (see the
# .st-key-topnav CSS comment for why position:fixed is used instead of
# sticky here).
st.markdown('<div class="topnav-spacer"></div>', unsafe_allow_html=True)
stage = st.session_state.active_stage

# Status strip (relocated from the old sidebar footer) -- right-aligned,
# always visible rather than tucked behind an icon, matching the same
# "stay discoverable" treatment as the Run Pipeline button above.
_status_parts = []
if models_exist:
    mtime = os.path.getmtime(model_path)
    last_update = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
    _status_parts.append(f'<span class="dot" style="background:{COLORS["green"]}"></span>Updated {last_update}')
else:
    _status_parts.append(f'<span class="dot" style="background:{COLORS["rust"]}"></span>Pipeline not yet run')

if _PRED_OK and _llm_mode_available():
    _status_parts.append(f'<span class="dot" style="background:{COLORS["green"]}"></span>LLM mode active')
elif _PRED_OK:
    _status_parts.append(f'<span class="dot" style="background:{COLORS["saffron"]}"></span>AI: rule-based mode')
else:
    _status_parts.append(f'<span class="dot" style="background:{COLORS["rust"]}"></span>Prediction engine offline')

st.markdown(
    '<div class="status-strip">' + '<span class="sep">|</span>'.join(_status_parts) + "</div>",
    unsafe_allow_html=True,
)

# ===========================================================================
# CACHED DATA LOADERS for AI Predictions (Stage 6)
# ===========================================================================

@st.cache_data(ttl=1800, show_spinner=False)
def _load_avg_sentiment_from_db(db_path: str) -> float:
    """Compute average FinBERT sentiment from speech_market_impact table."""
    try:
        conn = get_db_connection(db_path)
        df = pd.read_sql_query(
            "SELECT AVG(abnormal_return) as avg_ret FROM speech_market_impact",
            conn
        )
        conn.close()
        val = float(df['avg_ret'].iloc[0] or 0.0)
        return float(np.clip(val * 5, -1, 1))
    except Exception:
        return 0.0

@st.cache_data(ttl=1800, show_spinner=False)
def _load_topic_strength_from_npy() -> float:
    """Load the dominant topic strength from the combined topic distribution."""
    try:
        npy_path = './data/processed/topic_distributions_combined.npy'
        if os.path.exists(npy_path):
            dists = np.load(npy_path)
            return float(dists.max(axis=1).mean())
    except Exception:
        pass
    return 0.5

@st.cache_data(ttl=1800, show_spinner=False)
def _load_regime_from_csv(ticker: str) -> str:
    """Load the latest regime label for a ticker from the processed/ CSV."""
    csv_path = f'./data/processed/regime_labels_{ticker}.csv'
    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for candidate in ('regime', 'regime_label'):
                if candidate in df.columns:
                    regime_col = candidate
                    break
            else:
                regime_col = df.columns[-1]
            latest = str(df[regime_col].iloc[-1])
            if 'stable' in latest.lower() or 'bull' in latest.lower():
                return 'Bull'
            elif 'volatile' in latest.lower() or 'bear' in latest.lower():
                return 'Bear'
            return 'Neutral'
    except Exception:
        pass
    return 'Neutral'

@st.cache_data(ttl=1800, show_spinner=False)
def _load_sector_avg_returns_from_db(db_path: str) -> pd.DataFrame:
    """Load average 5-day & 10-day returns per ticker from speech_market_impact."""
    try:
        conn = get_db_connection(db_path)
        df = pd.read_sql_query(
            """
            SELECT ticker as sector,
                   AVG(return_t5)  as return_5d,
                   AVG(return_t10) as return_10d
            FROM speech_market_impact
            GROUP BY ticker
            """,
            conn
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame(columns=['sector', 'return_5d', 'return_10d'])

@st.cache_data(ttl=1800, show_spinner=False)
def _load_regime_df_for_sectors() -> pd.DataFrame:
    """Produce a sector -> regime DataFrame from the processed regime CSVs."""
    sector_ticker_map = {
        'Banking':      '^NSEBANK',
        'IT':           '^CNXIT',
        'Pharma':       '^CNXPHARMA',
        'Auto':         '^CNXAUTO',
        'Energy':       '^CNXENERGY',
        'Broad Market': '^NSEI',
    }
    rows = []
    for sector, ticker in sector_ticker_map.items():
        regime = _load_regime_from_csv(ticker)
        rows.append({'sector': sector, 'regime': regime})
    return pd.DataFrame(rows)

@st.cache_data(ttl=1800, show_spinner=False)
def _cached_company_predictions(
    sentiment: float, topic_str: float, regime: str, hist_ret: float
) -> list:
    """Cache bulk company predictions (recomputed only when inputs change)."""
    if not _PRED_OK:
        return []
    return get_all_company_predictions(
        sentiment_score=sentiment,
        topic_strength=topic_str,
        regime_label=regime,
        historical_return=hist_ret,
        use_llm=False,
    )

@st.cache_data(ttl=1800, show_spinner=False)
def _cached_sector_predictions(
    sentiment: float, topic_str: float,
    sector_returns_json: str, regime_json: str
) -> list:
    """Cache sector predictions. DataFrames serialised to JSON for hashing."""
    if not _PRED_OK:
        return []
    import io
    sector_returns = pd.read_json(io.StringIO(sector_returns_json)) if sector_returns_json else None
    regime_df      = pd.read_json(io.StringIO(regime_json))         if regime_json else None
    return get_all_sector_predictions(
        sentiment_score=sentiment,
        topic_strength=topic_str,
        sector_returns=sector_returns,
        regime_df=regime_df,
    )

# --- Page Logic ---

if stage == "Executive Summary":
    st.markdown('<div class="stage-eyebrow">BaatSeBharat · Research Console</div>', unsafe_allow_html=True)
    st.title("Leadership Rhetoric Driven Market Intelligence")
    st.markdown("Quantifying the impact of leadership narrative on market volatility.")

    s_count, m_count = load_db_stats()

    @st.cache_data(ttl=3600, show_spinner=False)
    def _active_topic_count():
        path = "./data/processed/topic_labels_combined.json"
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return len(json.load(f))
        return 0

    @st.cache_data(ttl=3600, show_spinner=False)
    def _directional_backtest():
        # Real out-of-sample metric (src/models/causal_validation.py):
        # trains a per-topic return-direction bias on the first 70% of
        # speech events by date, tests on the remaining 30%. Replaces a
        # previously hardcoded, never-computed "Baseline ROC-AUC: 0.72
        # (+5%)" metric -- that number wasn't derived from anything.
        try:
            from models.causal_validation import CausalValidator
        except ImportError:
            from src.models.causal_validation import CausalValidator
        try:
            return CausalValidator(db_path=DB_PATH).backtest_directional_hit_rate()
        except Exception:
            return {}

    _bt = _directional_backtest()
    if _bt:
        hit_pct = _bt['hit_rate'] * 100
        _delta = hit_pct - 50
        _delta_color = COLORS["green"] if _delta > 0 else (COLORS["rust"] if _delta < 0 else COLORS["ink_dim"])
        hit_value, hit_sub, hit_tip = (
            f"{hit_pct:.1f}%",
            f'<span style="color:{_delta_color}">{_delta:+.1f}pp vs. random</span>',
            (
                f"Out-of-sample backtest: topic-to-return bias learned on speeches "
                f"before {_bt['cutoff_date']}, tested on {_bt['n_events']} events after. "
                "50% = no better than a coin flip -- reported honestly, not adjusted to look better."
            ),
        )
    else:
        hit_value, hit_sub, hit_tip = "—", None, None

    metric_row([
        ("Processed Speeches", f"{s_count:,}", None),
        ("Market Data Points", f"{m_count:,}", None),
        ("Active Topics", _active_topic_count() or "—", None),
        ("Directional Hit Rate (OOS)", hit_value, hit_sub, hit_tip),
    ])

    st.markdown("---")

    # Source breakdown
    breakdown = load_source_breakdown()
    if not breakdown.empty:
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.subheader("Speech Sources")
            for _, r in breakdown.iterrows():
                color = SOURCE_COLORS.get(r['source'], '#9AA3B5')
                st.markdown(
                    f"<span style='color:{color}'>●</span> **{r['source']}**: {r['count']} speeches",
                    unsafe_allow_html=True
                )
        with col_b:
            fig_pie = px.pie(
                breakdown, values='count', names='source',
                title="Speech Distribution by Source",
                color='source',
                color_discrete_map=SOURCE_COLORS,
                hole=0.55,
            )
            fig_pie.update_traces(textfont=dict(family="IBM Plex Mono", size=12), marker=dict(line=dict(color=COLORS["surface"], width=2)))
            apply_chart_theme(fig_pie, height=340)
            st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    st.subheader("Live Pipeline Feed")
    if s_count > 0:
        conn = get_db_connection(DB_PATH)
        recent = pd.read_sql_query(
            "SELECT date, source, speaker, title FROM speeches ORDER BY date DESC LIMIT 10", conn
        )
        st.dataframe(recent, use_container_width=True)
        conn.close()
    else:
        st.warning("No data found. Please click '🚀 Run Prototype Pipeline' from the sidebar.")

elif stage == "1. Data Ingestion":
    stage_header("01", "Data Ingestion & Storage")

    tab1, tab2 = st.tabs(["Speeches (Text)", "Market (Numerical)"])

    with tab1:
        @st.cache_data(ttl=1800, show_spinner=False)
        def _load_speech_index(db_path):
            # Metadata only (no full_text) -- this just populates a
            # dropdown. Previously this pulled full_text for every one of
            # the ~1300 speeches into memory on every page load/widget
            # interaction just to show a list of titles; full_text is only
            # needed for the single selected speech, fetched separately below.
            conn = get_db_connection(db_path)
            df = pd.read_sql_query(
                "SELECT id, date, source, speaker, title FROM speeches ORDER BY date DESC", conn
            )
            conn.close()
            return df

        @st.cache_data(ttl=1800, show_spinner=False)
        def _load_speech_text(db_path, speech_id):
            conn = get_db_connection(db_path)
            row = pd.read_sql_query(
                "SELECT full_text FROM speeches WHERE id = ?", conn, params=(speech_id,)
            )
            conn.close()
            return row['full_text'].iloc[0] if not row.empty else None

        df = _load_speech_index(DB_PATH)
        if not df.empty:
            # Filter by source
            sources = ['All'] + sorted(df['source'].dropna().unique().tolist())
            sel_source = st.selectbox("Filter by Source", sources)
            if sel_source != 'All':
                df = df[df['source'] == sel_source]

            # Vectorized display-name build (was a row-wise .apply()).
            display_names = (
                "[" + df['id'].astype(str) + "] " + df['source'] + " | "
                + df['date'].fillna('N/A') + " | " + df['title'].fillna('Untitled')
            )
            selected_option = st.selectbox("Select Speech to Preview", display_names.tolist())

            selected_id = int(selected_option.split(']')[0][1:])
            speech_row = df[df['id'] == selected_id].iloc[0]
            full_text = _load_speech_text(DB_PATH, selected_id)

            st.markdown(
                f"**Source:** {speech_row['source']} &nbsp;|&nbsp; "
                f"**Speaker:** {speech_row.get('speaker', 'N/A')} &nbsp;|&nbsp; "
                f"**Date:** {speech_row['date']}"
            )
            st.text_area("Transcript Preview", full_text or "(no text)", height=250)
        else:
            st.info("Database empty. Run the pipeline first.")

    with tab2:
        @st.cache_data(ttl=1800, show_spinner=False)
        def _load_market_close(db_path):
            conn = get_db_connection(db_path)
            df = pd.read_sql_query("SELECT date, ticker, close FROM market_data", conn)
            conn.close()
            return df

        df_m = _load_market_close(DB_PATH)
        if not df_m.empty:
            df_m['date'] = pd.to_datetime(df_m['date'])
            fig = px.line(
                df_m, x='date', y='close', color='ticker',
                title="Index Performance",
                color_discrete_sequence=CATEGORY_SEQUENCE,
            )
            apply_chart_theme(fig, height=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No market data. Run the pipeline first.")

elif stage == "2. NLP Intelligence":
    stage_header("02", "NLP & Topic Modeling")

    st.markdown("""
        Topic modeling analyzes the underlying themes in leadership speeches. 
        Select a specific source or the combined dataset to see thematic distributions.
    """)

    # Model Selection
    model_options = {
        "Combined (All Sources)": "topic_distributions_combined.npy",
        "Federal Reserve (Fed)": "topic_distributions_fed.npy",
        "European Central Bank (ECB)": "topic_distributions_ecb.npy",
        "Mann Ki Baat (MKB)": "topic_distributions_mann_ki_baat.npy"
    }
    
    selected_model_name = st.selectbox("Select Topic Model", list(model_options.keys()))
    current_topic_file = os.path.join("./data/processed", model_options[selected_model_name])

    labels_file = os.path.join(
        "./data/processed",
        f"topic_labels_{selected_model_name.lower().replace(' (all sources)', '').replace('federal reserve (fed)', 'fed').replace('european central bank (ecb)', 'ecb').replace('mann ki baat (mkb)', 'mann_ki_baat').replace(' ', '_')}.json"
    )
    labels_data = {}
    if os.path.exists(labels_file):
        with open(labels_file, 'r', encoding='utf-8') as f:
            labels_data = json.load(f)

    def topic_label(i):
        info = labels_data.get(str(i))
        return info['label'] if info else f"Topic {i+1}"

    if os.path.exists(current_topic_file):
        topics = np.load(current_topic_file)
        topic_names = [topic_label(i) for i in range(topics.shape[1])]

        st.subheader(f"Topic distribution: {selected_model_name}")
        st.caption(f"TF-IDF + NMF topic model for {selected_model_name}, deterministically labeled from top keywords.")

        # Show distribution for first speech in this set
        fig = px.bar(
            x=topic_names,
            y=topics[0],
            labels={'x': 'Topic', 'y': 'Probability'},
            title=f"Dominant Rhetoric Components ({selected_model_name})",
        )
        fig.update_traces(marker_color=COLORS["saffron"])
        fig.update_layout(xaxis_tickangle=-30)
        apply_chart_theme(fig, height=420)
        st.plotly_chart(fig, use_container_width=True)

        # Heatmap: topic distributions per speech (first 30)
        if topics.shape[0] > 1:
            st.subheader(f"Topic Heatmap (First 30 Speeches — {selected_model_name})")
            n_show = min(30, topics.shape[0])
            heat_df = pd.DataFrame(
                topics[:n_show],
                columns=topic_names
            )
            fig_heat = px.imshow(
                heat_df.T,
                aspect="auto",
                color_continuous_scale=SEQUENTIAL_SCALE,
                title="Topic Probability Heatmap",
                labels={'y': 'Topic'},
            )
            apply_chart_theme(fig_heat, height=420)
            st.plotly_chart(fig_heat, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Topics & Top Keywords")
            if labels_data:
                for i in range(topics.shape[1]):
                    info = labels_data.get(str(i))
                    if info:
                        keywords = ", ".join(info['keywords'][:5])
                        st.write(f"**{info['label']}:** {keywords}")
            else:
                st.warning(f"No label file found at `{labels_file}`. Run the pipeline to generate topic labels.")
        with col2:
            st.markdown("### Model Insight")
            st.info(f"Model trained on {topics.shape[0]} documents with {topics.shape[1]} latent topics.")
    else:
        st.warning(f"No results found for {selected_model_name}.")
        st.info("💡 Use the **Run Pipeline** button in the sidebar to generate results.")
        st.warning("No topic distributions found. Run the pipeline first.")

elif stage == "3. Market Impact":
    stage_header("03", "Speech Impact on Markets")

    @st.cache_data(ttl=1800, show_spinner=False)
    def _load_stage3_data(db_path):
        # Previously re-ran on every widget interaction (ticker selectbox,
        # source multiselect trigger a full script rerun) with no caching.
        conn = get_db_connection(db_path)
        impact = pd.read_sql_query('''
            SELECT
                s.date, s.source, s.speaker, s.title,
                i.ticker, i.return_t1, i.return_t5, i.return_t10, i.abnormal_return
            FROM speech_market_impact i
            JOIN speeches s ON i.speech_id = s.id
            WHERE s.date IS NOT NULL
            ORDER BY s.date DESC
        ''', conn)
        market = pd.read_sql_query(
            "SELECT date, ticker, close FROM market_data ORDER BY date", conn
        )
        conn.close()
        return impact, market

    impact_df, market_df = _load_stage3_data(DB_PATH)

    if impact_df.empty or market_df.empty:
        st.warning(
            "No impact data yet. Click '🚀 Run Prototype Pipeline' to populate the database."
        )
    else:
        impact_df['date'] = pd.to_datetime(impact_df['date'])
        market_df['date'] = pd.to_datetime(market_df['date'])

        # --- Market chart with speech event overlays ---
        st.subheader("Market Performance with Speech Events")
        tickers = market_df['ticker'].unique().tolist()
        sel_ticker = st.selectbox("Select Ticker", tickers)

        ticker_market = market_df[market_df['ticker'] == sel_ticker]
        ticker_impact = impact_df[impact_df['ticker'] == sel_ticker].drop_duplicates('date')

        # --- Overall market signal metric at the top ---
        overall_avg = ticker_impact['return_t5'].mean() if not ticker_impact.empty else 0.0
        if overall_avg > 0.002:
            ov_signal, ov_emoji, ov_color = "Bullish", "🟢", "#2F6F4E"
        elif overall_avg < -0.002:
            ov_signal, ov_emoji, ov_color = "Bearish", "🔴", "#A6503A"
        else:
            ov_signal, ov_emoji, ov_color = "Neutral", "⚪", "#9AA3B5"

        sig_c1, sig_c2, sig_c3 = st.columns(3)
        with sig_c1:
            st.markdown(
                f"""
                <div style='background:#131B2C;border-radius:10px;padding:14px 18px;
                            border:1px solid {ov_color};text-align:center'>
                  <div style='color:#9AA3B5;font-size:0.80rem'>Overall Market Signal</div>
                  <div style='font-size:1.6rem;font-weight:700;color:{ov_color}'>
                    {ov_emoji} {ov_signal}
                  </div>
                  <div style='font-size:0.72rem;color:#475569'>Based on 5-Day Fwd Returns</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with sig_c2:
            n_bullish = (ticker_impact['return_t5'] > 0.002).sum()
            n_bearish = (ticker_impact['return_t5'] < -0.002).sum()
            n_neutral = len(ticker_impact) - n_bullish - n_bearish
            st.markdown(
                f"""
                <div style='background:#131B2C;border-radius:10px;padding:14px 18px;
                            border:1px solid #26324A;text-align:center'>
                  <div style='color:#9AA3B5;font-size:0.80rem'>Signal Breakdown</div>
                  <div style='font-size:0.92rem;margin-top:4px'>
                    <span style='color:#2F6F4E'>🟢 {n_bullish} Bull</span>&nbsp;
                    <span style='color:#9AA3B5'>⚪ {n_neutral} Neutral</span>&nbsp;
                    <span style='color:#A6503A'>🔴 {n_bearish} Bear</span>
                  </div>
                  <div style='font-size:0.72rem;color:#475569'>Across {len(ticker_impact)} events</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with sig_c3:
            avg_conf = abs(overall_avg) * 2000
            conf_disp = min(100, max(30, avg_conf))
            st.markdown(
                f"""
                <div style='background:#131B2C;border-radius:10px;padding:14px 18px;
                            border:1px solid #26324A;text-align:center'>
                  <div style='color:#9AA3B5;font-size:0.80rem'>Signal Confidence</div>
                  <div style='font-size:1.4rem;font-weight:700;color:{ov_color}'>{conf_disp:.0f}%</div>
                  <div style='font-size:0.72rem;color:#475569'>Avg Abnormal: {overall_avg*100:+.3f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ticker_market['date'], y=ticker_market['close'],
            name=sel_ticker, mode='lines',
            line=dict(color=COLORS["ink"], width=1.8)
        ))

        # Add vertical markers per source. Speech events can number in the
        # thousands for a heavily-scraped ticker; calling fig.add_vline()
        # once per event recomputes the whole figure layout on every call
        # (effectively O(n^2) in Plotly), which is what made this page hang
        # for minutes. Drawing all of a source's vertical lines as ONE
        # Scatter trace (x/y interleaved with None separators) is O(n) and
        # renders in milliseconds instead.
        y_lo = float(ticker_market['close'].min()) if not ticker_market.empty else 0.0
        y_hi = float(ticker_market['close'].max()) if not ticker_market.empty else 1.0
        for src, color in SOURCE_COLORS.items():
            src_dates = ticker_impact[ticker_impact['source'] == src]['date'].unique()
            if len(src_dates):
                line_x, line_y = [], []
                for d in src_dates:
                    line_x.extend([d, d, None])
                    line_y.extend([y_lo, y_hi, None])
                fig.add_trace(go.Scatter(
                    x=line_x, y=line_y, mode='lines',
                    line=dict(color=color, width=1, dash='dot'),
                    opacity=0.22, showlegend=False, hoverinfo='skip',
                ))
            # Invisible scatter just for legend
            if len(src_dates):
                # Ensure we have a date-indexed series for nearest-neighbor lookup
                market_series = ticker_market.set_index('date')['close']
                fig.add_trace(go.Scatter(
                    x=src_dates,
                    y=market_series.reindex(pd.DatetimeIndex(src_dates), method='nearest').values
                    if not ticker_market.empty else [None]*len(src_dates),
                    mode='markers',
                    marker=dict(color=color, size=8, symbol='triangle-down'),
                    name=src,
                    hovertemplate=(
                        "<b>%{x|%Y-%m-%d}</b><br>"
                        f"Source: {src}<br>"
                        "Price: %{y:.2f}<extra></extra>"
                    )
                ))

        apply_chart_theme(fig, height=450)
        fig.update_layout(
            title=f"{sel_ticker} Price with Speech Events",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Impact summary table with Signal column ---
        st.subheader("Speech Event Forward Returns")
        filter_src = st.multiselect(
            "Filter by Source", options=list(SOURCE_COLORS.keys()),
            default=list(SOURCE_COLORS.keys())
        )
        disp_df = impact_df[
            (impact_df['ticker'] == sel_ticker) & (impact_df['source'].isin(filter_src))
        ][['date', 'source', 'speaker', 'title', 'return_t1', 'return_t5', 'return_t10', 'abnormal_return']].copy()

        # Add Bullish/Bearish/Neutral signal column based on 5-day return
        def _return_signal(v):
            if pd.isna(v):
                return "—"
            if v > 0.002:
                return "🟢 Bullish"
            elif v < -0.002:
                return "🔴 Bearish"
            else:
                return "⚪ Neutral"

        disp_df['Signal'] = disp_df['return_t5'].apply(_return_signal)

        for col in ['return_t1', 'return_t5', 'return_t10', 'abnormal_return']:
            disp_df[col] = disp_df[col].map(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—")

        disp_df.rename(columns={
            'return_t1': '1-Day Fwd Ret',
            'return_t5': '5-Day Fwd Ret',
            'return_t10': '10-Day Fwd Ret',
            'abnormal_return': 'Abnormal Ret'
        }, inplace=True)

        # Reorder to put Signal first after date/source
        cols_order = ['date', 'source', 'Signal', 'speaker', 'title',
                      '1-Day Fwd Ret', '5-Day Fwd Ret', '10-Day Fwd Ret', 'Abnormal Ret']
        disp_df = disp_df[[c for c in cols_order if c in disp_df.columns]]

        st.dataframe(disp_df, use_container_width=True)

        # --- Average abnormal returns by source ---
        st.subheader("Average 5-Day Abnormal Return by Source")
        avg_df = impact_df[
            (impact_df['ticker'] == sel_ticker) & impact_df['abnormal_return'].notna()
        ].groupby('source')['abnormal_return'].mean().reset_index()
        avg_df.columns = ['Source', 'Avg Abnormal 5D Return']

        # Add signal label to bar chart
        avg_df['Signal'] = avg_df['Avg Abnormal 5D Return'].apply(
            lambda v: '🟢 Bullish' if v > 0.002 else ('🔴 Bearish' if v < -0.002 else '⚪ Neutral')
        )

        fig_bar = px.bar(
            avg_df, x='Source', y='Avg Abnormal 5D Return',
            color='Source', color_discrete_map=SOURCE_COLORS,
            title=f"Average 5-Day Abnormal Return by Source ({sel_ticker})",
            text='Signal'
        )
        fig_bar.update_traces(textposition='outside')
        fig_bar.add_hline(y=0, line_dash="dash", line_color=COLORS["line"])
        apply_chart_theme(fig_bar, height=380)
        st.plotly_chart(fig_bar, use_container_width=True)

        st.info(
            "💡 **Interpretation:** A positive abnormal return means speeches from this source "
            "tend to coincide with above-average 5-day forward returns."
        )

        # --- Topic-Market Alignment ---
        st.markdown("---")
        st.subheader("🎯 Topic-Market Correlation Analysis")
        st.markdown("""
            This section aligns leadership rhetoric (topics) with market performance to identify 
            which themes drive the highest returns.
        """)
        
        # Join impact with topic distributions (for the 'Combined' model).
        # topic_distributions stores one row PER TOPIC per speech (a full
        # probability vector, not just the dominant topic), so a plain JOIN
        # attaches the SAME market-impact row to every topic_id for a given
        # speech. An unweighted AVG() then averages the identical set of
        # returns for every topic, producing identical bars regardless of
        # topic. Weighting by td.probability makes topics with a stronger
        # presence in a speech contribute more to that topic's average,
        # which is what actually differentiates topics.
        @st.cache_data(ttl=1800, show_spinner=False)
        def _load_topic_impact(db_path, ticker):
            conn_topic = get_db_connection(db_path)
            topic_impact_query = '''
                SELECT
                    td.topic_id,
                    SUM(td.probability * i.return_t5) / NULLIF(SUM(td.probability), 0) as avg_ret_t5,
                    SUM(td.probability * i.abnormal_return) / NULLIF(SUM(td.probability), 0) as avg_abnormal,
                    COUNT(DISTINCT i.id) as speech_count
                FROM topic_distributions td
                JOIN speech_market_impact i ON td.speech_id = i.speech_id
                WHERE td.model_name = 'Combined' AND i.ticker = ?
                GROUP BY td.topic_id
                ORDER BY avg_abnormal DESC
            '''
            result = pd.read_sql_query(topic_impact_query, conn_topic, params=(ticker,))

            if result.empty:
                # Fallback: Overall average across all tickers
                topic_impact_query_fallback = '''
                    SELECT
                        td.topic_id,
                        SUM(td.probability * i.return_t5) / NULLIF(SUM(td.probability), 0) as avg_ret_t5,
                        SUM(td.probability * i.abnormal_return) / NULLIF(SUM(td.probability), 0) as avg_abnormal,
                        COUNT(DISTINCT i.id) as speech_count
                    FROM topic_distributions td
                    JOIN speech_market_impact i ON td.speech_id = i.speech_id
                    WHERE td.model_name = 'Combined'
                    GROUP BY td.topic_id
                    ORDER BY avg_abnormal DESC
                '''
                result = pd.read_sql_query(topic_impact_query_fallback, conn_topic)
            conn_topic.close()
            return result

        topic_impact_df = _load_topic_impact(DB_PATH, sel_ticker)

        if not topic_impact_df.empty:
            _labels_path = "./data/processed/topic_labels_combined.json"
            _topic_label_map = {}
            if os.path.exists(_labels_path):
                with open(_labels_path, 'r', encoding='utf-8') as f:
                    _topic_label_map = {int(k): v['label'] for k, v in json.load(f).items()}
            topic_impact_df['topic_label'] = topic_impact_df['topic_id'].apply(
                lambda x: _topic_label_map.get(x, f"Topic {x+1}")
            )
            # Add signal column for topics too
            topic_impact_df['topic_signal'] = topic_impact_df['avg_abnormal'].apply(
                lambda v: '🟢 Bullish' if v > 0 else ('🔴 Bearish' if v < 0 else '⚪ Neutral')
            )
            fig_topic = px.bar(
                topic_impact_df,
                x='topic_label',
                y='avg_abnormal',
                color='avg_abnormal',
                color_continuous_scale=DIVERGING_SCALE,
                color_continuous_midpoint=0,
                title=f"Avg 5D Abnormal Return by Dominant Topic ({sel_ticker})",
                labels={'avg_abnormal': 'Avg Abnormal Return (5D)', 'topic_label': 'Topic'},
                hover_data=['speech_count', 'topic_signal'],
                text='topic_signal'
            )
            fig_topic.update_traces(textposition='outside')
            fig_topic.add_hline(y=0, line_dash="dash", line_color=COLORS["line"])
            fig_topic.update_layout(xaxis_tickangle=-30)
            apply_chart_theme(fig_topic, height=440)
            st.plotly_chart(fig_topic, use_container_width=True)
            
            best_topic = topic_impact_df.iloc[0]
            best_signal = best_topic['topic_signal']
            st.success(
                f"**Alpha Driver:** **{best_topic['topic_label']}** is the most impactful theme for {sel_ticker} "
                f"— Signal: **{best_signal}** · Avg 5-Day Abnormal Return: "
                f"**{best_topic['avg_abnormal']*100:.2f}%**"
            )
        else:
            st.warning("No topic-alignment data available. Run the pipeline first.")

elif stage == "4. Regime Intelligence":
    stage_header("04", "Market Regime Intelligence (HMM)", "Quantifying structural market shifts using Hidden Markov Models.")

    @st.cache_data(ttl=1800, show_spinner=False)
    def _load_regime_and_market(db_path):
        # Previously these two full-table reads ran on every widget
        # interaction (not just page load), because Streamlit reruns the
        # whole script on any interaction and this query had no caching.
        conn = get_db_connection(db_path)
        regimes_df = pd.read_sql_query(
            "SELECT date, sector, regime, confidence FROM regime_classifications ORDER BY date", conn
        )
        market_df = pd.read_sql_query("SELECT date, ticker, close FROM market_data ORDER BY date", conn)
        conn.close()
        regimes_df['date'] = pd.to_datetime(regimes_df['date'])
        market_df['date'] = pd.to_datetime(market_df['date'])
        return regimes_df, market_df

    regimes, market = _load_regime_and_market(DB_PATH)

    if regimes.empty or market.empty:
        st.warning("No regime data found. Run the pipeline first.")
    else:
        tickers = market['ticker'].unique()
        sel_ticker = st.selectbox("Select Ticker for Regime Timeline", tickers)

        t_market = market[market['ticker'] == sel_ticker]
        t_regimes = regimes[regimes['sector'] == sel_ticker].sort_values('date')

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t_market['date'], y=t_market['close'], name="Price", line=dict(color=COLORS["ink"], width=1.5)))

        colors = {'Stable': 'rgba(47, 111, 78, 0.28)', 'Transitional': 'rgba(201, 122, 43, 0.24)', 'Volatile': 'rgba(166, 80, 58, 0.30)'}

        if not t_regimes.empty:
            # Vectorized run-length encoding of consecutive same-regime
            # segments (no row-by-row .iloc access). Shapes are built as
            # plain dicts and assigned to the figure in one shot via
            # update_layout(shapes=...) instead of calling add_vrect() in a
            # loop -- add_vrect/add_shape revalidate the ENTIRE shapes list
            # on every call, which is O(n^2) in the number of calls (the
            # same class of bug that made Stage 3 hang; here it's smaller
            # in scale but still the single biggest cost in profiling this
            # page, ~60% of total render time even with only ~36 segments).
            regime_vals = t_regimes['regime'].values
            dates_vals = t_regimes['date'].values
            segment_id = np.cumsum(np.concatenate(([0], regime_vals[1:] != regime_vals[:-1])))
            seg_df = pd.DataFrame({'segment': segment_id, 'regime': regime_vals, 'date': dates_vals})
            bounds = seg_df.groupby('segment').agg(regime=('regime', 'first'), x0=('date', 'first'), x1=('date', 'last'))

            shapes = [
                dict(
                    type='rect', xref='x', yref='paper',
                    x0=row.x0, x1=row.x1, y0=0, y1=1,
                    fillcolor=colors.get(row.regime, 'gray'), opacity=0.5,
                    line_width=0, layer='below',
                )
                for row in bounds.itertuples()
            ]
            fig.update_layout(shapes=shapes)

        apply_chart_theme(fig, height=550)
        fig.update_layout(title=f"{sel_ticker} Regime Timeline (Green=Stable, Saffron=Transitional, Rust=Volatile)")
        st.plotly_chart(fig, use_container_width=True)

        if t_regimes.empty:
            st.info(
                f"No HMM regime classifications computed yet for **{sel_ticker}**. "
                "Regime data currently only covers a subset of tickers "
                f"({', '.join(sorted(regimes['sector'].unique()))}) — run the regime "
                "modeling step (src/models/market_modeling.py) for the rest."
            )

elif stage == "5. Company Analytics":
    stage_header("05", "Company Specific Returns vs. Rhetoric", "Analyzing how leadership topics impact individual company performance.")

    # Map each company to its NSE ticker so the heatmap is scoped to that
    # company's own speech-market-impact events, not the entire corpus.
    COMPANY_TICKER_MAP = {
        "HDFC Bank": "HDFCBANK.NS",
        "Reliance Industries": "RELIANCE.NS",
        "Infosys": "INFY.NS",
        "TCS": "TCS.NS",
        "ICICI Bank": "ICICIBANK.NS",
    }
    company = st.selectbox("Select Company", list(COMPANY_TICKER_MAP.keys()))
    company_ticker = COMPANY_TICKER_MAP[company]

    @st.cache_data(ttl=1800, show_spinner=False)
    def _load_company_topics(db_path, ticker):
        # Pull each speech's topic probabilities together with THIS
        # company's own forward return (i.return_t5). Raw topic probability
        # has no company dimension at all (the same speech, same topic
        # weights, feed every ticker) -- filtering by ticker alone left the
        # heatmap pixel-identical across companies. Weighting probability
        # by the company's own return is what actually varies per company.
        conn = get_db_connection(db_path)
        result = pd.read_sql_query(
            """
            SELECT s.date, td.topic_id, td.probability, i.return_t5
            FROM topic_distributions td
            JOIN speeches s ON td.speech_id = s.id
            JOIN speech_market_impact i ON i.speech_id = s.id
            WHERE td.model_name = 'Combined' AND i.ticker = ?
            """,
            conn, params=(ticker,)
        )
        conn.close()
        return result

    topics_df = _load_company_topics(DB_PATH, company_ticker)

    if topics_df.empty:
        st.warning(f"No topic-impact data found for {company}. Run the pipeline first.")
    else:
        topics_df['date'] = pd.to_datetime(topics_df['date'])
        topics_df['weighted_strength'] = topics_df['probability'] * topics_df['return_t5']

        # Topic labels (deterministic, from src/models/topic_modeling.py),
        # so the heatmap shows real topic names instead of raw topic_id.
        labels_path = "./data/processed/topic_labels_combined.json"
        topic_label_map = {}
        if os.path.exists(labels_path):
            with open(labels_path, 'r', encoding='utf-8') as f:
                topic_label_map = {int(k): v['label'] for k, v in json.load(f).items()}

        st.subheader(f"{company} Topic Impact Heatmap")

        # Aggregate to monthly buckets: with 800+ distinct speech dates per
        # ticker, a per-date heatmap has too many columns to read (or even
        # show axis ticks for). Monthly mean keeps the signal real while
        # making the chart legible.
        topics_df['month'] = topics_df['date'].dt.to_period('M').dt.to_timestamp()
        pivot_topics = (
            topics_df.groupby(['month', 'topic_id'])['weighted_strength']
            .mean().unstack().fillna(0)
        )
        topic_names = [topic_label_map.get(i, f"Topic {i}") for i in pivot_topics.columns]

        zmax = float(np.abs(pivot_topics.values).max()) or 1.0
        fig_heat = go.Figure(data=go.Heatmap(
            z=pivot_topics.values.T,
            x=pivot_topics.index,
            y=topic_names,
            colorscale=DIVERGING_SCALE,
            zmid=0, zmin=-zmax, zmax=zmax,
            colorbar=dict(title="Avg 5D Fwd<br>Return × Topic<br>Probability", tickfont=dict(family="IBM Plex Mono")),
            hovertemplate="%{y}<br>%{x|%Y-%m}<br>Impact: %{z:.4f}<extra></extra>",
        ))
        apply_chart_theme(fig_heat, height=450)
        fig_heat.update_layout(
            title=f"Leadership Topic Impact Over Time vs {company} (Monthly, 5D Fwd Return-Weighted)",
            xaxis_title="Month",
            yaxis_title="Topic",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.info(
            f"💡 Each cell is the average of (topic probability × {company}'s 5-day forward return) "
            f"for speeches in that month. Green = that theme historically preceded {company} gains; "
            f"red = historically preceded {company} declines."
        )

elif stage == "6. AI Predictions":
    stage_header("06", "AI Market Predictions", "Company & sector forecasts driven by BaatSeBharat NLP signals.")

    if not _PRED_OK:
        st.error(f"Prediction engine could not load: `{_pred_err_msg}`")
        st.info("Ensure `src/prediction_engine.py` is present and `yfinance` is installed.")
        st.stop()

    # ── Load live signal values from DB / processed files ──────────────────
    with st.spinner("Loading cached signal data…"):
        _live_sentiment  = _load_avg_sentiment_from_db(DB_PATH)
        _live_topic_str  = _load_topic_strength_from_npy()
        _live_hist_ret   = 0.0
        _live_regime_raw = _load_regime_from_csv('^NSEI')
        _sector_returns  = _load_sector_avg_returns_from_db(DB_PATH)
        _regime_df       = _load_regime_df_for_sectors()

        if not _sector_returns.empty:
            _live_hist_ret = float(_sector_returns['return_5d'].mean())

    # ── Override controls ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("⚙️ Signal Overrides")
    st.caption(
        "Defaults are read live from the DB and processed files. "
        "Adjust to run what-if scenarios."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        user_sent = st.slider(
            "FinBERT Sentiment", -1.0, 1.0,
            float(round(_live_sentiment, 3)), 0.01,
            help="Derived from average abnormal returns in speech_market_impact"
        )
    with c2:
        user_topic = st.slider(
            "Rhetoric Signal", 0.0, 1.0,
            float(round(_live_topic_str, 3)), 0.01,
            help="Avg dominant rhetoric weight from topic_distributions_combined.npy"
        )
    with c3:
        regime_options = ["Bull", "Neutral", "Bear", "Stable", "Transitional", "Volatile"]
        user_regime = st.selectbox(
            "Market Regime",
            regime_options,
            index=regime_options.index(_live_regime_raw)
                  if _live_regime_raw in regime_options else 1,
            help="Latest regime from ^NSEI regime_labels CSV"
        )
    with c4:
        use_llm = st.checkbox(
            "LLM Mode",
            value=_llm_mode_available(),
            disabled=not _llm_mode_available(),
            help="Needs OPENAI_API_KEY or GOOGLE_API_KEY in .env"
        )

    st.markdown("---")

    # ── Tabs ───────────────────────────────────────────────────────────────
    tab_co, tab_sec = st.tabs(["🏢 Company Predictions", "📦 Sector Predictions"])

    # ────────────────── COMPANY TAB ────────────────────────────────────────
    with tab_co:
        st.subheader("🏢 Company-Level Predictions")
        st.caption(
            "Signals: FinBERT sentiment · topic strength · regime · yfinance price momentum"
        )

        sel_co = st.selectbox(
            "Select Company for Detail View",
            list(COMPANY_UNIVERSE.keys()),
            key="ai_co_select"
        )

        # Per-company regime from its actual regime CSV
        co_ticker   = COMPANY_UNIVERSE.get(sel_co, '')
        co_regime   = _load_regime_from_csv(co_ticker) if co_ticker else user_regime
        co_hist_ret = _live_hist_ret
        if not _sector_returns.empty and co_ticker:
            row = _sector_returns[_sector_returns['sector'] == co_ticker]
            if not row.empty:
                co_hist_ret = float(row['return_5d'].iloc[0])

        with st.spinner(f"Computing prediction for {sel_co}…"):
            co_pred = get_company_prediction(
                sel_co,
                sentiment_score=user_sent,
                topic_strength=user_topic,
                regime_label=co_regime,
                historical_return=co_hist_ret,
                use_llm=use_llm,
            )

        # Signal banner
        sig_col = {"Bullish": "#2F6F4E", "Bearish": "#A6503A", "Neutral": "#9AA3B5"}[
            co_pred["signal"]
        ]
        st.markdown(
            f'<div style="background:#131B2C;border-radius:6px;padding:18px 22px;border:1px solid {sig_col};border-left:3px solid {sig_col};margin-bottom:14px">'
            f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{sig_col};margin-right:10px"></span>'
            f'<span style="font-family:\'Fraunces\',serif;font-size:1.7rem;font-weight:500;color:{sig_col}">{co_pred["signal"].upper()}</span>'
            f'<span style="color:#9AA3B5;font-size:0.85rem;margin-left:18px;font-family:\'IBM Plex Sans\',sans-serif">'
            f'{sel_co} &middot; <span style="font-family:\'IBM Plex Mono\',monospace">{co_pred.get("ticker","N/A")}</span>'
            f' &middot; Confidence: <b style="color:{sig_col};font-family:\'IBM Plex Mono\',monospace">{co_pred["confidence"]:.0f}%</b>'
            f' &middot; Mode: {co_pred["mode"].upper()}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Current price metric
        if co_pred.get('current_price'):
            st.metric(
                f"Current Market Price ({co_pred['ticker']})",
                f"₹{co_pred['current_price']:,.2f}"
            )

        # Horizon cards
        st.markdown("#### 📅 Forecast Horizons")
        hc1, hc2, hc3 = st.columns(3)
        for col_widget, h in zip([hc1, hc2, hc3], [1, 5, 10]):
            fc  = co_pred["predictions"].get(h, {})
            ret = fc.get("return_pct", 0.0)
            rlo = fc.get("return_low",  0.0)
            rhi = fc.get("return_high", 0.0)
            rc  = "#2F6F4E" if ret > 0 else ("#A6503A" if ret < 0 else "#9AA3B5")
            pm  = fc.get("price_mid")
            pl  = fc.get("price_low")
            ph  = fc.get("price_high")
            price_html = (
                f"<div style='font-size:0.78rem;color:#475569;margin-top:4px'>"
                f"₹{pl:,.0f} – ₹{ph:,.0f}</div>" if pm else ""
            )
            with col_widget:
                st.markdown(
                    f"""
                    <div style='background:#131B2C;border-radius:10px;padding:14px;
                                border:1px solid #26324A;text-align:center'>
                      <div style='color:#9AA3B5;font-size:0.82rem'>{fc.get('label','')}</div>
                      <div style='font-size:1.55rem;font-weight:700;color:{rc}'>{ret:+.2f}%</div>
                      <div style='font-size:0.72rem;color:#475569'>
                        Range: {rlo:+.1f}% → {rhi:+.1f}%
                      </div>
                      {price_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Input signals expander
        with st.expander("🔍 Signal Inputs Used", expanded=False):
            inp = co_pred.get("inputs", {})
            _llm_strength = inp.get("llm_strength")
            _llm_sentiment = inp.get("llm_sentiment")
            st.table(pd.DataFrame([
                {"Signal": "FinBERT Sentiment",    "Value": f"{inp.get('sentiment', 0):+.4f}"},
                {"Signal": "Rhetoric Signal",      "Value": f"{inp.get('rhetoric_signal', inp.get('topic_strength', 0)):.4f}"},
                {"Signal": "Market Regime",        "Value": inp.get('regime', 'N/A')},
                {"Signal": "5-Day Price Momentum", "Value": f"{inp.get('momentum_5d_pct', 0):+.2f}%"},
                {"Signal": "Hist. Avg Return (5D)","Value": f"{inp.get('historical_return_pct', 0):+.4f}%"},
                {"Signal": "Groq Topic Strength",  "Value": f"{_llm_strength:.4f}" if _llm_strength is not None else "N/A — not yet classified"},
                {"Signal": "Groq Sentiment",       "Value": f"{_llm_sentiment:+.4f}" if _llm_sentiment is not None else "N/A — not yet classified"},
            ]))

        if use_llm and co_pred.get("llm_decision"):
            with st.expander("🤖 LLM Reasoning", expanded=False):
                st.text(co_pred["llm_decision"])

        st.markdown("---")

        # ── All Companies Overview ──────────────────────────────────────────
        st.subheader("📋 All Companies — 5-Day Forecast")
        st.caption("Cached for 30 min. Adjust any slider above to invalidate cache.")

        @st.cache_data(ttl=1800, show_spinner=False)
        def _companies_missing_market_data(db_path):
            conn = get_db_connection(db_path)
            tickers_with_data = set(
                pd.read_sql_query("SELECT DISTINCT ticker FROM market_data", conn)['ticker']
            )
            conn.close()
            return [c for c, t in COMPANY_UNIVERSE.items() if t not in tickers_with_data]

        _missing = _companies_missing_market_data(DB_PATH)
        if _missing:
            st.caption(
                f"⚠️ No market_data downloaded yet for: {', '.join(_missing)}. "
                "Their predictions fall back to profile-only baselines (no live "
                "momentum or historical-return signal) until yfinance data is fetched for them."
            )

        with st.spinner("Loading cached bulk predictions…"):
            all_co_preds = _cached_company_predictions(
                round(user_sent, 3), round(user_topic, 3),
                user_regime, round(_live_hist_ret, 6)
            )

        if all_co_preds:
            pr = []
            for p in all_co_preds:
                pr.append({
                    "Company":      p["company"],
                    "Signal":       p["signal"],
                    "Confidence":   p["confidence"],
                    "Score":        p["score"],
                    "1D %":         p["predictions"].get(1,  {}).get("return_pct", 0),
                    "5D %":         p["predictions"].get(5,  {}).get("return_pct", 0),
                    "10D %":        p["predictions"].get(10, {}).get("return_pct", 0),
                })
            pr_df = pd.DataFrame(pr).sort_values("Score", ascending=False)
            sig_clr = {"Bullish": "#2F6F4E", "Neutral": "#9AA3B5", "Bearish": "#A6503A"}

            fig_bar = go.Figure(go.Bar(
                x=pr_df["Company"],
                y=pr_df["5D %"],
                marker_color=[sig_clr.get(s, "#9AA3B5") for s in pr_df["Signal"]],
                text=[f"{v:+.2f}%" for v in pr_df["5D %"]],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>5D: %{y:+.2f}%<extra></extra>",
            ))
            fig_bar.add_hline(y=0, line_dash="dot", line_color=COLORS["line"])
            apply_chart_theme(fig_bar, height=370)
            fig_bar.update_layout(
                title="All Companies — 5-Day Return Forecast",
                xaxis_tickangle=-35,
                yaxis_title="Forecast Return (%)",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            fig_sc = px.scatter(
                pr_df, x="Score", y="Confidence",
                color="Signal", size="Confidence",
                hover_name="Company",
                color_discrete_map=sig_clr,
                title="Signal Score vs Prediction Confidence",
            )
            apply_chart_theme(fig_sc, height=330)
            st.plotly_chart(fig_sc, use_container_width=True)

            with st.expander("📋 Full Table"):
                tbl = pr_df.copy()
                tbl["Signal"] = tbl["Signal"].map(
                    {"Bullish": "🟢 Bullish", "Neutral": "⚪ Neutral", "Bearish": "🔴 Bearish"}
                )
                for c in ["1D %", "5D %", "10D %"]:
                    tbl[c] = tbl[c].map(lambda v: f"{v:+.2f}%")
                tbl["Confidence"] = tbl["Confidence"].map(lambda v: f"{v:.0f}%")
                st.dataframe(tbl, use_container_width=True, hide_index=True)

    # ────────────────── SECTOR TAB ─────────────────────────────────────────
    with tab_sec:
        st.subheader("📦 Sector-Level Predictions")
        st.caption(
            "Aggregated from constituent company momentum + BaatSeBharat regime data · cached 30 min"
        )

        with st.spinner("Loading cached sector predictions…"):
            sr_json  = _sector_returns.to_json()  if not _sector_returns.empty  else ""
            rdf_json = _regime_df.to_json()        if not _regime_df.empty        else ""
            sec_preds = _cached_sector_predictions(
                round(user_sent, 3), round(user_topic, 3), sr_json, rdf_json
            )

        if sec_preds:
            sec_rows = []
            for sp in sec_preds:
                sec_rows.append({
                    "Sector":    sp["sector"],
                    "Signal":    sp["signal"],
                    "Emoji":     sp["emoji"],
                    "Conf":      sp["confidence"],
                    "Score":     sp["score"],
                    "1D %":      sp["predictions"].get(1,  {}).get("return_pct", 0),
                    "5D %":      sp["predictions"].get(5,  {}).get("return_pct", 0),
                    "10D %":     sp["predictions"].get(10, {}).get("return_pct", 0),
                })
            sd = pd.DataFrame(sec_rows).sort_values("Score", ascending=False)

            # Sector cards grid
            for i in range(0, len(sd), 3):
                cols = st.columns(3)
                for j, row in enumerate(sd.iloc[i:i+3].itertuples()):
                    ret5 = getattr(row, '_7', 0)   # 5D %
                    ret1 = getattr(row, '_6', 0)   # 1D %
                    ret10= getattr(row, '_8', 0)   # 10D %
                    sc   = {"Bullish":"#2F6F4E","Bearish":"#A6503A","Neutral":"#9AA3B5"}[row.Signal]
                    rc   = "#2F6F4E" if ret5 > 0 else ("#A6503A" if ret5 < 0 else "#9AA3B5")
                    with cols[j]:
                        st.markdown(
                            f"""
                            <div style='background:#131B2C;border-radius:6px;padding:16px;
                                        border:1px solid {sc};border-left:3px solid {sc};margin-bottom:10px'>
                              <b style='color:{sc};font-family:"Fraunces",serif;font-weight:500'>
                                <span style='display:inline-block;width:8px;height:8px;border-radius:50%;background:{sc};margin-right:6px'></span>{row.Sector}</b>
                              <div style='color:#9AA3B5;font-size:0.78rem;font-family:"IBM Plex Sans",sans-serif'>
                                Confidence: <span style="font-family:'IBM Plex Mono',monospace">{row.Conf:.0f}%</span> &nbsp;·&nbsp; Regime: {user_regime}
                              </div>
                              <table style='width:100%;margin-top:8px;font-size:0.83rem;font-family:"IBM Plex Mono",monospace'>
                                <tr><td style='color:#9AA3B5'>1D</td>
                                    <td style='color:{rc};text-align:right'>{ret1:+.2f}%</td></tr>
                                <tr><td style='color:#9AA3B5'>1W</td>
                                    <td style='color:{rc};text-align:right'>{ret5:+.2f}%</td></tr>
                                <tr><td style='color:#9AA3B5'>10D</td>
                                    <td style='color:{rc};text-align:right'>{ret10:+.2f}%</td></tr>
                              </table>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            # Multi-horizon bar chart
            fig_sec = go.Figure()
            for (col_name, label), bar_color in zip(
                [("1D %", "1-Day"), ("5D %", "1-Week"), ("10D %", "10-Day")], CATEGORY_SEQUENCE
            ):
                fig_sec.add_trace(go.Bar(
                    name=label, x=sd["Sector"], y=sd[col_name],
                    text=[f"{v:+.2f}%" for v in sd[col_name]],
                    textposition="outside",
                    marker_color=bar_color,
                ))
            fig_sec.add_hline(y=0, line_dash="dot", line_color=COLORS["line"])
            apply_chart_theme(fig_sec, height=370)
            fig_sec.update_layout(
                title="Sector Return Forecasts — 1D / 1W / 10D",
                barmode="group",
                yaxis_title="Forecast Return (%)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1),
            )
            st.plotly_chart(fig_sec, use_container_width=True)

    st.info(
        "💡 **Caching:** Predictions refresh every 30 minutes, or immediately "
        "when you adjust the signal sliders. Price momentum is fetched live from yfinance."
    )

# ===========================================================================
# ── STAGE 7: GLOBAL INFLUENCE MAP ──────────────────────────────────────────
# ===========================================================================

elif stage == "7. Global Influence Map":
    if not _GEO_OK:
        st.error(f"GeoDashboard module failed to load: `{_geo_err_msg}`")
        st.info("Ensure `src/geo_dashboard.py` exists and `wbdata` is installed.")
        st.stop()

    # Build quick company predictions for the geo map (rule-based, cached)
    _map_preds = None
    if _PRED_OK:
        try:
            with st.spinner("Preparing geo signal data…"):
                _map_sent  = _load_avg_sentiment_from_db(DB_PATH)
                _map_topic = _load_topic_strength_from_npy()
                _map_preds = _cached_company_predictions(
                    round(_map_sent, 3), round(_map_topic, 3), "Neutral", 0.0
                )
        except Exception:
            _map_preds = None

    render_global_influence_map(company_predictions=_map_preds)

# ===========================================================================
# ── STAGE 8: GLOBAL PREVIEW ─────────────────────────────────────────────────
# ===========================================================================

elif stage == "8. Global Preview":
    stage_header("08", "Global Preview", "Past speeches: what the pipeline predicted vs. what actually happened.")

    if not _PREDHIST_OK:
        st.error(f"Prediction history module failed to load: `{_predhist_err_msg}`")
        st.info("Ensure `src/prediction_history.py` is present.")
        st.stop()

    gp_c1, gp_c2 = st.columns(2)
    with gp_c1:
        gp_source = st.selectbox(
            "Source", ["All", "Mann Ki Baat", "ECB", "Fed"], index=0, key="gp_source"
        )
    with gp_c2:
        gp_company = st.selectbox(
            "Company", ["All"] + list(COMPANY_UNIVERSE.keys()), index=0, key="gp_company"
        )

    with st.spinner("Replaying past predictions against realized outcomes…"):
        gp_df = compute_prediction_vs_actual(
            source=None if gp_source == "All" else gp_source,
            company=None if gp_company == "All" else gp_company,
        )

    if gp_df.empty:
        st.info(
            "Not enough data yet to build a Global Preview for this filter — "
            "needs speeches with computed market impact (speech_market_impact)."
        )
        st.stop()

    gp_summary = summarize_prediction_history(gp_df)

    hr = gp_summary.get("overall_hit_rate")
    metric_row([
        ("Directional Hit Rate (5D)", f"{hr*100:.1f}%" if hr is not None else "N/A", "sign(predicted) == sign(actual)"),
        ("Mean Abs. Error (1D)", f"{gp_summary.get('mean_abs_error_1d', 0):.2f}%", None),
        ("Mean Abs. Error (5D)", f"{gp_summary.get('mean_abs_error_5d', 0):.2f}%", None),
        ("Speeches Covered", f"{gp_summary.get('n_events', 0):,}", None),
    ])

    st.markdown("#### Predicted vs. Actual Return (5-Day)")
    gp_plot_df = gp_df.dropna(subset=["hit"]).copy()
    if not gp_plot_df.empty:
        gp_plot_df["Result"] = gp_plot_df["hit"].map({True: "Hit", False: "Miss"})
        fig_gp = px.scatter(
            gp_plot_df,
            x="predicted_return_5d", y="actual_return_5d",
            color="Result", hover_data=["date", "source", "company"],
            color_discrete_map={"Hit": COLORS["green"], "Miss": COLORS["rust"]},
        )
        _lo = gp_plot_df["predicted_return_5d"].min()
        _hi = gp_plot_df["predicted_return_5d"].max()
        fig_gp.add_shape(
            type="line", x0=_lo, y0=_lo, x1=_hi, y1=_hi,
            line=dict(color=COLORS["ink_dim"], dash="dot")
        )
        apply_chart_theme(fig_gp, height=420)
        fig_gp.update_layout(
            xaxis_title="Predicted 5D Return (%)", yaxis_title="Actual 5D Return (%)",
        )
        st.plotly_chart(fig_gp, use_container_width=True)
    else:
        st.caption("No events with a non-zero actual return to plot yet.")

    st.markdown("#### Speech-Level Detail")
    st.dataframe(
        gp_df.sort_values("date", ascending=False)[[
            "date", "source", "company", "predicted_signal",
            "predicted_return_1d", "predicted_return_5d",
            "actual_return_1d", "actual_return_5d", "hit"
        ]],
        use_container_width=True,
        hide_index=True,
    )

    _pc1, _pc2 = st.columns(2)
    with _pc1:
        if gp_summary.get("per_company") is not None and not gp_summary["per_company"].empty:
            st.markdown("#### Accuracy by Company")
            st.dataframe(gp_summary["per_company"], use_container_width=True, hide_index=True)
    with _pc2:
        if gp_summary.get("per_source") is not None and not gp_summary["per_source"].empty:
            st.markdown("#### Accuracy by Source")
            st.dataframe(gp_summary["per_source"], use_container_width=True, hide_index=True)
