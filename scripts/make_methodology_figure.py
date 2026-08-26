"""
make_methodology_figure.py
==========================
High-level design diagram for the paper: the layered architecture, showing
the textual and numerical branches converging into a single evaluation
harness that emits two contrasting outcomes.

    python scripts/make_methodology_figure.py

Writes BOTH formats from one source, so they cannot diverge:
    figs/methodology.svg   editable vector (matplotlib's native SVG backend)
    figs/methodology.png   400 dpi raster, what the .tex includes

No SVG->PNG converter is involved; matplotlib renders each backend directly.

This is a DESIGN diagram, not a data-flow trace: it shows components and how
they relate, not per-stage record counts. The only quantities shown are the
few that characterise a component (corpus size, instrument count), and those
are read from the database at draw time rather than typed in.

The diagram depicts only components that are implemented and exercised in
the reported results. Mechanisms considered during design but not built are
absent rather than greyed out, so a reader need not decode which boxes are
real.
"""
import os
import sqlite3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

DB = './data/market_rhetoric.db'
OUT = './figs'
DPI = 400

INK = '#1a1a1a'
DIM = '#4a4a4a'
EDGE = '#8a8a8a'
FILL = '#f5f5f5'
FILL_ALT = '#ececec'
ACCENT_OK = '#2f6fb3'    # positive control
ACCENT_NULL = '#b3402f'  # the null

TITLE_FS = 7.4
BODY_FS = 6.1

# Characters that fit per line, by box width in axes units, at BODY_FS on a
# 3.4in-wide figure. Enforced rather than trusted: an earlier version of this
# figure silently overflowed every box because the strings were written
# without measuring them against the box that had to hold them.
CHARS_PER_UNIT = 62.0


def counts():
    c = sqlite3.connect(DB)
    q = lambda s: c.execute(s).fetchone()[0]
    d = dict(
        total=q("SELECT COUNT(*) FROM speeches"),
        instr=q("SELECT COUNT(DISTINCT ticker) FROM market_data"),
    )
    c.close()
    return d


def _render_len(s):
    """Approximate the DRAWN width of a label in characters.

    Counting raw string length overstates any mathtext: '$\\rightarrow$' is
    13 characters of markup that renders as a single arrow glyph, and an
    escaped '\\%' is two characters that render as one. The guard has to
    measure what appears on the canvas, not what is typed.
    """
    out, i = 0, 0
    while i < len(s):
        if s[i] == '$':                       # mathtext span -> ~1 glyph
            j = s.find('$', i + 1)
            if j == -1:
                out += 1
                break
            out += 1
            i = j + 1
        elif s[i] == BACKSLASH and i + 1 < len(s):   # escaped char -> 1 glyph
            out += 1
            i += 2
        else:
            out += 1
            i += 1
    return out


BACKSLASH = chr(92)


def box(ax, x, y, w, h, title, lines, fill=FILL, accent=None, title_fs=TITLE_FS):
    cap = int(w * CHARS_PER_UNIT)
    bad = [f'{_render_len(s)}c > {cap}c: {s!r}'
           for s in ([title] + list(lines)) if _render_len(s) > cap]
    if bad:
        raise ValueError(f'box "{title}" overflows:\n  ' + '\n  '.join(bad))

    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.006,rounding_size=0.012',
        linewidth=1.2, edgecolor=accent or EDGE, facecolor=fill, zorder=2))
    ax.text(x + w / 2, y + h - 0.022, title, ha='center', va='top',
            fontsize=title_fs, fontweight='bold', color=accent or INK, zorder=3)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.062 - i * 0.031, ln, ha='center', va='top',
                fontsize=BODY_FS, color=DIM, zorder=3)


def arrow(ax, x1, y1, x2, y2, color=EDGE):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=8,
        linewidth=1.0, color=color, zorder=5, shrinkA=1, shrinkB=1))


def band(ax, y, label):
    """Layer label down the left margin."""
    ax.text(-0.055, y, label, ha='center', va='center', rotation=90,
            fontsize=5.8, color='#909090', fontweight='bold', zorder=1)


def draw(d, ext):
    fig, ax = plt.subplots(figsize=(3.42, 4.15))
    # margin beyond [0,1]: FancyBboxPatch draws its border outside the
    # nominal rectangle, and the layer labels sit left of x=0
    ax.set_xlim(-0.085, 1.02)
    ax.set_ylim(-0.01, 1.01)
    ax.axis('off')

    FULL_X, FULL_W = 0.02, 0.96
    # wider centre gutter so the paired boxes read as two components rather
    # than one split box
    LX, RX, HW = 0.02, 0.535, 0.445

    # ---- layer 1: sources ------------------------------------------------
    y = 0.845
    box(ax, FULL_X, y, FULL_W, 0.145, 'Data Sources', [
        f"{d['total']:,} leadership documents",
        'Federal Reserve / ECB / Mann ki Baat',
        f"{d['instr']} equity instruments + India VIX",
    ], fill=FILL_ALT)
    band(ax, y + 0.072, 'INPUT')

    # ---- layer 2: two branches ------------------------------------------
    y2 = 0.605
    box(ax, LX, y2, HW, 0.185, 'Textual Branch', [
        'POS filter + lemmatize',
        'TF-IDF $\\rightarrow$ NMF topics',
        'FinBERT sentiment',
    ])
    box(ax, RX, y2, HW, 0.185, 'Numerical Branch', [
        'forward returns',
        'realized volatility',
        'HMM regimes, ASBN',
    ])
    band(ax, y2 + 0.092, 'REPRESENT')

    arrow(ax, 0.28, y, 0.255, y2 + 0.185)
    arrow(ax, 0.72, y, 0.745, y2 + 0.185)

    # ---- layer 3: alignment ---------------------------------------------
    y3 = 0.435
    box(ax, FULL_X, y3, FULL_W, 0.125, 'Event Alignment', [
        'each document paired with each instrument',
        'strictly pre-event features only',
    ])
    band(ax, y3 + 0.062, 'FUSE')

    arrow(ax, 0.255, y2, 0.40, y3 + 0.125)
    arrow(ax, 0.745, y2, 0.60, y3 + 0.125)

    # ---- layer 4: evaluation --------------------------------------------
    y4 = 0.245
    box(ax, FULL_X, y4, FULL_W, 0.145, 'Evaluation Harness', [
        'walk-forward refits, no look-ahead',
        # plain '%': this is matplotlib text, not LaTeX -- an escaped
        # '\%' renders the backslash literally on the canvas
        'cross-sectional target, 50% baseline',
        'standard errors clustered by event date',
    ])
    band(ax, y4 + 0.072, 'TEST')

    arrow(ax, 0.5, y3, 0.5, y4 + 0.145)

    # ---- layer 5: outcomes ----------------------------------------------
    y5 = 0.045
    box(ax, LX, y5, HW, 0.15, 'Return Direction', [
        'no detectable signal',
        'at any horizon',
        '1 to 252 days',
    ], accent=ACCENT_NULL)
    box(ax, RX, y5, HW, 0.15, 'Volatility', [
        'strong effect recovered',
        'positive control:',
        'the harness works',
    ], accent=ACCENT_OK)
    band(ax, y5 + 0.075, 'OUTCOME')

    arrow(ax, 0.40, y4, 0.255, y5 + 0.15, ACCENT_NULL)
    arrow(ax, 0.60, y4, 0.745, y5 + 0.15, ACCENT_OK)

    fig.tight_layout(pad=0.10)
    path = f'{OUT}/methodology.{ext}'
    if ext == 'png':
        fig.savefig(path, dpi=DPI)
    else:
        fig.savefig(path)
    plt.close(fig)
    return path


def main():
    os.makedirs(OUT, exist_ok=True)
    d = counts()
    print(f"read from database: {d['total']} documents, {d['instr']} instruments\n")
    for ext in ('svg', 'png'):
        p = draw(d, ext)
        print(f'  wrote {p}  ({os.path.getsize(p) / 1024:.0f} KB)')
    print('\nBoth formats rendered from the same source -- no conversion step.')


if __name__ == '__main__':
    main()
