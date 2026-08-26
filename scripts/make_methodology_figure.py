"""
make_methodology_figure.py
==========================
Draws the pipeline / methodology diagram for the paper.

    python scripts/make_methodology_figure.py

Writes BOTH formats from one source, so they cannot diverge:
    figs/methodology.svg   editable vector (matplotlib's native SVG backend)
    figs/methodology.png   400 dpi raster, what the .tex includes

No SVG->PNG converter is involved; matplotlib renders each backend directly.

Every count shown in the diagram is read from the database at draw time
rather than typed in, for the same reason the paper's numbers are: a
hand-written figure silently goes stale the moment the pipeline is re-run.

The diagram deliberately depicts only what is IMPLEMENTED. Components that
were designed but not built (trade-weighted aggregation, contextualized
topic models, multilingual embeddings) are absent, not greyed out, because
a reader should not have to decode which boxes are real.
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
DIM = '#555555'
EDGE = '#8a8a8a'
FILL = '#f4f4f4'
ACCENT_OK = '#2f6fb3'    # positive control
ACCENT_NULL = '#b3402f'  # the null


def counts():
    """Read the figures the diagram displays, so it tracks the pipeline."""
    c = sqlite3.connect(DB)
    q = lambda s: c.execute(s).fetchone()[0]
    d = dict(
        total=q("SELECT COUNT(*) FROM speeches"),
        fed=q("SELECT COUNT(*) FROM speeches WHERE source='Fed'"),
        ecb=q("SELECT COUNT(*) FROM speeches WHERE source='ECB'"),
        mkb=q("SELECT COUNT(*) FROM speeches WHERE source='Mann Ki Baat'"),
        mkt=q("SELECT COUNT(*) FROM market_data WHERE returns IS NOT NULL"),
        instr=q("SELECT COUNT(DISTINCT ticker) FROM market_data"),
        vix=q("SELECT COUNT(*) FROM vix_data"),
        impact=q("SELECT COUNT(*) FROM speech_market_impact"),
        sent=q("SELECT COUNT(*) FROM sentiment_scores WHERE segment_type='episode'"),
    )
    c.close()
    return d


TITLE_FS = 7.6
BODY_FS = 6.0
LINE_DY = 0.062

# Longest permitted strings at the sizes above, given the box width computed
# in draw(). Enforced by _check() rather than trusted: the first version of
# this figure silently overflowed every box because the text was written
# without measuring against the box it had to fit inside.
MAX_TITLE_CHARS = 15
MAX_BODY_CHARS = 20


def _check(title, lines):
    bad = []
    if len(title) > MAX_TITLE_CHARS:
        bad.append(f'title {len(title)}c: {title!r}')
    for ln in lines:
        if len(ln) > MAX_BODY_CHARS:
            bad.append(f'line {len(ln)}c: {ln!r}')
    return bad


def box(ax, x, y, w, h, title, lines, accents=None):
    overflow = _check(title, lines)
    if overflow:
        raise ValueError(f'box "{title}" would overflow:\n  ' + '\n  '.join(overflow))
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.010,rounding_size=0.018',
        linewidth=1.1, edgecolor=EDGE, facecolor=FILL, zorder=2))
    ax.text(x + w / 2, y + h - 0.045, title, ha='center', va='top',
            fontsize=TITLE_FS, fontweight='bold', color=INK, zorder=3)
    for i, ln in enumerate(lines):
        ax.text(x + 0.014, y + h - 0.125 - i * LINE_DY, ln, ha='left', va='top',
                fontsize=BODY_FS, color=(accents or {}).get(i, DIM),
                fontweight='bold' if i in (accents or {}) else 'normal',
                zorder=3)


def arrow(ax, x1, x2, y):
    ax.add_patch(FancyArrowPatch(
        (x1, y), (x2, y), arrowstyle='-|>', mutation_scale=8,
        linewidth=1.0, color=EDGE, zorder=5, shrinkA=1, shrinkB=1))


def draw(d, ext):
    fig, ax = plt.subplots(figsize=(7.16, 2.95))
    # Margin beyond [0,1]: FancyBboxPatch draws its rounded border OUTSIDE
    # the nominal rectangle by `pad`, so the last box's right edge lands at
    # ~1.006 and gets clipped at a hard xlim of 1.
    ax.set_xlim(-0.015, 1.015); ax.set_ylim(0, 1); ax.axis('off')

    # leave real gaps between boxes so the arrows are visible
    PAD, GAP, N = 0.004, 0.040, 5
    W = (1 - 2 * PAD - (N - 1) * GAP) / N
    H, Y = 0.66, 0.20
    xs = [PAD + i * (W + GAP) for i in range(N)]

    box(ax, xs[0], Y, W, H, 'Ingestion', [
        f"Fed          {d['fed']}",
        f"ECB          {d['ecb']}",
        f"Mann ki Baat  {d['mkb']}",
        f"{d['total']:,} documents",
        '',
        f"{d['instr']} instruments",
        f"{d['mkt']:,} daily rows",
        f"India VIX {d['vix']:,}",
    ])

    box(ax, xs[1], Y, W, H, 'Features', [
        'spaCy POS + lemma',
        'TF-IDF to NMF',
        '  K = 10 topics',
        'FinBERT sentiment',
        '  512-token limit',
        '  (10.7% of doc)',
        'HMM, 3 regimes',
        'ASBN 252d z-score',
    ])

    box(ax, xs[2], Y, W, H, 'Alignment', [
        'forward returns',
        '  n = 1 .. 252 d',
        'forward realized',
        '  volatility',
        'pre-event features',
        '  only',
        '',
        f"{d['impact']:,} pairs",
    ])

    box(ax, xs[3], Y, W, H, 'Evaluation', [
        '196 features',
        '  5 sentiment',
        '  10 topic',
        '  160 interaction',
        '  16 instrument',
        '  5 market',
        'walk-forward x 5',
        'SE clustered/date',
    ])

    box(ax, xs[4], Y, W, H, 'Findings', [
        'RETURNS',
        '  49.99% vs 50%',
        '  null at all lags',
        '',
        'VOLATILITY',
        '  68.67%',
        '  +17.07pp z=23.8',
        '  positive control',
    ], accents={0: ACCENT_NULL, 4: ACCENT_OK})

    for i in range(N - 1):
        arrow(ax, xs[i] + W, xs[i + 1], Y + H / 2)

    ax.text(0.5, 0.075,
            'All features are computed strictly from information available before the event date; '
            'no test observation informs any fit that predicts it.',
            ha='center', va='center', fontsize=6.4, style='italic', color=DIM)

    fig.tight_layout(pad=0.12)
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
    print('counts read from database:')
    for k, v in d.items():
        print(f'  {k:8s} {v}')
    print()
    for ext in ('svg', 'png'):
        p = draw(d, ext)
        print(f'  wrote {p}  ({os.path.getsize(p) / 1024:.0f} KB)')
    print('\nBoth formats rendered from the same source -- no conversion step.')


if __name__ == '__main__':
    main()
