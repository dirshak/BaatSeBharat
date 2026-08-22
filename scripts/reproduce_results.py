"""
reproduce_results.py
====================
Emits every numeric claim in paper_conference.tex, plus its two figures.

Run from the project root:
    python scripts/reproduce_results.py

Outputs:
    stdout          -- all tables, section by section, keyed to the paper
    figs/fig_baserate.pdf
    figs/fig_coverage.pdf

Determinism: the walk-forward split is index-quantile based, gradient
boosting is seeded, and logistic regression is convex, so repeated runs are
bit-identical. This matters -- an earlier draft quoted numbers from a
feature set that differed by one column (realized volatility), which moved
the headline from 51.32% to 50.53%. The canonical 196-feature
specification below is the one the paper reports, and nothing else.

CANONICAL FEATURE SET (196):
      5  sentiment           compound, optimism, risk, positive, negative
     10  topic probabilities NMF 'Combined' model, K=10
    160  topic x instrument  the hypothesis under test
     16  instrument one-hot
      5  market controls     rel_mom 5/10/20/60, rel_vol_20 (demeaned
                             within event, as a relative target requires)
"""
import os
import sqlite3
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

warnings.filterwarnings('ignore')

DB = './data/market_rhetoric.db'
OUT = './figs'
HORIZONS = [1, 5, 10, 20, 60, 120, 252]
N_FOLDS = 5


def hdr(t):
    print('\n' + '=' * 72)
    print(t)
    print('=' * 72)


# ---------------------------------------------------------------- load ----
def load():
    conn = sqlite3.connect(DB)
    d = dict(
        total=conn.execute("SELECT COUNT(*) FROM speeches").fetchone()[0],
        dated=conn.execute(
            "SELECT COUNT(*) FROM speeches WHERE date IS NOT NULL AND date != 'N/A'"
        ).fetchone()[0],
        by_src=pd.read_sql_query(
            "SELECT source, COUNT(*) n FROM speeches GROUP BY source ORDER BY n DESC", conn),
        td_all=conn.execute("SELECT COUNT(*) FROM topic_distributions").fetchone()[0],
        td_comb=conn.execute(
            "SELECT COUNT(*) FROM topic_distributions WHERE model_name='Combined'").fetchone()[0],
        n_regime=conn.execute("SELECT COUNT(*) FROM regime_classifications").fetchone()[0],
        n_vix=conn.execute("SELECT COUNT(*) FROM vix_data").fetchone()[0],
        n_llm=conn.execute("SELECT COUNT(*) FROM llm_company_signals").fetchone()[0],
        n_llm_sp=conn.execute(
            "SELECT COUNT(DISTINCT speech_id) FROM llm_company_signals").fetchone()[0],
        speeches=pd.read_sql_query(
            "SELECT id, date, source FROM speeches WHERE date IS NOT NULL", conn),
        sent=pd.read_sql_query(
            """SELECT speech_id, optimism_intensity, risk_awareness, positive,
                      negative, compound
               FROM sentiment_scores WHERE segment_type='episode'""", conn),
        topics=pd.read_sql_query(
            """SELECT speech_id, topic_id, probability FROM topic_distributions
               WHERE model_name='Combined'""", conn),
        impact=pd.read_sql_query(
            """SELECT speech_id, ticker, return_t5, abnormal_return
               FROM speech_market_impact""", conn),
        market=pd.read_sql_query(
            "SELECT date, ticker, returns FROM market_data WHERE returns IS NOT NULL", conn),
    )
    conn.close()
    return d


def build(d):
    sp = d['speeches'].copy()
    sp['date'] = pd.to_datetime(sp['date'], errors='coerce')
    sp = sp.dropna(subset=['date'])
    mk = d['market'].copy()
    mk['date'] = pd.to_datetime(mk['date'])
    mk = mk.sort_values(['ticker', 'date'])
    tw = d['topics'].pivot_table(index='speech_id', columns='topic_id',
                                values='probability', fill_value=0.0)
    tw.columns = [f'topic_{c}' for c in tw.columns]
    per = {t: (g['date'].to_numpy(), g['returns'].to_numpy(dtype=float))
           for t, g in mk.groupby('ticker', sort=False)}
    return sp, mk, tw, per


def make_fwd(per):
    def fwd(tk, ed, n):
        dts, r = per[tk]
        p = np.searchsorted(dts, np.datetime64(ed), 'right')
        w = r[p:p + n]
        return float(np.prod(1 + w) - 1) if w.size else np.nan
    return fwd


def make_past(per):
    def pastw(tk, ed, n, vol=False):
        dts, r = per[tk]
        p = np.searchsorted(dts, np.datetime64(ed), 'left')
        if p < n:
            return np.nan
        w = r[p - n:p]
        return float(np.std(w)) if vol else float(np.prod(1 + w) - 1)
    return pastw


def clustered(hit, dates):
    """Mean and SE of the hit rate, clustered by event date. Each speech
    contributes one row per instrument and those rows are ~0.41 correlated,
    so unclustered SEs are understated ~2.68x."""
    per = pd.DataFrame({'d': dates, 'h': hit.astype(float)}).groupby('d')['h'].mean()
    return float(per.mean()), float(per.std(ddof=1) / np.sqrt(len(per))), len(per)


def walk_forward(F, y, dates, model_fn, n_folds=N_FOLDS):
    """Expanding window: fit on [0, a), predict [a, e). No test row ever
    informs the fit that predicts it."""
    b = np.quantile(np.arange(len(y)), np.linspace(0.4, 1.0, n_folds + 1)).astype(int)
    P, Q, A, D = [], [], [], []
    for i in range(n_folds):
        a, e = b[i], b[i + 1]
        if a < 200 or e <= a or len(np.unique(y[:a])) < 2:
            continue
        m = model_fn()
        m.fit(F[:a], y[:a])
        Q.append(m.predict_proba(F[a:e])[:, 1])
        P.append(m.predict(F[a:e]))
        A.append(y[a:e])
        D.append(dates[a:e])
    return np.concatenate(P), np.concatenate(Q), np.concatenate(A), np.concatenate(D)


MODELS = {
    'Logistic (C=0.05)': lambda: make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=3000, C=0.05)),
    'Logistic (C=1)': lambda: make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=3000, C=1.0)),
    'Gradient boosting': lambda: HistGradientBoostingClassifier(
        max_iter=200, random_state=0),
}


def main():
    os.makedirs(OUT, exist_ok=True)
    d = load()
    sp, mk, tw, per = build(d)
    fwd, pastw = make_fwd(per), make_past(per)

    # ---------------------------------------------------- Table I: corpus --
    hdr('TABLE I  Corpus')
    print(d['by_src'].to_string(index=False))
    print(f"documents total            : {d['total']}")
    print(f"  with parseable date      : {d['dated']}")
    print(f"market rows (with returns) : {len(mk)}   instruments: {mk.ticker.nunique()}")
    print(f"event observations         : {len(d['impact'])}")
    print(f"sentiment scores           : {len(d['sent'])}")
    print(f"topic assignments (pooled) : {d['td_comb']}   (all models: {d['td_all']})")
    print(f"regime classifications     : {d['n_regime']}   VIX rows: {d['n_vix']}")
    print(f"LLM signals                : {d['n_llm']} across {d['n_llm_sp']} documents")
    print(f"span                       : {sp.date.min().date()} .. {sp.date.max().date()}")

    df = (d['impact'].merge(sp, left_on='speech_id', right_on='id')
                     .merge(d['sent'], on='speech_id', how='left')
                     .merge(tw, left_on='speech_id', right_index=True, how='left'))
    df = df.dropna(subset=['compound', 'return_t5']).copy()

    # -------------------------------------------- Fig 1: base rate curve --
    sub = df.drop_duplicates(subset=['speech_id', 'ticker'])
    base_by_h = []
    for h in HORIZONS:
        r = np.array([fwd(t, dt, h) for t, dt in zip(sub['ticker'], sub['date'])])
        r = r[~np.isnan(r)]
        base_by_h.append((r > 0).mean())

    # ------------------------------------------------ canonical features --
    df['med'] = df.groupby('speech_id')['return_t5'].transform('median')
    df['y_rel'] = (df['return_t5'] > df['med']).astype(int)
    df['y_abs'] = (df['return_t5'] > 0).astype(int)
    df = df[df['return_t5'] != df['med']].copy()          # drop median ties

    for n in (5, 10, 20, 60):
        df[f'mom_{n}'] = [pastw(t, dt, n) for t, dt in zip(df['ticker'], df['date'])]
        df[f'rel_mom_{n}'] = df[f'mom_{n}'] - df.groupby('speech_id')[f'mom_{n}'].transform('mean')
    df['vol_20'] = [pastw(t, dt, 20, vol=True) for t, dt in zip(df['ticker'], df['date'])]
    df['rel_vol_20'] = df['vol_20'] - df.groupby('speech_id')['vol_20'].transform('mean')

    MKT = ['rel_mom_5', 'rel_mom_10', 'rel_mom_20', 'rel_mom_60', 'rel_vol_20']
    df = df.dropna(subset=MKT).sort_values('date').reset_index(drop=True)

    tick = pd.get_dummies(df['ticker'], prefix='tk').astype(float)
    TOPS = list(tw.columns)
    inter = pd.DataFrame({f'{tc}_x_{t}': df[tc].to_numpy() * tick[t].to_numpy()
                          for tc in TOPS for t in tick.columns})
    SENTF = ['compound', 'optimism_intensity', 'risk_awareness', 'positive', 'negative']
    FEAT = pd.concat([inter, tick, df[SENTF], df[TOPS], df[MKT]], axis=1)
    F = FEAT.fillna(0.0).to_numpy(dtype=float)
    dates = df['date'].to_numpy()
    y_rel, y_abs = df['y_rel'].to_numpy(), df['y_abs'].to_numpy()

    print(f"\nanalysis set: rows={len(df)}  documents={df.speech_id.nunique()}  "
          f"event dates={df.date.nunique()}  features={F.shape[1]}")
    print(f"  = {len(SENTF)} sent + {len(TOPS)} topic + {inter.shape[1]} interaction "
          f"+ {tick.shape[1]} instrument + {len(MKT)} market")

    # --------------------------------------- Table II: main + ablation ----
    hdr('TABLE II  Cross-sectional target (baseline exactly 50.0%)')
    best = None
    for nm, fn in MODELS.items():
        p, q, a, dd = walk_forward(F, y_rel, dates, fn)
        h, se, nc = clustered(p == a, dd)
        print(f"  {nm:20s} hit={h*100:6.2f}%  SE={se*100:4.2f}pp  "
              f"z={(h-.5)/se:+5.2f}   n={len(p)}  clusters={nc}")
        if best is None or h > best[0]:
            best = (h, q, a, dd, nm)

    print('\n  Feature-group ablation (logistic C=1):')
    n_int, n_tk = inter.shape[1], tick.shape[1]
    groups = {
        'rhetoric only (sent+topics)': list(range(n_int + n_tk, F.shape[1] - len(MKT))),
        'topic x instrument only': list(range(n_int)),
        'market controls only': list(range(F.shape[1] - len(MKT), F.shape[1])),
    }
    for gn, cols in groups.items():
        p, q, a, dd = walk_forward(F[:, cols], y_rel, dates, MODELS['Logistic (C=1)'])
        h, se, _ = clustered(p == a, dd)
        print(f"    {gn:30s} hit={h*100:6.2f}%  z={(h-.5)/se:+5.2f}")

    # ------------------------------------------- absolute-target result ---
    hdr('Absolute-direction target (baseline = majority class)')
    p, q, a, dd = walk_forward(F, y_abs, dates, MODELS['Gradient boosting'])
    h, se, _ = clustered(p == a, dd)
    base = max(a.mean(), 1 - a.mean())
    print(f"  Gradient boosting  hit={h*100:.2f}%  baseline={base*100:.2f}%  "
          f"edge={(h-base)*100:+.2f}pp  z_vs_baseline={(h-base)/se:+.2f}")

    # ---------------------------------- Fig 2: selective prediction curve --
    hdr('Selective prediction (best cross-sectional model)')
    _, qb, ab, db, bname = best
    conf = np.abs(qb - 0.5)
    cov_pts = []
    print(f"  model: {bname}")
    for c in (1.0, 0.5, 0.25, 0.10, 0.05, 0.02):
        k = max(30, int(len(qb) * c))
        idx = np.argsort(-conf)[:k]
        pr = (qb[idx] > 0.5).astype(int)
        h, se, _ = clustered(pr == ab[idx], db[idx])
        cov_pts.append((c, k, h, se))
        print(f"    coverage {c*100:5.1f}%  n={k:6d}  hit={h*100:5.1f}%  "
              f"SE={se*100:4.2f}pp  z={(h-.5)/se:+5.2f}")

    # --------------------------------------------- Table IV: inflation ----
    hdr('TABLE IV  Accuracy obtainable without predictive skill')
    print(f"  [1] horizon 252d           : {max(base_by_h[-1], 1-base_by_h[-1])*100:.1f}%"
          f"  (baseline identical)")
    rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    rf.fit(F, y_rel)
    print(f"  [2] random forest in-sample: {(rf.predict(F)==y_rel).mean()*100:.1f}%")
    split = int(len(df) * 0.7)
    leak = np.column_stack([F, df['abnormal_return'].fillna(0).to_numpy()])
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    m.fit(leak[:split], y_rel[:split])
    print(f"  [3] + post-event variable  : "
          f"{(m.predict(leak[split:])==y_rel[split:]).mean()*100:.1f}%")

    # knob 4 under two targets, same 15 features and split -> sign flips
    F15 = df[SENTF + TOPS].fillna(0).to_numpy(float)
    out15 = {}
    for tag, yy in (('absolute', y_abs), ('cross-sectional', y_rel)):
        mm = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        mm.fit(F15[:split], yy[:split])
        qq = mm.predict_proba(F15[split:])[:, 1]
        aa = yy[split:]
        cc = np.abs(qq - .5)
        k = max(20, int(len(qq) * 0.01))
        idx = np.argsort(-cc)[:k]
        out15[tag] = (((qq[idx] > .5).astype(int) == aa[idx]).mean(), k)
    print(f"  [4] top 1% most-confident  : "
          f"{out15['absolute'][0]*100:.1f}% (absolute target) vs "
          f"{out15['cross-sectional'][0]*100:.1f}% (cross-sectional), "
          f"n={out15['absolute'][1]} -- sign is specification-dependent")

    # ------------------------------------------------ pseudo-replication --
    hdr('Pseudo-replication and lookahead')
    piv = d['impact'].dropna(subset=['abnormal_return']).pivot_table(
        index='speech_id', columns='ticker', values='abnormal_return')
    cm = piv.corr().values
    iu = np.triu_indices_from(cm, 1)
    rho, k = float(np.nanmean(cm[iu])), piv.shape[1]
    print(f"  instruments={k}  mean pairwise rho={rho:.3f}  "
          f"design effect={1+(k-1)*rho:.1f}x  SE inflation={np.sqrt(1+(k-1)*rho):.2f}x")
    ii = d['impact'].dropna(subset=['abnormal_return', 'return_t5'])
    flip = (np.sign(ii.return_t5) != np.sign(ii.abnormal_return)).mean()
    print(f"  full-sample abnormal baseline flips sign of {flip*100:.1f}% of events")

    hdr('Base rate by horizon (always-up; no model)')
    for h_, b_ in zip(HORIZONS, base_by_h):
        print(f"  {h_:4d}d  {b_*100:5.1f}%")

    # ------------------------------------------------------- figures ------
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False})

    fig, ax = plt.subplots(figsize=(3.4, 2.25))
    ax.plot(range(len(HORIZONS)), [b * 100 for b in base_by_h], 'o-',
            color='#b3402f', lw=1.6, ms=4)
    ax.axhline(50, ls=':', color='#666', lw=1)
    ax.set_xticks(range(len(HORIZONS)))
    ax.set_xticklabels([str(h) for h in HORIZONS])
    ax.set_xlabel('Forecast horizon (trading days)')
    ax.set_ylabel('Always-up accuracy (%)')
    ax.text(0.03, 0.95, 'no model involved', transform=ax.transAxes,
            fontsize=7.5, style='italic', color='#555', va='top')
    ax.set_ylim(45, 78)
    fig.tight_layout(pad=0.3)
    fig.savefig(f'{OUT}/fig_baserate.pdf')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.4, 2.25))
    xs = [c * 100 for c, _, _, _ in cov_pts]
    ys = [h * 100 for _, _, h, _ in cov_pts]
    es = [1.96 * se * 100 for _, _, _, se in cov_pts]
    ax.errorbar(xs, ys, yerr=es, fmt='o-', color='#2f6fb3', lw=1.6, ms=4,
                capsize=2.5, elinewidth=1)
    ax.axhline(50, ls=':', color='#666', lw=1, label='baseline 50%')
    ax.set_xscale('log')
    ax.set_xticks(xs)
    ax.set_xticklabels([f'{x:g}' for x in xs])
    ax.invert_xaxis()
    ax.set_xlabel('Coverage retained (%, log scale)')
    ax.set_ylabel('Hit rate (%)')
    ax.legend(frameon=False, fontsize=7.5, loc='upper left')
    fig.tight_layout(pad=0.3)
    fig.savefig(f'{OUT}/fig_coverage.pdf')
    plt.close(fig)

    print(f"\nfigures written: {OUT}/fig_baserate.pdf, {OUT}/fig_coverage.pdf")
    print("\nNote: Granger/FDR results are produced by "
          "src/models/causal_validation.py (see paper Table III).")
    print("Note: FinBERT truncation coverage -- run with --tokens "
          "(loads the transformers tokenizer, slower).")


def token_coverage():
    """Measure how much of each document FinBERT actually reads.

    Kept behind a flag because it pulls in transformers. Worth running: an
    earlier draft estimated 'about 32%' from a character-count heuristic;
    the real figure for Mann ki Baat is 10.7%, which changes how seriously
    the truncation limitation should be taken.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('ProsusAI/finbert')
    conn = sqlite3.connect(DB)
    hdr('FinBERT truncation coverage (512-token limit)')
    for src in ('Mann Ki Baat', 'ECB', 'Fed'):
        rows = conn.execute(
            "SELECT full_text FROM speeches WHERE source=? AND full_text IS NOT NULL "
            "AND length(full_text)>200 LIMIT 40", (src,)).fetchall()
        lens = np.array([len(tok.encode(r[0], truncation=False,
                                        add_special_tokens=True)) for r in rows])
        cov = np.minimum(lens, 512) / lens
        print(f"  {src:14s} n={len(lens):3d}  median tokens={int(np.median(lens)):5d}  "
              f"median coverage={np.median(cov)*100:4.1f}%")
    conn.close()


if __name__ == '__main__':
    import sys
    if '--tokens' in sys.argv:
        token_coverage()
    else:
        main()
