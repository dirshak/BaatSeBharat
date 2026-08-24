"""
reproduce_results.py
====================
Emits every numeric claim in paper_conference.tex, plus its three figures.

Run from the project root:
    python scripts/reproduce_results.py            # all tables + figures
    python scripts/reproduce_results.py --tokens   # FinBERT truncation only

Outputs:
    stdout                  -- tables, keyed to the paper by section
    figs/fig_horizon.png    -- Fig 1: model vs baseline accuracy by horizon
    figs/fig_coverage.png   -- Fig 2: hit rate vs retained coverage
    figs/fig_volatility.png -- Fig 3: volatility target, model vs baseline

Determinism: walk-forward splits are index-quantile based, gradient boosting
is seeded, logistic regression is convex. Repeated runs are bit-identical.
This matters -- an earlier draft quoted numbers from a feature set that
differed by one column (realized volatility), which moved the headline by
0.8pp. The canonical 196-feature specification below is what the paper
reports, and nothing else.

CANONICAL RETURN FEATURE SET (196):
      5  sentiment           compound, optimism, risk, positive, negative
     10  topic probabilities NMF 'Combined' model, K=10
    160  topic x instrument  the hypothesis under test
     16  instrument one-hot
      5  market controls     rel_mom 5/10/20/60, rel_vol_20 (demeaned
                             within event, as a relative target requires)
"""
import os
import sqlite3
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

warnings.filterwarnings('ignore')

DB = './data/market_rhetoric.db'
OUT = './figs'
HORIZONS = [1, 5, 10, 20, 30, 60, 120, 252]
VOL_HORIZONS = [10, 20, 30]
N_FOLDS = 5

# Raster output at 400 dpi. IEEE requires >=300 dpi for raster figures; at
# the 3.4in column width used below that is ~1360px across, which stays
# crisp in print. Vector (pdf) would be sharper still, but the paper's
# figures are requested as png.
FIG_EXT = 'png'
FIG_DPI = 400

SENTF = ['compound', 'optimism_intensity', 'risk_awareness', 'positive', 'negative']
GBM = lambda: HistGradientBoostingClassifier(max_iter=200, random_state=0)
LOG05 = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.05))
LOG1 = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))


def hdr(t):
    print('\n' + '=' * 78)
    print(t)
    print('=' * 78)


# ------------------------------------------------------------------ data --
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
        vix=pd.read_sql_query("SELECT date, vix_close FROM vix_data", conn),
    )
    conn.close()
    return d


class Series:
    """Per-instrument return series with strictly-causal accessors."""

    def __init__(self, market):
        self.per = {t: (g['date'].to_numpy(), g['returns'].to_numpy(dtype=float))
                    for t, g in market.groupby('ticker', sort=False)}

    def fwd(self, tk, ed, n):
        d, r = self.per[tk]
        p = np.searchsorted(d, np.datetime64(ed), 'right')
        w = r[p:p + n]
        return float(np.prod(1 + w) - 1) if w.size == n else np.nan

    def past(self, tk, ed, n, vol=False):
        d, r = self.per[tk]
        p = np.searchsorted(d, np.datetime64(ed), 'left')
        if p < n:
            return np.nan
        w = r[p - n:p]
        return float(np.std(w)) if vol else float(np.prod(1 + w) - 1)

    def fwd_vol(self, tk, ed, n):
        d, r = self.per[tk]
        p = np.searchsorted(d, np.datetime64(ed), 'right')
        w = r[p:p + n]
        return float(np.std(w)) if w.size == n else np.nan


def clustered(hit, dates):
    """Per-date hit means. Each speech contributes one row per instrument and
    those rows are ~0.41 correlated, so unclustered SEs understate ~2.68x."""
    return pd.DataFrame({'d': dates, 'h': hit.astype(float)}).groupby('d')['h'].mean()


def summarize(series):
    m = float(series.mean())
    se = float(series.std(ddof=1) / np.sqrt(len(series)))
    return m, se, len(series)


def walk_forward(F, y, dates, model_fn, n_folds=N_FOLDS):
    """Expanding window: fit on [0,a), predict [a,e). No test row informs the
    fit that predicts it."""
    b = np.quantile(np.arange(len(y)), np.linspace(0.4, 1.0, n_folds + 1)).astype(int)
    P, Q, A, D = [], [], [], []
    for i in range(n_folds):
        a, e = b[i], b[i + 1]
        if a < 200 or e <= a or len(np.unique(y[:a])) < 2:
            continue
        m = model_fn()
        m.fit(F[:a], y[:a])
        try:
            Q.append(m.predict_proba(F[a:e])[:, 1])
        except Exception:
            Q.append(np.full(e - a, np.nan))
        P.append(m.predict(F[a:e]))
        A.append(y[a:e])
        D.append(dates[a:e])
    return (np.concatenate(P), np.concatenate(Q),
            np.concatenate(A), np.concatenate(D))


def build_return_features(df, tw_cols):
    """Canonical 196-feature matrix for the cross-sectional return target."""
    tick = pd.get_dummies(df['ticker'], prefix='tk').astype(float)
    inter = pd.DataFrame({f'{tc}_x_{t}': df[tc].to_numpy() * tick[t].to_numpy()
                          for tc in tw_cols for t in tick.columns})
    MKT = ['rel_mom_5', 'rel_mom_10', 'rel_mom_20', 'rel_mom_60', 'rel_vol_20']
    FEAT = pd.concat([inter, tick, df[SENTF], df[tw_cols], df[MKT]], axis=1)
    return FEAT.fillna(0.0).to_numpy(dtype=float), inter.shape[1], tick.shape[1], MKT


# =========================================================== main analysis --
def main():
    os.makedirs(OUT, exist_ok=True)
    d = load()
    sp = d['speeches'].copy()
    sp['date'] = pd.to_datetime(sp['date'], errors='coerce')
    sp = sp.dropna(subset=['date'])
    mk = d['market'].copy()
    mk['date'] = pd.to_datetime(mk['date'])
    mk = mk.sort_values(['ticker', 'date'])
    tw = d['topics'].pivot_table(index='speech_id', columns='topic_id',
                                values='probability', fill_value=0.0)
    tw.columns = [f'topic_{c}' for c in tw.columns]
    TOPS = list(tw.columns)
    S = Series(mk)

    hdr('TABLE I  Corpus')
    print(d['by_src'].to_string(index=False))
    print(f"documents total            : {d['total']}")
    print(f"  with parseable date      : {d['dated']}")
    _cb = d['by_src'].loc[d['by_src'].source.isin(['Fed', 'ECB']), 'n'].sum()
    print(f"  central-bank share       : {_cb / d['total'] * 100:.1f}%")
    print(f"market rows (with returns) : {len(mk)}   instruments: {mk.ticker.nunique()}")
    print(f"event observations         : {len(d['impact'])}")
    print(f"sentiment scores           : {len(d['sent'])}")
    print(f"topic assignments (pooled) : {d['td_comb']}   (all models: {d['td_all']})")
    print(f"regime classifications     : {d['n_regime']}   VIX rows: {d['n_vix']}")
    print(f"LLM signals                : {d['n_llm']} across {d['n_llm_sp']} documents")
    print(f"span                       : {sp.date.min().date()} .. {sp.date.max().date()}")

    df0 = (d['impact'].merge(sp, left_on='speech_id', right_on='id')
                      .merge(d['sent'], on='speech_id', how='left')
                      .merge(tw, left_on='speech_id', right_index=True, how='left'))
    df0 = df0.dropna(subset=['compound', 'return_t5']).copy()

    # ---------------------------------- canonical 5-day cross-sectional ----
    df = df0.copy()
    df['med'] = df.groupby('speech_id')['return_t5'].transform('median')
    df['y_rel'] = (df['return_t5'] > df['med']).astype(int)
    df['y_abs'] = (df['return_t5'] > 0).astype(int)
    df = df[df['return_t5'] != df['med']].copy()
    for n in (5, 10, 20, 60):
        df[f'mom_{n}'] = [S.past(t, dt, n) for t, dt in zip(df['ticker'], df['date'])]
        df[f'rel_mom_{n}'] = df[f'mom_{n}'] - df.groupby('speech_id')[f'mom_{n}'].transform('mean')
    df['vol_20'] = [S.past(t, dt, 20, vol=True) for t, dt in zip(df['ticker'], df['date'])]
    df['rel_vol_20'] = df['vol_20'] - df.groupby('speech_id')['vol_20'].transform('mean')
    MKT = ['rel_mom_5', 'rel_mom_10', 'rel_mom_20', 'rel_mom_60', 'rel_vol_20']
    df = df.dropna(subset=MKT).sort_values('date').reset_index(drop=True)

    F, n_int, n_tk, MKT = build_return_features(df, TOPS)
    dates = df['date'].to_numpy()
    y_rel, y_abs = df['y_rel'].to_numpy(), df['y_abs'].to_numpy()
    print(f"\nanalysis set: rows={len(df)}  documents={df.speech_id.nunique()}  "
          f"event dates={df.date.nunique()}  features={F.shape[1]}")
    print(f"  = {len(SENTF)} sent + {len(TOPS)} topic + {n_int} interaction "
          f"+ {n_tk} instrument + {len(MKT)} market")

    hdr('TABLE II  Cross-sectional 5-day target (baseline exactly 50.0%)')
    for nm, fn in (('Logistic (C=0.05)', LOG05), ('Logistic (C=1)', LOG1),
                   ('Gradient boosting', GBM)):
        p, q, a, dd = walk_forward(F, y_rel, dates, fn)
        h, se, nc = summarize(clustered(p == a, dd))
        print(f"  {nm:20s} hit={h*100:6.2f}%  SE={se*100:4.2f}pp  "
              f"z={(h-.5)/se:+5.2f}   n={len(p)}  clusters={nc}")

    print('\n  Feature-group ablation (logistic C=1):')
    groups = {
        'rhetoric only (sent+topics)': list(range(n_int + n_tk, F.shape[1] - len(MKT))),
        'topic x instrument only': list(range(n_int)),
        'market controls only': list(range(F.shape[1] - len(MKT), F.shape[1])),
    }
    for gn, cols in groups.items():
        p, q, a, dd = walk_forward(F[:, cols], y_rel, dates, LOG1)
        h, se, _ = summarize(clustered(p == a, dd))
        print(f"    {gn:30s} hit={h*100:6.2f}%  z={(h-.5)/se:+5.2f}")

    hdr('Absolute-direction 5-day target (baseline = majority class)')
    p, q, a, dd = walk_forward(F, y_abs, dates, GBM)
    h, se, _ = summarize(clustered(p == a, dd))
    base_abs = max(a.mean(), 1 - a.mean())
    print(f"  Gradient boosting  hit={h*100:.2f}%  baseline={base_abs*100:.2f}%  "
          f"edge={(h-base_abs)*100:+.2f}pp  z_vs_baseline={(h-base_abs)/se:+.2f}")

    # ------------------------------------------- TABLE III: horizon sweep --
    hdr('TABLE III  Horizon sweep -- does rhetoric act with a lag?')
    print(f"{'h':>5} {'raw':>8} {'baseline':>9} {'edge':>8} {'z':>7} | "
          f"{'x-sec':>8} {'z':>7} | {'rhet-only':>10} {'z':>7}")
    hz_rows = []
    for H in HORIZONS:
        e = df0.copy()
        e['fwd'] = [S.fwd(t, dt, H) for t, dt in zip(e['ticker'], e['date'])]
        e = e.dropna(subset=['fwd']).copy()
        for n in (5, 20):
            e[f'mom_{n}'] = [S.past(t, dt, n) for t, dt in zip(e['ticker'], e['date'])]
        e = e.dropna(subset=['mom_5', 'mom_20'])
        if len(e) < 2000:
            continue
        # absolute target
        ea = e.sort_values('date').reset_index(drop=True)
        tick = pd.get_dummies(ea['ticker'], prefix='tk').astype(float)
        Fa = pd.concat([ea[SENTF], ea[TOPS], tick, ea[['mom_5', 'mom_20']]],
                       axis=1).fillna(0).to_numpy(float)
        ya = (ea['fwd'] > 0).astype(int).to_numpy()
        p, q, a, dd = walk_forward(Fa, ya, ea['date'].to_numpy(), LOG05)
        raw, se, _ = summarize(clustered(p == a, dd))
        bl = max(a.mean(), 1 - a.mean())
        # cross-sectional target
        ex = e.copy()
        ex['med'] = ex.groupby('speech_id')['fwd'].transform('median')
        ex = ex[ex['fwd'] != ex['med']].sort_values('date').reset_index(drop=True)
        for n in (5, 20):
            ex[f'rel_{n}'] = ex[f'mom_{n}'] - ex.groupby('speech_id')[f'mom_{n}'].transform('mean')
        tickx = pd.get_dummies(ex['ticker'], prefix='tk').astype(float)
        Fx = pd.concat([ex[SENTF], ex[TOPS], tickx, ex[['rel_5', 'rel_20']]],
                       axis=1).fillna(0).to_numpy(float)
        yx = (ex['fwd'] > ex['med']).astype(int).to_numpy()
        p2, _, a2, d2 = walk_forward(Fx, yx, ex['date'].to_numpy(), LOG05)
        xs, sx, _ = summarize(clustered(p2 == a2, d2))
        # rhetoric only, cross-sectional
        Fr = ex[SENTF + TOPS].fillna(0).to_numpy(float)
        p3, _, a3, d3 = walk_forward(Fr, yx, ex['date'].to_numpy(), LOG05)
        rh, sr, _ = summarize(clustered(p3 == a3, d3))
        hz_rows.append((H, raw, bl, xs, rh))
        print(f"{H:>5} {raw*100:7.2f}% {bl*100:8.2f}% {(raw-bl)*100:+7.2f}pp "
              f"{(raw-bl)/se:+7.2f} | {xs*100:7.2f}% {(xs-.5)/sx:+7.2f} | "
              f"{rh*100:9.2f}% {(rh-.5)/sr:+7.2f}")

    # ------------------------------------------- TABLE IV: volatility -----
    hdr('TABLE IV  Volatility direction -- the one predictable target')
    v = d['vix'].copy()
    v.columns = ['date', 'vix']
    v['date'] = pd.to_datetime(v['date'], errors='coerce')
    v = v.dropna().drop_duplicates('date').sort_values('date').reset_index(drop=True)
    v['vix_chg5'] = v['vix'].pct_change(5)
    v['vix_z'] = (v['vix'] - v['vix'].rolling(60).mean()) / v['vix'].rolling(60).std()
    vd = v['date'].to_numpy()

    def vix_at(ed):
        p = np.searchsorted(vd, np.datetime64(ed), 'left')
        if p == 0:
            return (np.nan,) * 3
        r = v.iloc[p - 1]
        return (r['vix'], r['vix_chg5'], r['vix_z'])

    VIXF = ['vix', 'vix_chg5', 'vix_z']
    vol_rows = []
    for H in VOL_HORIZONS:
        e = df0.copy()
        e['vp'] = [S.past(t, dt, H, vol=True) for t, dt in zip(e['ticker'], e['date'])]
        e['vp2'] = [S.past(t, dt, H * 2, vol=True) for t, dt in zip(e['ticker'], e['date'])]
        e['vph'] = [S.past(t, dt, max(5, H // 2), vol=True) for t, dt in zip(e['ticker'], e['date'])]
        e['vf'] = [S.fwd_vol(t, dt, H) for t, dt in zip(e['ticker'], e['date'])]
        e['mm'] = [S.past(t, dt, H) for t, dt in zip(e['ticker'], e['date'])]
        e[VIXF] = pd.DataFrame([vix_at(dt) for dt in e['date']], index=e.index)
        e = e.dropna(subset=['vp', 'vp2', 'vph', 'vf', 'mm'] + VIXF)
        e = e.sort_values('date').reset_index(drop=True)
        tick = pd.get_dummies(e['ticker'], prefix='tk').astype(float)
        MKv = e[['vp', 'vp2', 'vph', 'mm']]
        y = (e['vf'] > e['vp']).astype(int).to_numpy()
        dts = e['date'].to_numpy()
        maj = max(y.mean(), 1 - y.mean())
        sets = {
            'market only': pd.concat([MKv, tick], axis=1),
            'market + VIX': pd.concat([MKv, tick, e[VIXF]], axis=1),
            'market + VIX + rhetoric': pd.concat([MKv, tick, e[VIXF], e[SENTF], e[TOPS]], axis=1),
            'rhetoric only': pd.concat([e[SENTF], e[TOPS]], axis=1),
        }
        got = {}
        for tag, FF in sets.items():
            p, q, a, dd = walk_forward(FF.fillna(0).to_numpy(float), y, dts, GBM)
            got[tag] = clustered(p == a, dd)
            h, se, _ = summarize(got[tag])
            print(f"  H={H:>2}d  {tag:24s} hit={h*100:6.2f}%  majority={maj*100:5.2f}%  "
                  f"edge={(h-maj)*100:+6.2f}pp  z={(h-maj)/se:+6.2f}")
        _, pv_vix = stats.ttest_rel(got['market + VIX'], got['market only'])
        _, pv_rh = stats.ttest_rel(got['market + VIX + rhetoric'], got['market + VIX'])
        inc_vix = (got['market + VIX'].mean() - got['market only'].mean()) * 100
        inc_rh = (got['market + VIX + rhetoric'].mean() - got['market + VIX'].mean()) * 100
        print(f"        VIX increment      {inc_vix:+5.2f}pp  p={pv_vix:.4f}")
        print(f"        rhetoric increment {inc_rh:+5.2f}pp  p={pv_rh:.4f}")
        vol_rows.append((H, got['market only'].mean(), maj, inc_rh, pv_rh))
        print()

    # ------------------------------------------- TABLE V: inflation -------
    hdr('TABLE V  Accuracy obtainable without predictive skill')
    sub = df0.drop_duplicates(subset=['speech_id', 'ticker'])
    r252 = np.array([S.fwd(t, dt, 252) for t, dt in zip(sub['ticker'], sub['date'])])
    r252 = r252[~np.isnan(r252)]
    p252 = (r252 > 0).mean()
    print(f"  [1] horizon 252d           : {max(p252, 1-p252)*100:.1f}%  (baseline identical)")
    rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    rf.fit(F, y_rel)
    print(f"  [2] random forest in-sample: {(rf.predict(F)==y_rel).mean()*100:.1f}%")
    split = int(len(df) * 0.7)
    leak = np.column_stack([F, df['abnormal_return'].fillna(0).to_numpy()])
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    m.fit(leak[:split], y_rel[:split])
    print(f"  [3] + post-event variable  : "
          f"{(m.predict(leak[split:])==y_rel[split:]).mean()*100:.1f}%")
    F15 = df[SENTF + TOPS].fillna(0).to_numpy(float)
    out15 = {}
    for tag, yy in (('absolute', y_abs), ('cross-sectional', y_rel)):
        mm = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        mm.fit(F15[:split], yy[:split])
        qq = mm.predict_proba(F15[split:])[:, 1]
        aa = yy[split:]
        k = max(20, int(len(qq) * 0.01))
        idx = np.argsort(-np.abs(qq - .5))[:k]
        out15[tag] = (((qq[idx] > .5).astype(int) == aa[idx]).mean(), k)
    print(f"  [4] top 1% most-confident  : {out15['absolute'][0]*100:.1f}% (absolute) vs "
          f"{out15['cross-sectional'][0]*100:.1f}% (cross-sectional), n={out15['absolute'][1]}"
          f" -- sign is specification-dependent")

    # ---------------------------------------- selective prediction curve --
    hdr('Selective prediction (logistic C=0.05, cross-sectional 5d)')
    p, q, a, dd = walk_forward(F, y_rel, dates, LOG05)
    conf = np.abs(q - 0.5)
    cov_pts = []
    for c in (1.0, 0.5, 0.25, 0.10, 0.05, 0.02):
        k = max(30, int(len(q) * c))
        idx = np.argsort(-conf)[:k]
        pr = (q[idx] > 0.5).astype(int)
        h, se, _ = summarize(clustered(pr == a[idx], dd[idx]))
        cov_pts.append((c, k, h, se))
        print(f"  coverage {c*100:5.1f}%  n={k:6d}  hit={h*100:5.1f}%  "
              f"SE={se*100:4.2f}pp  z={(h-.5)/se:+5.2f}")

    # -------------------------------------------- pseudo-replication -----
    hdr('Pseudo-replication and lookahead')
    piv = d['impact'].dropna(subset=['abnormal_return']).pivot_table(
        index='speech_id', columns='ticker', values='abnormal_return')
    cm = piv.corr().values
    iu = np.triu_indices_from(cm, 1)
    rho, k = float(np.nanmean(cm[iu])), piv.shape[1]
    deff = 1 + (k - 1) * rho
    print(f"  instruments={k}  mean pairwise rho={rho:.3f}  "
          f"design effect={deff:.1f}x  SE inflation={np.sqrt(deff):.2f}x")
    # effective sample size the paper quotes for the cross-sectional test
    print(f"  effective n (10099/deff) = {round(10099 / deff, -2):.0f}")
    ii = d['impact'].dropna(subset=['abnormal_return', 'return_t5'])
    flip = (np.sign(ii.return_t5) != np.sign(ii.abnormal_return)).mean()
    print(f"  full-sample abnormal baseline flips sign of {flip*100:.1f}% of events")

    _figures(hz_rows, cov_pts, vol_rows)
    print(f"\nfigures written: {OUT}/fig_horizon.{FIG_EXT}, "
          f"{OUT}/fig_coverage.{FIG_EXT}, {OUT}/fig_volatility.{FIG_EXT} "
          f"({FIG_DPI} dpi)")
    print("Note: Granger/FDR results come from src/models/causal_validation.py.")
    print("Note: FinBERT truncation coverage -- run with --tokens.")


def _figures(hz_rows, cov_pts, vol_rows):
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False})

    # Fig 1 -- the horizon trap: raw rises, baseline rises faster
    hs = [r[0] for r in hz_rows]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    x = range(len(hs))
    ax.plot(x, [r[2] * 100 for r in hz_rows], 'o-', color='#b3402f', lw=1.7, ms=4,
            label='always-up baseline')
    ax.plot(x, [r[1] * 100 for r in hz_rows], 's--', color='#2f6fb3', lw=1.5, ms=4,
            label='fitted model')
    ax.plot(x, [r[4] * 100 for r in hz_rows], '^:', color='#6a6a6a', lw=1.3, ms=4,
            label='rhetoric only')
    ax.axhline(50, ls=':', color='#999', lw=1)
    ax.set_xticks(list(x)); ax.set_xticklabels([str(h) for h in hs])
    ax.set_xlabel('Forecast horizon (trading days)')
    ax.set_ylabel('Accuracy (%)')
    ax.legend(frameon=False, fontsize=7, loc='upper left')
    fig.tight_layout(pad=0.3)
    fig.savefig(f'{OUT}/fig_horizon.{FIG_EXT}', dpi=FIG_DPI); plt.close(fig)

    # Fig 2 -- coverage curve
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    xs = [c * 100 for c, _, _, _ in cov_pts]
    ys = [h * 100 for _, _, h, _ in cov_pts]
    es = [1.96 * se * 100 for _, _, _, se in cov_pts]
    ax.errorbar(xs, ys, yerr=es, fmt='o-', color='#2f6fb3', lw=1.6, ms=4,
                capsize=2.5, elinewidth=1)
    ax.axhline(50, ls=':', color='#666', lw=1, label='baseline 50%')
    ax.set_xscale('log'); ax.set_xticks(xs)
    ax.set_xticklabels([f'{v:g}' for v in xs]); ax.invert_xaxis()
    ax.set_xlabel('Coverage retained (%, log scale)')
    ax.set_ylabel('Hit rate (%)')
    ax.legend(frameon=False, fontsize=7.5, loc='upper left')
    fig.tight_layout(pad=0.3)
    fig.savefig(f'{OUT}/fig_coverage.{FIG_EXT}', dpi=FIG_DPI); plt.close(fig)

    # Fig 3 -- volatility: real edge, and rhetoric's null contribution
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    vh = [str(r[0]) + 'd' for r in vol_rows]
    xv = np.arange(len(vol_rows))
    ax.bar(xv - 0.19, [r[1] * 100 for r in vol_rows], 0.38, color='#2f6fb3',
           label='model')
    ax.bar(xv + 0.19, [r[2] * 100 for r in vol_rows], 0.38, color='#c9c9c9',
           label='majority baseline')
    for i, r in enumerate(vol_rows):
        ax.text(i - 0.19, r[1] * 100 + 1.0, f'{r[1]*100:.0f}', ha='center', fontsize=7)
    ax.set_xticks(xv); ax.set_xticklabels(vh)
    ax.set_ylim(0, 85)
    ax.set_xlabel('Volatility horizon')
    ax.set_ylabel('Accuracy (%)')
    ax.legend(frameon=False, fontsize=7, loc='upper right')
    fig.tight_layout(pad=0.3)
    fig.savefig(f'{OUT}/fig_volatility.{FIG_EXT}', dpi=FIG_DPI); plt.close(fig)


def token_coverage():
    """How much of each document FinBERT actually reads. Behind a flag
    because it pulls in transformers. An earlier draft estimated 'about 32%'
    from a character heuristic; the real Mann ki Baat figure is 10.7%."""
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
    if '--tokens' in sys.argv:
        token_coverage()
    else:
        main()
