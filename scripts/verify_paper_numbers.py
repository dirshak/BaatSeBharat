"""
verify_paper_numbers.py
=======================
Reproducibility gate: every number stated in paper_conference.tex must be
reproducible from the code in this repository.

    python scripts/verify_paper_numbers.py             # recompute, then check
    python scripts/verify_paper_numbers.py --use-cache # reuse last run's output

How it works. Three producers are run and their stdout concatenated:

    scripts/reproduce_results.py             tables I-V, selective prediction,
                                             pseudo-replication, base rates
    scripts/reproduce_results.py --tokens    FinBERT truncation coverage
    src/models/causal_validation.py          Granger p-values (+ BH FDR here)

Every numeric literal in the paper body (the bibliography is excluded --
it is years, volumes and page ranges) is then required to appear in that
output. Anything that does not is either a transcription error or a value
that legitimately comes from outside this codebase; the latter are listed
explicitly in EXTERNAL below, so the set of unverifiable numbers in the
paper is itself auditable rather than implicit.

Exit code 0 = every paper number is reproducible. Non-zero = drift.
"""
import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, 'paper_conference.tex')
REPRO = os.path.join(ROOT, 'scripts', 'reproduce_results.py')
CACHE = os.path.join(ROOT, 'figs', '_verify_cache.txt')

BS = chr(92)

# Numbers that are NOT outputs of this codebase. Each needs a reason.
EXTERNAL = {
    # --- values quoted from the cited literature ---
    '0.9866': 'Bhandari et al. reported R^2',
    '0.922': 'Khadke et al. CNN R^2',
    '0.88': 'Khadke et al. LSTM R^2',
    '2.36': 'Kumar et al. reported error %',
    '89.6': 'Paul et al. FinBERT accuracy',
    '74': 'Jain et al. reported accuracy / band lower bound',
    '87': 'Biswal et al. reported accuracy / band upper bound',
    '70000': 'Buehlmaier & Whited corpus size (8-K filings)',
    '3044': 'Mishev et al. corpus size',
    '11': 'Srivastava & Mitra event-window length (days)',

    # --- design constants chosen by us, not computed ---
    '196': 'feature count (design)',
    '160': 'topic x instrument interaction count (design)',
    '512': 'FinBERT token limit (model constant)',
    '252': 'trading days per year / horizon label',
    '50': 'cross-sectional baseline, exact by construction',
    '50.0': 'cross-sectional baseline, exact by construction',
    '0.05': 'significance threshold / logistic C',
    '0.95': 'complement of 0.05 in the min-over-lags derivation',
    '1.96': 'normal quantile for 95% intervals',
    '95': '95% interval label',
    '400': 'figure dpi',
    '1': 'lag / horizon / enumeration',
    '2': 'lag / enumeration',
    '3': 'HMM states / enumeration',
    '4': 'enumeration',
    '5': 'lags, folds, sentiment feature count',
    '10': 'topic count K / horizon label',
    '15': 'k-1 in the design-effect formula',
    '16': 'instrument count',
    '20': 'horizon label',
    '30': 'horizon label',
    '60': 'horizon label',
    '120': 'horizon label',
    '125': 'Mann ki Baat episodes numbered to date (external fact)',
    '22': 'missing-episode range start',
    '75': 'missing-episode range end',
    '102': 'missing episode number',
    '40': 'documents sampled per source for the truncation measurement',
    '1000': 'order-of-magnitude corpus size in prose (10^3)',
    '0': 'zero discoveries / enumeration',

    # --- quantities derived in prose from values the code does emit ---
    '0.226': '1 - 0.95^5, the min-over-5-lags null rate (derivation)',
    '9': 'chunk-and-aggregate cost ~ 4804/512 tokens, stated as "roughly 9x"',
    '0.2': 'prose bound: horizon sweep stays within 0.2pp of 50%',
    '0.09': 'width of the rhetoric-only CI, i.e. 50.03 - 49.94 (both emitted)',
}

# Tokens that are section/table/figure cross-references or LaTeX plumbing
SKIP_CONTEXT = re.compile(
    BS + BS + r'(ref|label|cite|includegraphics|documentclass|usepackage|'
    r'IEEEauthorblock\w*|title|author|begin|end|multirow|column|textwidth)')


def run(cmd, label):
    print(f'  running {label} ...', flush=True)
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-1500:])
        print(r.stderr[-1500:])
        raise SystemExit(f'{label} FAILED -- cannot verify')
    return r.stdout


GRANGER_SNIPPET = '''
import warnings, sys, os
warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())
import numpy as np
from src.models.causal_validation import CausalValidator
r = CausalValidator(db_path="./data/market_rhetoric.db").test_causality()
ps = np.array(list(r.values())); m = len(ps)
o = np.argsort(ps)
bh = np.minimum.accumulate((ps[o]*m/np.arange(1, m+1))[::-1])[::-1]
print("topics", m, "raw_hits", int((ps < 0.05).sum()))
print("expected_fp", round(m*(1-0.95**5), 2))
for i, idx in enumerate(o):
    print("p", f"{ps[idx]:.4f}", "q", f"{min(bh[i],1):.4f}")
print("surviving", int((bh < 0.05).sum()))
'''


def collect_outputs(use_cache):
    if use_cache and os.path.isfile(CACHE):
        print(f'  --use-cache: reading {os.path.relpath(CACHE, ROOT)}')
        return open(CACHE, encoding='utf-8').read()
    out = run([sys.executable, REPRO], 'reproduce_results.py')
    out += run([sys.executable, REPRO, '--tokens'], 'reproduce_results.py --tokens')
    out += run([sys.executable, '-c', GRANGER_SNIPPET], 'causal_validation (Granger + FDR)')
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, 'w', encoding='utf-8') as fh:
        fh.write(out)
    return out


def norm(tok):
    """Normalise a numeric literal for comparison: strip LaTeX thousands
    separators and trailing zeros so 1{,}288 == 1288 and 50.0 == 50."""
    t = tok.replace('{,}', '').replace(',', '').lstrip('+')
    if t.startswith('-'):
        t = t[1:]
    if '.' in t:
        t = t.rstrip('0').rstrip('.')
    return t or '0'


# Match a numeric literal, allowing LaTeX thousands separators, but WITHOUT
# swallowing surrounding braces. A looser pattern picks up the digits inside
# \multicolumn{4}{l} and reports them as unverifiable paper claims.
NUM = re.compile(r'\d+(?:\{,\}\d{3}|,\d{3})*(?:\.\d+)?')


def rounds_to(paper_value, computed_values):
    """True if `paper_value` is the correct rounding of some computed value.

    A paper legitimately reports p=0.17 for a computed 0.1694, or "+17pp"
    for 17.07. Demanding exact string equality would flag every such case,
    so match at the precision the paper actually chose: if the paper wrote
    two decimals, round each computed value to two decimals and compare.
    """
    try:
        pv = float(paper_value)
    except ValueError:
        return False
    dp = len(paper_value.split('.')[1]) if '.' in paper_value else 0
    for c in computed_values:
        try:
            if round(float(c), dp) == pv:
                return True
        except ValueError:
            continue
    return False


# Commands whose braces contain identifiers, not claims. These are stripped
# document-wide BEFORE line scanning, because a \cite{...} spanning two
# lines leaves its continuation line with no \cite on it -- so a line-level
# skip silently lets citation keys like "bailey2014" through as if they were
# numeric claims.
STRIP_CMDS = re.compile(
    BS + BS + r'(cite|ref|label|includegraphics|usepackage|documentclass)'
    r'(\[[^\]]*\])?\{[^}]*\}', re.S)


def paper_body(tex):
    """Body only, with identifier-bearing commands removed.

    The bibliography is excluded outright: it is years, volumes and page
    ranges, none of which this codebase produces or should be asked to.
    """
    i = tex.find(BS + 'begin{thebibliography}')
    body = tex[:i] if i > 0 else tex
    return STRIP_CMDS.sub(' ', body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--use-cache', action='store_true')
    args = ap.parse_args()

    print('Collecting computed values')
    out = collect_outputs(args.use_cache)
    computed = {norm(m.group()) for m in NUM.finditer(out)}
    print(f'  {len(computed)} distinct numeric values emitted by the code\n')

    tex = open(TEX, encoding='utf-8').read()
    body = paper_body(tex)

    claimed, unmatched, rounded = set(), [], set()
    for line in body.splitlines():
        if SKIP_CONTEXT.search(line) or line.lstrip().startswith('%'):
            continue
        for m in NUM.finditer(line):
            n = norm(m.group())
            claimed.add(n)
            if n in computed or n in EXTERNAL:
                continue
            if rounds_to(n, computed):
                rounded.add(n)
                continue
            unmatched.append((n, line.strip()[:88]))

    exact = sorted(claimed & computed, key=lambda v: (len(v), v))
    external = sorted(claimed & set(EXTERNAL), key=lambda v: (len(v), v))

    print(f'Paper states {len(claimed)} distinct numeric values')
    print(f'  exact match to code    : {len(exact)}')
    print(f'  correct rounding of it : {len(rounded)}')
    print(f'  declared external      : {len(external)}')
    print(f'  UNACCOUNTED            : {len(set(n for n, _ in unmatched))}')

    if unmatched:
        print('\nNot reproducible and not declared external:')
        seen = set()
        for n, ctx in unmatched:
            if n in seen:
                continue
            seen.add(n)
            print(f'   {n:>12s}   ...{ctx}')
        print('\nFAIL: paper contains numbers the code does not produce.')
        return 1

    print('\nPASS: every number in the paper body is either reproducible '
          'from the code or explicitly declared external.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
