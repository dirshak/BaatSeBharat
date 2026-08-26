"""
build_overleaf_bundle.py
========================
Produces a self-contained folder (and .zip) ready to drag into Overleaf.

    python scripts/build_overleaf_bundle.py           # regenerate figures, then bundle
    python scripts/build_overleaf_bundle.py --fast    # reuse existing figures
    python scripts/build_overleaf_bundle.py --no-zip  # folder only

Output layout (overleaf_bundle/):

    main.tex              <- paper_conference.tex, renamed
    figs/fig_horizon.pdf
    figs/fig_coverage.pdf
    figs/fig_volatility.pdf
    README.txt

Why `main.tex`: Overleaf picks the main document automatically when a zip
contains exactly one .tex file, and `main.tex` is the name it defaults to.
This is NOT the repository's main.tex, which is the superseded proposal
draft -- the bundle deliberately contains only the conference paper so the
two cannot be confused.

Which figures get copied is parsed out of the .tex itself rather than
hardcoded, so adding or removing a figure in the paper needs no change
here. Anything the paper references but that does not exist on disk is a
hard error, because the failure mode otherwise is an Overleaf build that
silently drops a figure.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX_SRC = os.path.join(ROOT, 'paper_conference.tex')
BUNDLE = os.path.join(ROOT, 'overleaf_bundle')
ZIP_PATH = os.path.join(ROOT, 'overleaf_bundle.zip')
REPRO = os.path.join(ROOT, 'scripts', 'reproduce_results.py')

BS = chr(92)
INCLUDE_RE = re.compile(BS + BS + r'includegraphics[^{]*\{([^}]*)\}')

# An unescaped % starts a LaTeX comment. Comments must be stripped before
# scanning for figures: the pattern above allows any run of non-brace
# characters between \includegraphics and its argument, so the word
# "\includegraphics" appearing inside a comment swallows everything up to
# the next brace anywhere in the file. That produced a build failure
# reporting a missing figure named "IEEEtran", picked up from
# \documentclass{IEEEtran} several lines below a comment that merely
# mentioned the command.
COMMENT_RE = re.compile(r'(?<!' + BS + BS + r')%.*?$', re.M)


def strip_comments(tex):
    return COMMENT_RE.sub('', tex)

README = """\
BaatSeBharat -- conference paper bundle
=======================================

Upload overleaf_bundle.zip to Overleaf (New Project -> Upload Project).

  main.tex   the paper (IEEEtran, conference option, SINGLE column)
  figs/      the figures it references

Overleaf will select main.tex automatically -- it is the only .tex here.
If it does not, use Menu -> Main document -> main.tex.

Compiler: pdfLaTeX (Overleaf's default). IEEEtran.cls and the multirow
package both ship with Overleaf, so nothing extra needs installing.

Every number in the paper is emitted by scripts/reproduce_results.py in the
source repository. Do not hand-edit figures here; regenerate and rebuild:

    python scripts/build_overleaf_bundle.py
"""


def regenerate_figures():
    print('regenerating figures via reproduce_results.py ...')
    r = subprocess.run([sys.executable, REPRO], cwd=ROOT,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise SystemExit('figure regeneration FAILED -- bundle not built')
    tail = [ln for ln in r.stdout.splitlines() if 'figures written' in ln]
    print('  ' + (tail[0] if tail else 'done'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fast', action='store_true',
                    help='reuse figures already in figs/ instead of regenerating')
    ap.add_argument('--no-zip', action='store_true', help='build the folder only')
    args = ap.parse_args()

    if not os.path.isfile(TEX_SRC):
        raise SystemExit(f'not found: {TEX_SRC}')

    if args.fast:
        print('--fast: reusing existing figures (not regenerated)')
    else:
        regenerate_figures()

    tex = open(TEX_SRC, encoding='utf-8').read()
    figs = INCLUDE_RE.findall(strip_comments(tex))
    if not figs:
        raise SystemExit('no \\includegraphics found in the paper -- nothing to bundle')

    # resolve each reference, allowing a missing .pdf extension
    resolved, missing = [], []
    for f in figs:
        cands = [f, f + '.pdf', f + '.png']
        hit = next((c for c in cands if os.path.isfile(os.path.join(ROOT, c))), None)
        (resolved.append((f, hit)) if hit else missing.append(f))
    if missing:
        for m in missing:
            print(f'  MISSING: {m}')
        raise SystemExit('paper references figures that do not exist -- fix before bundling')

    if os.path.isdir(BUNDLE):
        shutil.rmtree(BUNDLE)
    os.makedirs(BUNDLE)

    shutil.copy2(TEX_SRC, os.path.join(BUNDLE, 'main.tex'))
    print(f'\n  main.tex            <- {os.path.basename(TEX_SRC)}')

    for ref, path in resolved:
        # keep the reference path intact so \includegraphics needs no edit
        dest = os.path.join(BUNDLE, ref if ref.lower().endswith(('.pdf', '.png'))
                            else ref + os.path.splitext(path)[1])
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(os.path.join(ROOT, path), dest)
        print(f'  {os.path.relpath(dest, BUNDLE):22s}<- {path}')

    with open(os.path.join(BUNDLE, 'README.txt'), 'w', encoding='utf-8') as fh:
        fh.write(README)
    print('  README.txt')

    # verify: every reference now resolves relative to the bundle root
    bad = []
    for ref, _ in resolved:
        if not any(os.path.isfile(os.path.join(BUNDLE, ref + e)) for e in ('', '.pdf', '.png')):
            bad.append(ref)
    if bad:
        raise SystemExit(f'bundle is inconsistent, unresolved: {bad}')
    print(f'\nverified: all {len(resolved)} figure reference(s) resolve inside the bundle')

    if not args.no_zip:
        if os.path.isfile(ZIP_PATH):
            os.remove(ZIP_PATH)
        with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(BUNDLE):
                for fn in files:
                    full = os.path.join(root, fn)
                    z.write(full, os.path.relpath(full, BUNDLE))
        size = os.path.getsize(ZIP_PATH) / 1024
        print(f'zip written: {os.path.relpath(ZIP_PATH, ROOT)}  ({size:.0f} KB)')

    print('\nUpload to Overleaf: New Project -> Upload Project -> overleaf_bundle.zip')


if __name__ == '__main__':
    main()
