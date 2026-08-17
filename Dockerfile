# Reproducible environment for the BaatSeBharat data pipeline:
#   scripts/run_prototype.py       (scrape -> FinBERT -> NMF topic models ->
#                                    HMM regimes -> speech-vs-market impact ->
#                                    Granger causality)
#   scripts/classify_speeches_groq.py  (optional Groq LLM topic->company
#                                        classification, needs GROQ_API_KEY)
#   scripts/export_static_data.py  (snapshots the above as static JSON for
#                                    the frontend)
#
# This image exists so a local `docker build && docker run` reproduces
# EXACTLY the same environment .github/workflows/pipeline.yml runs in --
# it is not used to host anything live. The production frontend is a
# static site (see README.md); this container never serves traffic.
#
# The Actions workflow mounts the live checkout over /app at run time, so
# the image only needs to provide the installed Python environment -- the
# code that actually executes is always whatever commit triggered the run,
# never a stale COPY baked in at image-build time.

FROM python:3.12.6-slim

WORKDIR /app

# build-essential: required for a couple of source-built wheels pulled in
# by scipy/scikit-learn/hmmlearn on some platforms.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# --extra-index-url pulls CPU-only torch wheels (the pipeline never needs
# a GPU and the default PyPI torch wheel is several GB larger).
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# spaCy's English model is a separate download, not bundled with the
# spacy package itself -- src/features/text_preprocessing.py loads
# en_core_web_sm by name and fails at runtime (OSError: [E050]) without it.
RUN python -m spacy download en_core_web_sm

# src/data/centralized_scraper.py drives a headless Chromium via Playwright
# to scrape speech transcripts -- `playwright install` pulls the browser
# binary itself (not on PyPI), --with-deps adds the OS-level libraries
# (fonts, codecs) Chromium needs on a bare Debian slim image.
RUN playwright install --with-deps chromium

# FinBERT weights (ProsusAI/finbert, ~440MB) are fetched from the Hugging
# Face Hub on first use by src/models/sentiment_overlay.py and cached
# under /root/.cache/huggingface inside the container -- not baked into
# this image, to keep image builds fast and avoid duplicating weights in
# every layer. Mount that cache dir as a volume for faster repeat local
# runs if desired: -v bsb-hf-cache:/root/.cache/huggingface

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["sh", "-c", "python scripts/run_prototype.py && python scripts/export_static_data.py"]
