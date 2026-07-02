"""
Time-decay + importance weighting for speech-derived topic/sentiment signals
used by the prediction engine.

    final_weight(speech) = decay(age_days) * importance(speech)

decay(age_days) = 0.5 ** (age_days / half_life_days)
    Standard exponential half-life decay. `half_life_days` is a tunable
    config value (config/config.yaml -> models.speech_weighting), not a
    hardcoded constant -- e.g. half_life_days=180 means a speech's decay
    factor drops to 0.5 after 6 months, 0.25 after a year, etc.

importance(speech) = importance_floor + (1 - importance_floor) * (
        topic_weight * topic_concentration + reaction_weight * market_reaction
    )
    Two signals, blended (weights configurable, default 0.5/0.5):

    - topic_concentration in [0, 1]: how "on-message" the speech was.
      A speech whose topic distribution is concentrated on one dominant
      theme (e.g. one topic at 0.8 probability) is more likely to be a
      deliberate policy-signaling speech than one that's diffuse across
      many topics. Computed as the max per-topic probability, rescaled so
      a perfectly uniform distribution (1/n_topics) maps to 0 and a
      fully concentrated one (1.0) maps to 1.
    - market_reaction in [0, 1]: |abnormal_return| observed after the
      speech, capped at `reaction_cap` (default 5%) and rescaled to [0, 1].
      A speech that historically preceded a large abnormal market move is
      "important" regardless of age.

    `importance_floor` (default 0.3) guarantees every speech retains at
    least 30% of its topic weight even with zero concentration/reaction,
    so decay alone still dominates for genuinely unremarkable speeches,
    while a high-importance old speech (concentrated + large historical
    reaction) can still reach importance = 1.0 and resist decay.

This module is pure computation over a DataFrame; it doesn't know about
SQL or the DB schema, so it's usable both by the training/backfill scripts
and by prediction_engine.py at request time.
"""

import os

import numpy as np
import pandas as pd
import yaml

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'config', 'config.yaml'
)


def load_weighting_config(path=DEFAULT_CONFIG_PATH):
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg.get('models', {}).get('speech_weighting', {}) or {}


def decay_factor(age_days, half_life_days):
    age_days = np.asarray(age_days, dtype=float)
    half_life_days = max(float(half_life_days), 1.0)
    return np.power(0.5, age_days / half_life_days)


def topic_concentration(max_topic_prob, n_topics):
    """Rescale max per-topic probability so uniform (1/n_topics) -> 0 and
    fully concentrated (1.0) -> 1."""
    n_topics = max(int(n_topics), 2)
    baseline = 1.0 / n_topics
    span = 1.0 - baseline
    if span <= 0:
        return np.zeros_like(np.asarray(max_topic_prob, dtype=float))
    return np.clip((np.asarray(max_topic_prob, dtype=float) - baseline) / span, 0.0, 1.0)


def market_reaction_score(abnormal_return, reaction_cap):
    reaction_cap = max(float(reaction_cap), 1e-6)
    return np.clip(np.abs(np.asarray(abnormal_return, dtype=float)) / reaction_cap, 0.0, 1.0)


def importance_score(topic_conc, reaction, cfg=None):
    cfg = cfg or load_weighting_config()
    floor = float(cfg.get('importance_floor', 0.3))
    tw = float(cfg.get('topic_weight', 0.5))
    rw = float(cfg.get('reaction_weight', 0.5))
    blend = tw * np.nan_to_num(topic_conc) + rw * np.nan_to_num(reaction)
    return floor + (1.0 - floor) * np.clip(blend, 0.0, 1.0)


def compute_speech_weights(df, n_topics, as_of=None, cfg=None):
    """
    df must have columns: 'date' (datetime-like), 'max_topic_prob',
    'abnormal_return' (may contain NaN).
    Returns df with added columns: decay, topic_conc, reaction, importance, weight.
    """
    cfg = cfg or load_weighting_config()
    as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now()

    out = df.copy()
    dates = pd.to_datetime(out['date'], errors='coerce')
    age_days = (as_of - dates).dt.days.clip(lower=0).fillna(9999)

    out['decay'] = decay_factor(age_days, cfg.get('half_life_days', 180))
    out['topic_conc'] = topic_concentration(out['max_topic_prob'], n_topics)
    out['reaction'] = market_reaction_score(
        out['abnormal_return'].fillna(0.0), cfg.get('reaction_cap', 0.05)
    )
    out['importance'] = importance_score(out['topic_conc'], out['reaction'], cfg)
    out['weight'] = out['decay'] * out['importance']
    return out
