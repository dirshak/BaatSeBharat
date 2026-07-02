"""
Topic modeling for leadership-speech rhetoric.

Approach: TF-IDF + NMF (Non-negative Matrix Factorization), not the previous
LDA/NMF/BERTopic ensemble.

Why NMF over the BERTopic-centric ensemble it replaces: this corpus is small
(~1.3k speeches after dedup, ~70-750 per source). BERTopic's UMAP+HDBSCAN
stage needs a much larger, denser embedding space to form stable clusters --
on a corpus this size it is highly sensitive to random seed/parameters and
tends to either collapse everything into one giant cluster or explode into
dozens of one-document "micro-topics" (exactly what the previous
`min_cluster_size=5, n_neighbors=5` config was doing). NMF over TF-IDF is
the standard, well-understood choice at this corpus size: deterministic
given a fixed seed, fast (no GPU/embedding step required), and produces
non-negative topic weights that map directly onto this project's existing
"topic strength" semantics (every downstream consumer already treats topic
weight as a probability-like [0,1] strength, which is exactly NMF's native
output after row-normalization).

Two-tier stopword handling (config/rhetoric_stopwords.yaml):
  - `universal`: hard-excluded from the vectorizer vocabulary (honorifics,
    greetings, stock political phrases -- carry no market signal in any
    context).
  - `contextual`: NOT hard-excluded (e.g. "bank", "inflation" are generic
    filler in a sign-off but load-bearing in a Monetary Policy topic).
    These are naturally down-weighted by TF-IDF's own idf term and by
    capping the vectorizer's `max_df` at `auto_detect_threshold`, so a term
    that appears in nearly every document (regardless of tier) is
    automatically excluded -- this is the "programmatic" half of the
    requirement: high corpus-wide document frequency is itself evidence of
    boilerplate, independent of any hand-curated list.

Topic labeling is deterministic: top TF-IDF-weighted keywords per topic are
scored against a curated label lexicon (config/rhetoric_stopwords.yaml ->
label_lexicon); the highest-scoring bucket wins. If nothing scores above
`label_min_score`, the label falls back to a Title Case join of the top 3
keywords -- never "Topic N".
"""

import json
import os
import pickle
import sqlite3
import sys

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

DEFAULT_STOPWORDS_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'config', 'rhetoric_stopwords.yaml'
)


def _load_stopwords_config(path=DEFAULT_STOPWORDS_CONFIG):
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg


class RhetoricTopicModel:
    """TF-IDF + NMF topic model with domain stopword filtering and
    deterministic, lexicon-based topic labeling."""

    def __init__(self, n_topics=10, random_state=42, stopwords_config_path=DEFAULT_STOPWORDS_CONFIG):
        self.n_topics = n_topics
        self.random_state = random_state

        cfg = _load_stopwords_config(stopwords_config_path)
        self.universal_stopwords = set(cfg.get('universal', []))
        self.contextual_stopwords = set(cfg.get('contextual', []))
        self.auto_detect_threshold = float(cfg.get('auto_detect_threshold', 0.6))
        self.label_lexicon = cfg.get('label_lexicon', {})
        self.label_min_score = float(cfg.get('label_min_score', 1.0))

        self.vectorizer = None
        self.model = None
        self.topic_dist = None          # (n_docs, n_topics), rows sum to 1
        self.auto_detected_terms = []   # terms excluded purely by high doc-freq

    def _stopword_list(self):
        # Hard-exclude universal terms + sklearn's generic English stopwords.
        # `contextual` terms stay IN the vocabulary; over-common ones are
        # suppressed via max_df instead of being permanently erased.
        return sorted(self.universal_stopwords | set(ENGLISH_STOP_WORDS))

    def fit(self, documents):
        """Fit on a list of preprocessed document strings. Returns the
        (n_docs, n_topics) row-normalized topic distribution."""
        if len(documents) < 2:
            raise ValueError(f"Need at least 2 documents to fit a topic model, got {len(documents)}")

        n_topics = max(2, min(self.n_topics, len(documents) - 1))
        self.n_topics = n_topics

        vectorizer = TfidfVectorizer(
            max_features=4000,
            min_df=2,
            max_df=self.auto_detect_threshold,
            stop_words=self._stopword_list(),
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        tfidf = vectorizer.fit_transform(documents)

        if tfidf.shape[1] == 0:
            raise ValueError("Vocabulary is empty after stopword filtering -- corpus too small or too uniform.")

        # Record which contextual/other terms got dropped purely by max_df
        # (i.e. present in the raw vocabulary scan but excluded from the
        # fitted vectorizer) for auditability.
        raw_vectorizer = TfidfVectorizer(stop_words=self._stopword_list(), min_df=2)
        try:
            raw_vectorizer.fit(documents)
            fitted_vocab = set(vectorizer.get_feature_names_out())
            self.auto_detected_terms = sorted(
                set(raw_vectorizer.get_feature_names_out()) - fitted_vocab
            )[:200]
        except Exception:
            self.auto_detected_terms = []

        model = NMF(
            n_components=n_topics,
            random_state=self.random_state,
            max_iter=500,
            init='nndsvda',
        )
        W = model.fit_transform(tfidf)

        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        topic_dist = W / row_sums

        self.vectorizer = vectorizer
        self.model = model
        self.topic_dist = topic_dist
        return topic_dist

    def top_keywords(self, topic_idx, n=15):
        feature_names = self.vectorizer.get_feature_names_out()
        row = self.model.components_[topic_idx]
        top_idx = row.argsort()[-n:][::-1]
        return [feature_names[i] for i in top_idx]

    def label_topic(self, topic_idx, n_keywords=15):
        keywords = self.top_keywords(topic_idx, n_keywords)
        best_label, best_score = None, 0.0
        for label, lex_terms in self.label_lexicon.items():
            score = 0.0
            for rank, kw in enumerate(keywords):
                weight = 1.0 / (rank + 1)
                if any(kw == term or kw in term or term in kw for term in lex_terms):
                    score += weight
            if score > best_score:
                best_score = score
                best_label = label
        if best_label and best_score >= self.label_min_score:
            return best_label
        return " & ".join(k.title() for k in keywords[:3] if k)

    def get_all_labels(self, n_keywords=15):
        """Return {topic_id: {'label': ..., 'keywords': [...]}} for every topic."""
        labels = {}
        for i in range(self.n_topics):
            labels[str(i)] = {
                'label': self.label_topic(i, n_keywords),
                'keywords': self.top_keywords(i, n_keywords),
            }
        return labels

    def topic_distinctness(self):
        """Average pairwise Jaccard distance between topics' top-10 keyword
        sets. Near 0 => topics are near-duplicates (degenerate model);
        closer to 1 => topics are well-separated."""
        keyword_sets = [set(self.top_keywords(i, 10)) for i in range(self.n_topics)]
        if len(keyword_sets) < 2:
            return 1.0
        distances = []
        for i in range(len(keyword_sets)):
            for j in range(i + 1, len(keyword_sets)):
                a, b = keyword_sets[i], keyword_sets[j]
                union = a | b
                if not union:
                    continue
                jaccard_sim = len(a & b) / len(union)
                distances.append(1.0 - jaccard_sim)
        return float(np.mean(distances)) if distances else 1.0

    def save(self, output_dir='./models/trained', prefix='rhetoric'):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f'{prefix}_nmf_model.pkl'), 'wb') as f:
            pickle.dump(self.model, f)
        with open(os.path.join(output_dir, f'{prefix}_vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        logger.info(f"Saved {prefix} NMF model + vectorizer to {output_dir}")


def train_and_save(name, documents, speech_ids, n_topics=10, output_dir='./data/processed'):
    """Fit a RhetoricTopicModel on `documents`, save the .npy distribution
    array and the topic_labels_<name>.json file. Returns (model, topic_dist)
    or (None, None) if there isn't enough data.
    """
    if len(documents) < 2:
        logger.warning(f"Not enough data for model '{name}' ({len(documents)} documents).")
        return None, None

    model = RhetoricTopicModel(n_topics=n_topics)
    topic_dist = model.fit(documents)

    os.makedirs(output_dir, exist_ok=True)
    slug = name.lower().replace(' ', '_')

    np.save(os.path.join(output_dir, f'topic_distributions_{slug}.npy'), topic_dist)

    labels = model.get_all_labels()
    with open(os.path.join(output_dir, f'topic_labels_{slug}.json'), 'w', encoding='utf-8') as f:
        json.dump(labels, f, indent=2)

    distinctness = model.topic_distinctness()
    logger.info(
        f"✓ {name}: {len(documents)} docs -> {model.n_topics} topics "
        f"(distinctness={distinctness:.3f}, auto-excluded {len(model.auto_detected_terms)} "
        f"high-frequency terms). Labels: {[v['label'] for v in labels.values()]}"
    )

    return model, topic_dist


if __name__ == "__main__":
    conn = sqlite3.connect('./data/market_rhetoric.db')
    df = pd.read_sql_query(
        "SELECT id, processed_text FROM speeches WHERE processed_text IS NOT NULL AND processed_text != ''",
        conn
    )
    conn.close()

    if len(df) > 0:
        train_and_save('Combined', df['processed_text'].tolist(), df['id'].tolist())
    else:
        logger.error("No documents to process. Run preprocessing first.")
