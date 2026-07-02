import streamlit as st
import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch

class AdvancedNLPEngine:
    """State-of-the-art NLP processing using BERTopic and FinBERT."""
    
    def __init__(self):
        # Using a lightweight transformer for embeddings
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.sentiment_pipe = None # Lazy load
        self.topic_model = None

    def get_sentiment_pipeline(self):
        """Lazy load FinBERT to save memory until needed."""
        if self.sentiment_pipe is None:
            # ProsusAI/finbert is the standard for financial sentiment
            self.sentiment_pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        return self.sentiment_pipe

    def analyze_sentiment(self, texts):
        """Batch process texts for FinBERT sentiment."""
        pipe = self.get_sentiment_pipeline()
        # Process in chunks to avoid OOM
        results = []
        for i in range(0, len(texts), 5):
            chunk = [t[:512] for t in texts[i:i+5]] # Truncate to BERT max length
            results.extend(pipe(chunk))
        return results

    def fit_bertopic(self, texts):
        """Trains BERTopic for contextual topic modeling."""
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            nr_topics="auto",
            calculate_probabilities=True,
            verbose=True
        )
        topics, probs = self.topic_model.fit_transform(texts)
        return topics, probs, self.topic_model.get_topic_info()

    def get_topic_hierarchy(self):
        """Generates a hierarchical tree of topics for deeper understanding."""
        if self.topic_model:
            return self.topic_model.hierarchical_topics(self.topic_model.documents)
        return None
