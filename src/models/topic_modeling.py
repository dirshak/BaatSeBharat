from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from bertopic import BERTopic
import numpy as np
import pandas as pd
import sqlite3
import pickle
import sys
import os
import torch
from transformers import pipeline

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class HybridTopicModeler:
    """
    Ensemble topic modeling: LDA + NMF + BERTopic
    """
    
    def __init__(self, n_topics=20):
        self.n_topics = n_topics
        
        # Models
        self.lda_model = None
        self.nmf_model = None
        self.bertopic_model = None
        
        # Vectorizers
        self.count_vectorizer = None
        self.tfidf_vectorizer = None

        # Domain-specific noise reduction
        self.custom_stopwords = [
            "countrymen", "new", "come", "dear", "episode", "mann", "baat", "modi", 
            "namaste", "friends", "brothers", "sisters", "people", "today", "year",
            "country", "nation", "time", "government", "india", "indian", "way"
        ]
    
    def fit_lda(self, documents):
        """Fit LDA model"""
        logger.info("Fitting LDA model (Ensemble Part 1/3)...")
        
        # Create vectorizer
        self.count_vectorizer = CountVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.8,
            stop_words=self.custom_stopwords
        )
        
        doc_term_matrix = self.count_vectorizer.fit_transform(documents)
        
        # Train LDA
        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=42,
            max_iter=50,
            learning_method='batch',
            n_jobs=-1
        )
        
        self.lda_model.fit(doc_term_matrix)
        logger.info("✓ LDA model fitted")
        return self.lda_model.transform(doc_term_matrix)
    
    def fit_nmf(self, documents):
        """Fit NMF model"""
        logger.info("Fitting NMF model (Ensemble Part 2/3)...")
        
        # Create vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.8,
            stop_words=self.custom_stopwords
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
        
        # Train NMF
        self.nmf_model = NMF(
            n_components=self.n_topics,
            random_state=42,
            max_iter=500,
            init='nndsvda' # Using SVD based initialization
        )
        
        self.nmf_model.fit(tfidf_matrix)
        logger.info("NMF model fitted")
        return self.nmf_model.transform(tfidf_matrix)
    
    def fit_bertopic(self, documents, embeddings):
        """Fit BERTopic model"""
        logger.info("Fitting BERTopic model (Ensemble Part 3/3)...")
        
        # Enhanced Vectorizer with N-grams (captures phrases like "digital india")
        vectorizer_model = CountVectorizer(
            stop_words=self.custom_stopwords, 
            min_df=2,
            ngram_range=(1, 3)
        )
        
        # High-granularity clustering configuration
        from umap import UMAP
        from hdbscan import HDBSCAN
        from bertopic.vectorizers import ClassTfidfTransformer
        
        # More local neighbors = higher resolution clusters
        umap_model = UMAP(n_neighbors=5, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
        
        # Very small min_cluster_size to find micro-topics
        hdbscan_model = HDBSCAN(min_cluster_size=5, min_samples=2, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

        # BM25 Weighting for more precise keywords
        ctfidf_model = ClassTfidfTransformer(bm25_weighting=True, reduce_frequent_words=True)

        # Initialize BERTopic with None to keep ALL discovered clusters
        self.bertopic_model = BERTopic(
            nr_topics=None, 
            vectorizer_model=vectorizer_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            ctfidf_model=ctfidf_model,
            calculate_probabilities=True,
            verbose=False
        )
        
        # Fit
        topics, probs = self.bertopic_model.fit_transform(documents, embeddings)
        
        logger.info("BERTopic model fitted")
        
        # Get actual number of topics found (including outlier -1)
        actual_n = len(self.bertopic_model.get_topic_info()) - 1
        if actual_n < 1: actual_n = 1
        
        # Convert to topic distributions
        topic_dist = np.zeros((len(documents), actual_n))
        for i, (topic, prob_dist) in enumerate(zip(topics, probs)):
            if topic != -1 and prob_dist is not None:
                # BERTopic probabilities are for all topics except outlier
                if isinstance(prob_dist, np.ndarray):
                    if prob_dist.shape[0] == actual_n:
                        topic_dist[i] = prob_dist
                    elif prob_dist.shape[0] > 0:
                        # Fallback if shapes differ slightly
                        limit = min(prob_dist.shape[0], actual_n)
                        topic_dist[i, :limit] = prob_dist[:limit]
        
        return topic_dist, topics
    
    def fit_ensemble(self, documents, embeddings):
        """
        Fit all three models in a high-precision cascade
        """
        logger.info(f"Training high-precision ensemble on {len(documents)} documents...")
        
        # 1. Fit BERTopic first to determine the natural number of micro-topics
        bert_dist, bert_topics = self.fit_bertopic(documents, embeddings)
        
        # Determine how many topics were actually found
        discovered_n = len(self.bertopic_model.get_topic_info()) - 1 # excluding outlier -1
        if discovered_n < 5: discovered_n = self.n_topics # fallback
        
        logger.info(f"BERTopic discovered {discovered_n} micro-topics. Syncing LDA/NMF...")
        self.n_topics = discovered_n
        
        # 2. Fit LDA and NMF with the same granularity
        lda_topics = self.fit_lda(documents)
        nmf_topics = self.fit_nmf(documents)
        
        # Ensure bert_dist is the correct shape if it wasn't before
        # (re-running fit_bertopic logic inside here to ensure shape match)
        # Actually, fit_bertopic already returns bert_dist, but we need to ensure it's (len, discovered_n)
        if bert_dist.shape[1] != discovered_n:
            # Re-pad or re-slice if necessary
            new_bert_dist = np.zeros((len(documents), discovered_n))
            min_cols = min(bert_dist.shape[1], discovered_n)
            new_bert_dist[:, :min_cols] = bert_dist[:, :min_cols]
            bert_dist = new_bert_dist

        distributions = {
            'lda': lda_topics,
            'nmf': nmf_topics,
            'bertopic': bert_dist
        }
        
        # 3. Create consensus
        consensus = self.create_consensus(distributions)
        
        logger.info(f"Ensemble training complete. Consensus reached for {discovered_n} topics.")
        
        return consensus, distributions, bert_topics
    
    def create_consensus(self, distributions):
        """
        Weighted ensemble of topic distributions
        """
        consensus = (
            0.3 * distributions['lda'] +
            0.3 * distributions['nmf'] +
            0.4 * distributions['bertopic']
        )
        
        # Normalize
        consensus = consensus / consensus.sum(axis=1, keepdims=True)
        
        return consensus
    
    def get_topic_labels(self):
        """
        Extract topic labels from each model
        """
        labels = {}
        
        # LDA topics
        if self.lda_model and self.count_vectorizer:
            feature_names = self.count_vectorizer.get_feature_names_out()
            for topic_idx, topic in enumerate(self.lda_model.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                labels[f'Topic_{topic_idx}'] = {
                    'model': 'LDA',
                    'keywords': top_words
                }
        
        # BERTopic topics
        if self.bertopic_model:
            topic_info = self.bertopic_model.get_topic_info()
            for idx, row in topic_info.iterrows():
                if row['Topic'] != -1:
                    labels[f'Topic_{row["Topic"]}']['bertopic_label'] = row['Name']
        
        return labels
    
    def save_models(self, output_dir='./models/trained'):
        """Save all models"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each model
        if self.lda_model:
            with open(f'{output_dir}/lda_model.pkl', 'wb') as f:
                pickle.dump(self.lda_model, f)
            with open(f'{output_dir}/count_vectorizer.pkl', 'wb') as f:
                pickle.dump(self.count_vectorizer, f)
        
        if self.nmf_model:
            with open(f'{output_dir}/nmf_model.pkl', 'wb') as f:
                pickle.dump(self.nmf_model, f)
            with open(f'{output_dir}/tfidf_vectorizer.pkl', 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
        
        if self.bertopic_model:
            with open(f'{output_dir}/bertopic_model.pkl', 'wb') as f:
                pickle.dump(self.bertopic_model, f)
        
        logger.info(f"✓ Models saved to {output_dir}")

class ZeroShotLabeler:
    """
    Zero-shot topic classification using BART large MNLI
    Maps raw topic distributions/keywords to high-level policy domains.
    """
    def __init__(self, model_name="facebook/bart-large-mnli", device=None):
        if device is None:
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = device
            
        logger.info(f"Initializing ZeroShotLabeler with {model_name}...")
        self.classifier = pipeline("zero-shot-classification", model=model_name, device=self.device)
        self.candidate_labels = [
            "Infrastructure", "Manufacturing", "Digital Economy", "Welfare", 
            "Monetary Policy", "Inflation", "Geopolitics", "Agriculture",
            "Healthcare", "Education", "Culture", "Taxation"
        ]
        
    def classify_keywords(self, topic_keywords):
        """
        Takes a list of keywords and maps it to a high level category
        """
        text = " ".join(topic_keywords)
        result = self.classifier(text, self.candidate_labels)
        
        # Return top label
        return result['labels'][0], result['scores'][0]

if __name__ == "__main__":
    # Load data
    conn = sqlite3.connect('./data/market_rhetoric.db')
    df = pd.read_sql_query(
        "SELECT processed_text FROM speeches WHERE processed_text IS NOT NULL",
        conn
    )
    conn.close()
    
    documents = df['processed_text'].tolist()
    
    if len(documents) > 0:
        # Load embeddings
        embeddings_dict = np.load('./data/processed/speech_embeddings.npy', allow_pickle=True).item()
        embeddings = np.array(list(embeddings_dict.values()))
        
        # Train models
        modeler = HybridTopicModeler(n_topics=20)
        consensus, distributions = modeler.fit_ensemble(documents, embeddings)
        
        # Save
        modeler.save_models()
        np.save('./data/processed/topic_distributions.npy', consensus)
    else:
        logger.error("No documents to process. Run preprocessing first.")
