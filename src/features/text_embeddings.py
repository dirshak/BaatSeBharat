from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import pandas as pd
import sqlite3
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class TextEmbeddingGenerator:
    """
    Generate text embeddings using SBERT and IndicBERT
    """
    
    def __init__(self):
        logger.info("Loading embedding models...")
        
        # SBERT for all (384-dim)
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        logger.info("✓ Models loaded successfully")
    
    def embed_sbert(self, texts, batch_size=32, show_progress=True):
        """
        Generate SBERT embeddings
        Returns: numpy array of shape (n_texts, 384)
        """
        logger.info(f"Generating SBERT embeddings for {len(texts)} texts...")
        
        embeddings = self.sbert_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        logger.info(f"✓ Generated embeddings with shape {embeddings.shape}")
        return embeddings
    
    def embed_multilingual(self, texts, batch_size=32, show_progress=True):
        """
        Generate Multilingual SBERT embeddings
        Returns: numpy array of shape (n_texts, 384)
        """
        logger.info(f"Generating Multilingual SBERT embeddings for {len(texts)} texts...")
        
        embeddings = self.multilingual_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        logger.info(f"✓ Generated embeddings with shape {embeddings.shape}")
        return embeddings
    
    def generate_and_save_embeddings(self, db_path='./data/market_rhetoric.db'):
        """
        Generate embeddings for all speeches and save
        """
        # Load speeches
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(
            "SELECT id, processed_text FROM speeches WHERE processed_text IS NOT NULL",
            conn
        )
        
        logger.info(f"Generating embeddings for {len(df)} speeches...")
        
        # Generate embeddings
        embeddings_dict = {}
        
        if len(df) > 0:
            logger.info(f"Processing {len(df)} speeches...")
            embeddings = self.embed_sbert(df['processed_text'].tolist())
            
            for idx, speech_id in enumerate(df['id']):
                embeddings_dict[speech_id] = embeddings[idx]
        
        # Save embeddings
        logger.info("Saving embeddings...")
        
        # Save as numpy file
        os.makedirs('./data/processed', exist_ok=True)
        np.save('./data/processed/speech_embeddings.npy', embeddings_dict)
        
        # Also save mapping
        import pickle
        with open('./data/processed/speech_id_mapping.pkl', 'wb') as f:
            pickle.dump(list(embeddings_dict.keys()), f)
        
        conn.close()
        
        logger.info(f"✓ Saved embeddings for {len(embeddings_dict)} speeches")
        
        return embeddings_dict

if __name__ == "__main__":
    generator = TextEmbeddingGenerator()
    embeddings = generator.generate_and_save_embeddings()
