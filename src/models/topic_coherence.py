from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
import pandas as pd
import sqlite3
import pickle
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def calculate_coherence(model_path, documents):
    """
    Calculate topic coherence for validation
    """
    logger.info("Calculating topic coherence...")
    
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Tokenize documents
    texts = [doc.split() for doc in documents]
    
    # Create dictionary
    dictionary = Dictionary(texts)
    
    # Calculate coherence
    try:
        coherence_model = CoherenceModel(
            model=model,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        
        coherence_score = coherence_model.get_coherence()
        
        logger.info(f"✓ Coherence score: {coherence_score:.4f}")
        
        return coherence_score
        
    except Exception as e:
        logger.error(f"Error calculating coherence: {e}")
        return None

if __name__ == "__main__":
    # Load documents
    conn = sqlite3.connect('./data/market_rhetoric.db')
    df = pd.read_sql_query(
        "SELECT processed_text FROM speeches WHERE processed_text IS NOT NULL",
        conn
    )
    conn.close()
    
    documents = df['processed_text'].tolist()
    
    # Calculate for each model
    lda_path = './models/trained/lda_model.pkl'
    if os.path.exists(lda_path):
        lda_coherence = calculate_coherence(lda_path, documents)
        print(f"LDA Coherence: {lda_coherence}")
    else:
        print("Model not found. Train topic models first.")
