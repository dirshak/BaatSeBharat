import sqlite3
import pandas as pd
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.features.text_preprocessing import TextPreprocessor
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def preprocess_all_speeches(db_path='./data/market_rhetoric.db'):
    """
    Preprocess all speeches in database
    """
    logger.info("Loading speeches from database...")
    
    # Load speeches
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT id, full_text FROM speeches", conn)
    
    logger.info(f"Loaded {len(df)} speeches")
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Process each speech
    logger.info("Preprocessing speeches...")
    
    processed_texts = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # Preprocess
            processed = preprocessor.preprocess(row['full_text'])
            processed_texts.append(processed)
        except Exception as e:
            logger.error(f"Error processing speech {row['id']}: {e}")
            processed_texts.append("")
    
    # Add to dataframe
    df['processed_text'] = processed_texts
    
    # Save processed text back to database
    logger.info("Saving processed text to database...")
    
    # Add column if doesn't exist
    try:
        conn.execute('''
            ALTER TABLE speeches 
            ADD COLUMN processed_text TEXT
        ''')
    except sqlite3.OperationalError:
        pass # Column might already exist
    
    # Update each row
    for idx, row in df.iterrows():
        conn.execute(
            "UPDATE speeches SET processed_text = ? WHERE id = ?",
            (row['processed_text'], row['id'])
        )
    
    conn.commit()
    conn.close()
    
    logger.info("✓ Preprocessing complete")
    
    # Statistics
    avg_length = df['processed_text'].str.len().mean()
    logger.info(f"Average processed text length: {avg_length:.0f} characters")

if __name__ == "__main__":
    preprocess_all_speeches()
