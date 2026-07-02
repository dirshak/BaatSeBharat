import spacy
import re
from indicnlp.tokenize import indic_tokenize
from indicnlp.normalize import indic_normalize
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class TextPreprocessor:
    """
    Comprehensive text preprocessing for multilingual speeches
    """
    
    def __init__(self):
        # Load English model
        try:
            self.nlp_en = spacy.load('en_core_web_sm')
        except:
            logger.error("English spaCy model not found. Run: python -m spacy download en_core_web_sm")
            raise
        
        # Hindi normalizer
        self.hindi_normalizer = indic_normalize.IndicNormalizerFactory().get_normalizer("hi")
        
        # Extended stopwords
        self.custom_stopwords = {
            'mann', 'ki', 'baat', 'friends', 'dear', 'countrymen',
            'namaskar', 'namaste', 'ji', 'today', 'time', 'country',
            'people', 'india', 'indian', 'government', 'year', 'month',
            'day', 'week', 'says', 'said', 'will', 'also', 'many', 'much',
            'make', 'made', 'take', 'taken', 'get', 'got', 'give', 'given'
        }
    
    def clean_text(self, text):
        """Basic text cleaning"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_english(self, text):
        """
        Preprocess English text with POS filtering and lemmatization
        """
        # Clean text
        text = self.clean_text(text)
        
        # Process with spaCy
        doc = self.nlp_en(text)
        
        # POS filtering and lemmatization
        filtered_tokens = []
        
        for token in doc:
            # Filter conditions
            if (token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN'] and
                not token.is_stop and
                not token.is_punct and
                len(token.text) > 2 and
                token.lemma_ not in self.custom_stopwords):
                
                filtered_tokens.append(token.lemma_)
        
        return ' '.join(filtered_tokens)
    
    def preprocess_hindi(self, text):
        """
        Preprocess Hindi text
        """
        # Normalize
        normalized = self.hindi_normalizer.normalize(text)
        
        # Tokenize
        tokens = indic_tokenize.trivial_tokenize(normalized)
        
        # Filter
        filtered = [
            t for t in tokens 
            if len(t) > 1 and t.isalnum()
        ]
        
        return ' '.join(filtered)
    
    def detect_language(self, text):
        """
        Simple language detection (Hindi vs English)
        """
        # Check for Devanagari script
        devanagari_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
        
        if devanagari_chars > len(text) * 0.3:
            return 'hindi'
        else:
            return 'english'
    
    def preprocess(self, text, language=None):
        """
        Automatic preprocessing based on detected language
        """
        if not language:
            language = self.detect_language(text)
        
        if language == 'hindi':
            return self.preprocess_hindi(text)
        else:
            return self.preprocess_english(text)
    
    def segment_sentences(self, text, language='english'):
        """
        Segment text into sentences
        Important for handling BERT's 512 token limit
        """
        if language == 'english':
            doc = self.nlp_en(text)
            return [sent.text for sent in doc.sents]
        else:
            # Hindi sentence segmentation
            sentences = text.split('।')  # Devanagari full stop
            return [s.strip() for s in sentences if s.strip()]
    
    def create_hierarchical_structure(self, text, language='english'):
        """
        Create Episode > Paragraph > Sentence hierarchy
        """
        paragraphs = text.split('\n\n')
        
        hierarchy = {
            'episode_text': text,
            'paragraphs': []
        }
        
        for para in paragraphs:
            if para.strip():
                sentences = self.segment_sentences(para, language)
                hierarchy['paragraphs'].append({
                    'text': para,
                    'sentences': sentences,
                    'n_sentences': len(sentences)
                })
        
        return hierarchy

# Testing
if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    
    # Test English
    test_text_en = "Friends, today I want to talk about India's progress in technology."
    processed = preprocessor.preprocess_english(test_text_en)
    print(f"Original: {test_text_en}")
    print(f"Processed: {processed}")
    
    # Test hierarchy
    long_text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
    hierarchy = preprocessor.create_hierarchical_structure(long_text)
    print(f"Paragraphs: {len(hierarchy['paragraphs'])}")
