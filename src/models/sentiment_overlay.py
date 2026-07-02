import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class SentimentOverlay:
    """
    Financial sentiment extraction using FinBERT.
    Captures optimism intensity and risk awareness from political/financial rhetoric.
    """
    
    def __init__(self, model_name="ProsusAI/finbert"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing FinBERT Sentiment Overlay on {self.device}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
            self.model.eval()
            logger.info("✓ FinBERT initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to load FinBERT model: {e}")
            raise
            
    def analyze_sentiment(self, text):
        """
        Analyzes sentiment of a given text and returns probabilities.
        FinBERT outputs: [positive, negative, neutral]
        
        Returns:
            dict containing positive, negative, neutral, optimism_intensity, risk_awareness, compound
        """
        if not text or len(text.strip()) == 0:
            return {
                'positive': 0.0, 'negative': 0.0, 'neutral': 1.0,
                'optimism_intensity': 0.0, 'risk_awareness': 0.0, 'compound': 0.0
            }
            
        # Truncate to maximum length FinBERT accepts (512 tokens)
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Softmax to get probabilities
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy()
            
        # ProsusAI/finbert mapping: 0 -> positive, 1 -> negative, 2 -> neutral
        pos, neg, neu = float(probs[0]), float(probs[1]), float(probs[2])
        
        # Define optimism and risk based on instruction requirements
        optimism_intensity = pos
        risk_awareness = neg
        
        # Compound score calculation: positive - negative (scaled by neutrality)
        compound = (pos - neg) * (1.0 - neu * 0.5)
        
        return {
            'positive': pos,
            'negative': neg,
            'neutral': neu,
            'optimism_intensity': optimism_intensity,
            'risk_awareness': risk_awareness,
            'compound': compound
        }

if __name__ == "__main__":
    # Simple test
    analyzer = SentimentOverlay()
    sample_text = "The economy is showing strong signs of recovery, but inflation risks remain elevated."
    scores = analyzer.analyze_sentiment(sample_text)
    print(f"Text: {sample_text}\nScores: {scores}")
