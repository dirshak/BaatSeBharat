import pandas as pd
import numpy as np
import sqlite3
import sys
import os
from statsmodels.tsa.stattools import grangercausalitytests

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class CausalValidator:
    """
    Evaluates Granger Causality: determines if leadership topic intensities
    predict sectoral volatility or abnormal returns.
    """
    def __init__(self, db_path='./data/market_rhetoric.db'):
        self.db_path = db_path
        
    def test_causality(self, maxlag=5):
        logger.info("Running Granger Causality Tests on Topic Probabilities vs Market Returns...")
        conn = sqlite3.connect(self.db_path)
        
        # We need a time series of topic intensities and a corresponding market time series
        # Easiest way: merge speech dates with market returns
        
        # 1. Topic Intensities per day (average if multiple)
        topic_ts_query = '''
            SELECT s.date, td.topic_id, AVG(td.probability) as avg_prob
            FROM speeches s
            JOIN topic_distributions td ON s.id = td.speech_id
            WHERE s.date IS NOT NULL
            GROUP BY s.date, td.topic_id
        '''
        topic_ts = pd.read_sql_query(topic_ts_query, conn)
        
        if topic_ts.empty:
            logger.warning("No topic distributions available for causality testing.")
            return None
            
        topic_ts['date'] = pd.to_datetime(topic_ts['date'])
        
        # 2. Daily Market Returns (we'll focus on the broad market or average)
        market_ts = pd.read_sql_query("SELECT date, returns, volatility FROM market_data WHERE returns IS NOT NULL", conn)
        market_ts['date'] = pd.to_datetime(market_ts['date'])
        
        # Aggregate market returns across all tickers per day to get a general market TS
        daily_market = market_ts.groupby('date')[['returns', 'volatility']].mean().reset_index()
        
        results = {}
        
        topics = topic_ts['topic_id'].unique()
        for topic in topics:
            t_df = topic_ts[topic_ts['topic_id'] == topic]
            
            # Merge with market
            merged = pd.merge(daily_market, t_df, on='date', how='left')
            merged['avg_prob'] = merged['avg_prob'].fillna(0.0) # 0 probability if no speech that day
            
            # Sort by date
            merged = merged.sort_values('date')
            
            # Formate for statsmodels granger: [target_variable, predictor_variable]
            # Does Topic (predictor) granger-cause Returns (target)?
            data_returns = merged[['returns', 'avg_prob']].dropna()
            
            if len(data_returns) > maxlag + 5:
                try:
                    # suppressing output inside the function causes issues sometimes, but we catch it
                    gc_res = grangercausalitytests(data_returns, maxlag=maxlag, verbose=False)
                    # Get p-value for the lag with lowest p-value
                    min_p_val = min([round(gc_res[i+1][0]['ssr_ftest'][1], 4) for i in range(maxlag)])
                    results[f"Topic_{topic}_Causes_Returns"] = min_p_val
                except Exception as e:
                    logger.debug(f"Granger causality failed for topic {topic}: {e}")
                    
        conn.close()
        
        if results:
            logger.info("Granger Causality P-Values (Lag 1-5, minimum F-test p-val):")
            for k, p in results.items():
                if p < 0.05:
                    logger.info(f"  {k}: {p} *** SIGNIFICANT ***")
                else:
                    logger.info(f"  {k}: {p}")
        return results

if __name__ == "__main__":
    validator = CausalValidator()
    validator.test_causality()
