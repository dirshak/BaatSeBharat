import pandas as pd
import numpy as np
import sqlite3
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class FusionEngine:
    """
    Rhetorical-Financial Superimposition Engine.
    Fuses text and market branches via PWM Shock Modeling and Temporal Topic Alignment.
    """
    def __init__(self, db_path='./data/market_rhetoric.db'):
        self.db_path = db_path
        
    def calculate_pwm_shock(self, returns_array, tail_sensitivity_r=2.0):
        """
        Probability-Weighted Moment (PWM) Shock Modeling.
        Formula: PWM = E_emp( X * F(X)^r )
        Quantifies the impact on market tails.
        """
        if len(returns_array) == 0:
            return 0.0
            
        # Empirical CDF calculation
        sorted_returns = np.sort(returns_array)
        n = len(sorted_returns)
        # F(X) is empirical cumulative probability
        F_x = np.arange(1, n + 1) / n
        
        # PWM = mean( X * F(X)^r )
        pwm = np.mean(sorted_returns * (F_x ** tail_sensitivity_r))
        return float(pwm)
        
    def compute_all_shocks(self):
        """
        Calculates the PWM shock impact score for each speech event across all tickers
        and updates the speech_market_impact table.
        """
        logger.info("Computing PWM Shock Impact Scores...")
        conn = sqlite3.connect(self.db_path)
        
        # Pull standard impacts
        impacts = pd.read_sql_query("SELECT id, ticker, event_date, return_t5 FROM speech_market_impact", conn)
        market_returns = pd.read_sql_query("SELECT ticker, returns FROM market_data WHERE returns IS NOT NULL", conn)
        
        updated_count = 0
        tickers = market_returns['ticker'].unique()
        
        for ticker in tickers:
            ticker_data = market_returns[market_returns['ticker'] == ticker]['returns'].values
            if len(ticker_data) < 20:
                continue
                
            ticker_impacts = impacts[impacts['ticker'] == ticker]
            
            for _, row in ticker_impacts.iterrows():
                if pd.isna(row['return_t5']):
                    continue
                    
                # To simulate the shock, we weight the specific event return against the empirical distribution of all returns
                # Here, we treat the event's forward return as the "shock" sample
                # A full PWM would be the integral over a regime window. For discrete events, we can score 
                # how 'extreme' the event is using the empirical CDF of the asset's history.
                event_return = row['return_t5']
                # F(x) definition: P(X <= x)
                f_x = np.sum(ticker_data <= event_return) / len(ticker_data)
                
                # Simplified shock score = X * F(X)^r
                # Focuses on extreme positive tails if X>0, or just measures position in distribution
                # r=2.0 emphasizes the right tail heavily.
                shock_score = event_return * (f_x ** 2.0)
                
                conn.execute(
                    "UPDATE speech_market_impact SET pwm_shock_score = ? WHERE id = ?",
                    (float(shock_score), int(row['id']))
                )
                updated_count += 1
                
        conn.commit()
        conn.close()
        logger.info(f"✓ Updated PWM shock scores for {updated_count} impact events.")

if __name__ == "__main__":
    fusion = FusionEngine()
    fusion.compute_all_shocks()
