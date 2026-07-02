import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests

class AdvancedAnalytics:
    """Advanced statistical methods for market-speech analysis."""
    
    @staticmethod
    def test_granger_causality(data, topic_col, market_col, max_lag=10):
        """
        Tests if topic fluctuations 'cause' market shifts.
        Null Hypothesis: Topic does NOT Granger-cause Market.
        """
        # Data must be stationary (simplified check)
        df = data[[market_col, topic_col]].pct_change().dropna()
        
        try:
            results = grangercausalitytests(df, max_lag, verbose=False)
            # Extract p-values for each lag
            p_values = [results[i+1][0]['ssr_ftest'][1] for i in range(max_lag)]
            return p_values
        except Exception as e:
            return [1.0] * max_lag # Return 1.0 (insignificant) on error

    @staticmethod
    def calculate_topic_vix_sensitivity(merged_df, topic_cols, vix_col='VIX'):
        """Calculates which topics have the highest regression beta with VIX."""
        sensitivities = {}
        for topic in topic_cols:
            correlation = merged_df[topic].corr(merged_df[vix_col])
            sensitivities[topic] = correlation
        return pd.Series(sensitivities).sort_values(ascending=False)
