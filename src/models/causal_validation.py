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

        topic_labels = {}
        labels_path = os.path.join(os.path.dirname(self.db_path) or '.', 'processed', 'topic_labels_combined.json')
        if os.path.exists(labels_path):
            import json
            with open(labels_path, 'r', encoding='utf-8') as f:
                topic_labels = {int(k): v['label'] for k, v in json.load(f).items()}
        
        # We need a time series of topic intensities and a corresponding market time series
        # Easiest way: merge speech dates with market returns
        
        # 1. Topic Intensities per day (average if multiple)
        topic_ts_query = '''
            SELECT s.date, td.topic_id, AVG(td.probability) as avg_prob
            FROM speeches s
            JOIN topic_distributions td ON s.id = td.speech_id
            WHERE s.date IS NOT NULL AND td.model_name = 'Combined'
            GROUP BY s.date, td.topic_id
        '''
        topic_ts = pd.read_sql_query(topic_ts_query, conn)

        if topic_ts.empty:
            logger.warning("No topic distributions available for causality testing.")
            return None

        # Some legacy speech rows have a literal "N/A" date string instead
        # of NULL, which `WHERE s.date IS NOT NULL` doesn't catch and which
        # crashes pd.to_datetime() with a strict format. Coerce and drop
        # unparseable rows instead.
        topic_ts['date'] = pd.to_datetime(topic_ts['date'], errors='coerce')
        topic_ts = topic_ts.dropna(subset=['date'])

        # 2. Daily Market Returns (we'll focus on the broad market or average)
        market_ts = pd.read_sql_query("SELECT date, returns, volatility FROM market_data WHERE returns IS NOT NULL", conn)
        market_ts['date'] = pd.to_datetime(market_ts['date'], errors='coerce')
        market_ts = market_ts.dropna(subset=['date'])
        
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
                    label = topic_labels.get(topic, f"Topic_{topic}")
                    results[f"{label} -> Returns"] = min_p_val
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

    def backtest_directional_hit_rate(self, train_frac: float = 0.7) -> dict:
        """Out-of-sample directional backtest: does the sign of a topic's
        TRAIN-period average (probability-weighted) abnormal return predict
        the sign of TEST-period abnormal returns for speeches dominated by
        that topic?

        This is a genuine train/test split by date (not in-sample fitting),
        reported honestly -- a hit rate near 50% means the signal is weak
        or absent out-of-sample, which is a real, reportable result, not a
        failure to hide. Returns a dict with hit_rate, n_events, n_train,
        n_test, cutoff_date -- or {} if there isn't enough data to backtest.
        """
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('''
            SELECT td.speech_id, s.date, td.topic_id, td.probability,
                   i.ticker, i.abnormal_return
            FROM topic_distributions td
            JOIN speeches s ON td.speech_id = s.id
            JOIN speech_market_impact i ON i.speech_id = s.id
            WHERE td.model_name = 'Combined' AND i.abnormal_return IS NOT NULL
        ''', conn)
        conn.close()

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        if df.empty:
            return {}

        cutoff = df['date'].quantile(train_frac)
        train, test = df[df['date'] <= cutoff], df[df['date'] > cutoff]
        if train.empty or test.empty:
            return {}

        # Per-topic bias learned ONLY from train: probability-weighted
        # average abnormal return following speeches dominated by that topic.
        topic_bias = train.groupby('topic_id', group_keys=True).apply(
            lambda g: np.average(g['abnormal_return'], weights=g['probability']),
            include_groups=False,
        )

        def _predict(g):
            weights = g['probability'].values
            biases = topic_bias.reindex(g['topic_id']).values
            return pd.Series({
                'predicted_signal': np.average(biases, weights=weights),
                'actual_return': g['abnormal_return'].iloc[0],
            })

        test_events = test.groupby(['speech_id', 'ticker'], group_keys=True).apply(
            _predict, include_groups=False
        ).reset_index()

        pred_dir = np.sign(test_events['predicted_signal'])
        actual_dir = np.sign(test_events['actual_return'])
        mask = actual_dir != 0
        n_events = int(mask.sum())
        if n_events == 0:
            return {}

        hit_rate = float((pred_dir[mask] == actual_dir[mask]).mean())
        return {
            'hit_rate': hit_rate,
            'n_events': n_events,
            'n_train_rows': len(train),
            'n_test_rows': len(test),
            'cutoff_date': str(cutoff.date()),
        }


if __name__ == "__main__":
    validator = CausalValidator()
    validator.test_causality()
    print(validator.backtest_directional_hit_rate())
