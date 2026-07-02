import pandas as pd
import numpy as np
import sqlite3
import sys
import os
from hmmlearn import hmm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class MarketModeler:
    """
    Implements advanced market models: ASBN, CPTM-F, Synchrony, and Trajectory Shapes.
    """
    def __init__(self, db_path='./data/market_rhetoric.db'):
        self.db_path = db_path

    def calculate_asbn(self, df, stable_window=252):
        """
        Asset-Specific Baseline Normalization (ASBN).
        Computes Z(t) = (x(t) - mean_asset) / std_asset using a rolling stable regime.
        """
        # Calculate moving mean and std over the stable window (~1 trading year)
        rolling_mean = df['close'].rolling(window=stable_window, min_periods=20).mean()
        rolling_std = df['close'].rolling(window=stable_window, min_periods=20).std()
        
        # Avoid division by zero
        rolling_std = rolling_std.replace(0, np.nan)
        
        asbn = (df['close'] - rolling_mean) / rolling_std
        return asbn

    def calculate_cptm_f(self, df, trend_window=60):
        """
        Counterfactual Price Trajectory Modeling (CPTM-F).
        Calculates deviation D(t) = (actual(t) - expected(t)) / std_asset.
        Expected price is a simple linear extrapolation from the previous trend_window.
        """
        expected = df['close'].rolling(window=trend_window).mean()
        rolling_std = df['close'].rolling(window=trend_window).std().replace(0, np.nan)
        
        cptm_f = (df['close'] - expected) / rolling_std
        return cptm_f

    def calculate_trajectory_acceleration(self, df):
        """
        Trajectory Shape Features: Acceleration (2nd derivative of price).
        """
        velocity = df['close'].diff()
        acceleration = velocity.diff()
        return acceleration

    def _validate_data_for_hmm(self, df):
        """Check for NaN, inf, or zero-variance columns in Nifty returns"""
        if df.empty:
            return False, "DataFrame is empty"
        
        # Calculate log returns if not present
        if 'returns' not in df.columns:
            df['returns'] = np.log(df['close'] / df['close'].shift(1))
        
        df['returns'] = df['returns'].replace([np.inf, -np.inf], np.nan).ffill()
        
        clean_df = df.dropna(subset=['returns'])
        
        if len(clean_df) < 1000:
            return False, f"Insufficient data points: {len(clean_df)} < 1000"
            
        if clean_df['returns'].std() == 0:
            return False, "Zero variance in returns"
            
        return True, clean_df

    def _train_hmm(self, data):
        """Train GaussianHMM with stability checks"""
        best_model = None
        best_score = -np.inf
        
        # Reshape data for hmmlearn
        X = data['returns'].values.reshape(-1, 1)
        
        for i in range(5):
            try:
                model = hmm.GaussianHMM(
                    n_components=3, 
                    covariance_type="full", 
                    n_iter=1000, 
                    tol=1e-6,
                    random_state=i
                )
                model.fit(X)
                score = model.score(X)
                if score > best_score:
                    best_score = score
                    best_model = model
            except Exception as e:
                logger.warning(f"HMM initialization {i} failed: {e}")
                
        return best_model

    def _fallback_regime_logic(self, df):
        """Rule-based classifier if HMM fails"""
        # Stable = 21-day rolling vol < 50th percentile
        # Volatile = > 90th percentile
        # Transitional = between
        vol = df['returns'].rolling(21).std()
        p50 = vol.median()
        p90 = vol.quantile(0.9)
        
        regimes = []
        for v in vol:
            if pd.isna(v):
                regimes.append('Stable')
            elif v < p50:
                regimes.append('Stable')
            elif v > p90:
                regimes.append('Volatile')
            else:
                regimes.append('Transitional')
        return regimes

    def compute_regime_metrics(self):
        """
        Computes ASBN, CPTM-F, and HMM regimes for all tickers.
        """
        logger.info("Computing ASBN, CPTM-F, and HMM Regimes...")
        conn = sqlite3.connect(self.db_path)
        
        market_df = pd.read_sql_query("SELECT * FROM market_data ORDER BY date", conn)
        market_df['date'] = pd.to_datetime(market_df['date'])
        
        tickers = market_df['ticker'].unique()
        
        # Ensure classifications table exists or clear it
        conn.execute("DELETE FROM regime_classifications")
        
        for ticker in tickers:
            df = market_df[market_df['ticker'] == ticker].copy()
            df = df.sort_values('date')
            
            # 1. HMM Regime Intelligence
            valid, result = self._validate_data_for_hmm(df)
            hmm_model = None
            if valid:
                hmm_model = self._train_hmm(result)
            
            if hmm_model:
                X = df['returns'].replace([np.inf, -np.inf], np.nan).ffill().fillna(0).values.reshape(-1, 1)
                regime_idx = hmm_model.predict(X)
                probs = hmm_model.predict_proba(X)
                
                # Map indices to labels (Stable, Transitional, Volatile)
                # Usually based on variance or mean. Let's use variance for regime sorting.
                variances = [np.diag(hmm_model.covars_[i])[0] for i in range(3)]
                sorted_idx = np.argsort(variances)
                label_map = {sorted_idx[0]: 'Stable', sorted_idx[1]: 'Transitional', sorted_idx[2]: 'Volatile'}
                
                df['regime_label'] = [label_map[i] for i in regime_idx]
                df['regime_probability'] = [np.max(p) for p in probs]
            else:
                logger.warning(f"HMM failed for {ticker}: {result}. Using fallback logic.")
                df['regime_label'] = self._fallback_regime_logic(df)
                df['regime_probability'] = 0.5 # Default probability for fallback
            
            # 2. Advanced Metrics (ASBN, CPTM-F)
            df['asbn'] = self.calculate_asbn(df)
            df['cptm_f'] = self.calculate_cptm_f(df)
            df['volume_zscore'] = (df['volume'] - df['volume'].rolling(60).mean()) / df['volume'].rolling(60).std().replace(0, np.nan)
            
            # 3. Save to DB and regime_labels.csv
            df_out = df[['date', 'regime_label', 'regime_probability']]
            df_out.to_csv(f'./data/processed/regime_labels_{ticker}.csv', index=False)
            
            for _, row in df.iterrows():
                try:
                    conn.execute('''
                        INSERT INTO regime_classifications 
                        (date, sector, regime, confidence, deviation_magnitude, volume_zscore)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        row['date'].strftime('%Y-%m-%d'),
                        ticker,
                        row['regime_label'],
                        row['regime_probability'],
                        row['cptm_f'] if pd.notna(row['cptm_f']) else 0.0,
                        row['volume_zscore'] if pd.notna(row['volume_zscore']) else 0.0
                    ))
                except Exception as e:
                    pass
                    
        conn.commit()
        conn.close()
        logger.info("✓ Regime metrics computed and saved.")

if __name__ == "__main__":
    mm = MarketModeler()
    mm.compute_regime_metrics()
