import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

class DualPredictiveModels:
    """Implements Model A (Stock only) and Model B (Hybrid rhetoric-enhanced)."""
    
    def __init__(self, window_size=20, prediction_horizon=5):
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.model_a = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model_b = RandomForestRegressor(n_estimators=100, random_state=42)

    def prepare_features(self, market_df, topic_df=None):
        """Creates lagged features for training."""
        df = market_df.copy()
        
        # Model A Features: Price lags
        for i in range(1, self.window_size + 1):
            df[f'price_lag_{i}'] = df['Price'].shift(i)
        
        # Target: Future Price
        df['target'] = df['Price'].shift(-self.prediction_horizon)
        
        if topic_df is not None:
            # Model B Features: Combine A + Topics
            # Align topic_df to market dates
            topic_df = topic_df.reindex(df.index).ffill(limit=30).fillna(0)
            df_hybrid = pd.concat([df, topic_df], axis=1)
            return df, df_hybrid
            
        return df, None

    def train_and_predict(self, df, feature_cols):
        """Trains on features and predicts the last available data point."""
        data = df.dropna(subset=['target'] + feature_cols)
        if data.empty:
            return None, None
            
        X = data[feature_cols]
        y = data['target']
        
        # Split conceptually (last 20% for testing)
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        test_preds = model.predict(X_test)
        metrics = {
            "RMSE": np.sqrt(mean_squared_error(y_test, test_preds)),
            "R2": r2_score(y_test, test_preds)
        }
        
        # Predict future (last row)
        last_features = df[feature_cols].iloc[-1:].values
        future_pred = model.predict(last_features)[0]
        
        return future_pred, metrics
