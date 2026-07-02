import json
import os
import pickle
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from langchain_core.messages import AIMessage

try:
    from sklearn.ensemble import RandomForestClassifier
    _SKLEARN_AVAILABLE = True
except ImportError:  # sklearn is optional — fall back to random-walk predictions
    _SKLEARN_AVAILABLE = False
    RandomForestClassifier = None  # type: ignore[assignment,misc]

def create_quant_analyst(llm):
    def quant_analyst_node(state):
        ticker = state["company_of_interest"]
        current_date_str = state["trade_date"]
        current_date = datetime.strptime(current_date_str, "%Y-%m-%d")
        
        # Load historical daily data (4 years lookback for training)
        start_date = current_date - timedelta(days=365 * 4)
        
        # Output directory for models
        model_dir = "./models/trained"
        os.makedirs(model_dir, exist_ok=True)
        
        try:
            df = yf.Ticker(ticker).history(start=start_date.strftime("%Y-%m-%d"), end=(current_date + timedelta(days=1)).strftime("%Y-%m-%d"))
            if df.empty or len(df) < 150:
                raise ValueError("Insufficient data to train ML models (needs at least 150 rows).")
                
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
                
            # Filter to <= current_date
            df = df[df.index <= current_date].copy()
            
            # Feature engineering
            df['Ret'] = df['Close'].pct_change()
            df['Lag1'] = df['Ret'].shift(1)
            df['Lag3'] = df['Ret'].shift(3)
            df['Lag5'] = df['Ret'].shift(5)
            df['Lag10'] = df['Ret'].shift(10)
            df['Lag20'] = df['Ret'].shift(20)
            df['Vol5'] = df['Ret'].rolling(5).std()
            df['Vol20'] = df['Ret'].rolling(20).std()
            df['SMA50_Ratio'] = df['Close'] / df['Close'].rolling(50).mean()
            
            # Target horizons
            horizons = [5, 20, 60]
            predictions = {}
            
            for h in horizons:
                # Forward return for target
                df[f'FwdRet_{h}'] = (df['Close'].shift(-h) - df['Close']) / df['Close']
                df[f'Target_{h}'] = (df[f'FwdRet_{h}'] > 0).astype(int)
                
                # Features list
                features = ['Lag1', 'Lag3', 'Lag5', 'Lag10', 'Lag20', 'Vol5', 'Vol20', 'SMA50_Ratio']
                
                # Training subset: rows where target is known (i.e. up to trade_date - h days)
                train_df = df.dropna(subset=features + [f'Target_{h}', f'FwdRet_{h}'])
                train_df = train_df[train_df.index < current_date - timedelta(days=h)]
                
                if len(train_df) < 50 or not _SKLEARN_AVAILABLE:
                    predictions[h] = {
                        "direction": "UP",
                        "expected_return": 0.01 * h,
                        "risk": float(df['Ret'].std() * np.sqrt(h)),
                        "confidence": 0.50
                    }
                    continue
                
                X_train = train_df[features]
                y_train = train_df[f'Target_{h}']
                
                # Fit Random Forest Classifier
                clf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
                clf.fit(X_train, y_train)
                
                # Persist model
                model_path = os.path.join(model_dir, f"rf_model_{ticker}_{h}d_{current_date_str}.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(clf, f)
                
                # Predict for current date row (the last row)
                current_features = df[features].iloc[[-1]]
                if current_features.isna().any().any():
                    pred_class = 1
                    prob = 0.50
                else:
                    pred_class = int(clf.predict(current_features)[0])
                    prob = float(clf.predict_proba(current_features)[0][pred_class])
                
                # Expected return calculation
                # Use mean return of predicted class in training set
                expected_ret = float(train_df[train_df[f'Target_{h}'] == pred_class][f'FwdRet_{h}'].mean())
                if pd.isna(expected_ret):
                    expected_ret = 0.01 * h if pred_class == 1 else -0.01 * h
                    
                predictions[h] = {
                    "direction": "UP" if pred_class == 1 else "DOWN",
                    "expected_return": expected_ret,
                    "risk": float(df['Ret'].std() * np.sqrt(h)),
                    "confidence": prob
                }
                
            # Derive a composite score from the 3-horizon confidence-weighted directions
            def _dir_score(pred):
                sign = 1.0 if pred.get("direction") == "UP" else -1.0
                return sign * (pred.get("confidence", 0.5) - 0.5) * 2.0  # range [-1, 1]

            quant_score = round(
                (_dir_score(predictions[5]) + _dir_score(predictions[20]) + _dir_score(predictions[60])) / 3.0,
                4
            )
            evidence = {
                "ticker": ticker,
                "prediction_date": current_date_str,
                "horizon_5d": predictions[5],
                "horizon_20d": predictions[20],
                "horizon_60d": predictions[60],
                "score": quant_score,
                "summary": (
                    f"5d={predictions[5]['direction']}({predictions[5]['confidence']:.0%}), "
                    f"20d={predictions[20]['direction']}({predictions[20]['confidence']:.0%}), "
                    f"60d={predictions[60]['direction']}({predictions[60]['confidence']:.0%})"
                ),
            }
            
        except Exception as e:
            evidence = {
                "ticker": ticker,
                "error": str(e),
                "horizon_5d": {"direction": "UP", "expected_return": 0.0, "risk": 0.05, "confidence": 0.5},
                "horizon_20d": {"direction": "UP", "expected_return": 0.0, "risk": 0.10, "confidence": 0.5},
                "horizon_60d": {"direction": "UP", "expected_return": 0.0, "risk": 0.17, "confidence": 0.5},
                "score": 0.0,
                "summary": f"Error: {str(e)[:120]}",
            }
            
        report_str = json.dumps(evidence, indent=2)
        
        return {
            "messages": [AIMessage(content=f"Quant analyst report: {report_str}")],
            "quant_report": report_str,
            "quant_evidence": evidence
        }
        
    return quant_analyst_node
