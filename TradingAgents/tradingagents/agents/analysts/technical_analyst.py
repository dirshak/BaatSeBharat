import json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from langchain_core.messages import AIMessage

def create_technical_analyst(llm):
    def technical_analyst_node(state):
        ticker = state["company_of_interest"]
        current_date_str = state["trade_date"]
        
        # Load historical daily data (1 year lookback to calculate moving averages)
        current_date = datetime.strptime(current_date_str, "%Y-%m-%d")
        start_date = current_date - timedelta(days=365)
        
        try:
            df = yf.Ticker(ticker).history(start=start_date.strftime("%Y-%m-%d"), end=(current_date + timedelta(days=1)).strftime("%Y-%m-%d"))
            if df.empty or len(df) < 20:
                raise ValueError("Not enough market data to calculate indicators.")
            
            # Ensure timezone-naive
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
                
            # Filter to current_date
            df = df[df.index <= current_date]
            
            # Calculations
            df['50_SMA'] = df['Close'].rolling(window=50).mean()
            df['200_SMA'] = df['Close'].rolling(window=200).mean()
            df['10_EMA'] = df['Close'].ewm(span=10, adjust=False).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema26 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = ema12 - ema26
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            
            # Bollinger Bands
            sma20 = df['Close'].rolling(window=20).mean()
            std20 = df['Close'].rolling(window=20).std()
            df['Boll_Middle'] = sma20
            df['Boll_Upper'] = sma20 + 2 * std20
            df['Boll_Lower'] = sma20 - 2 * std20
            
            # ATR
            high_low = df['High'] - df['Low']
            high_cp = (df['High'] - df['Close'].shift()).abs()
            low_cp = (df['Low'] - df['Close'].shift()).abs()
            tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
            df['ATR'] = tr.rolling(window=14).mean()
            
            # Momentum & Volume
            df['Momentum_10D'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)
            df['Volume_MA5'] = df['Volume'].rolling(5).mean()
            df['Volume_MA20'] = df['Volume'].rolling(20).mean()
            
            # OBV
            df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
            
            # Support / Resistance
            df['Support_20D'] = df['Low'].rolling(window=20).min()
            df['Resistance_20D'] = df['High'].rolling(window=20).max()
            
            # Volatility & Drawdown
            df['Returns'] = df['Close'].pct_change()
            df['Rolling_Vol'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
            
            roll_max = df['Close'].cummax()
            df['Drawdown'] = (df['Close'] - roll_max) / roll_max
            
            # Get latest values
            latest = df.iloc[-1]
            close_price = float(latest['Close'])
            rsi_val = float(latest['RSI']) if not pd.isna(latest['RSI']) else 50.0
            macd_val = float(latest['MACD']) if not pd.isna(latest['MACD']) else 0.0
            macd_sig = float(latest['MACD_Signal']) if not pd.isna(latest['MACD_Signal']) else 0.0
            sma50 = float(latest['50_SMA']) if not pd.isna(latest['50_SMA']) else close_price
            sma200 = float(latest['200_SMA']) if not pd.isna(latest['200_SMA']) else close_price
            mom10 = float(latest['Momentum_10D']) if not pd.isna(latest['Momentum_10D']) else 0.0
            atr_val = float(latest['ATR']) if not pd.isna(latest['ATR']) else (close_price * 0.02)
            vol_ratio = float(latest['Volume_MA5'] / latest['Volume_MA20']) if latest['Volume_MA20'] > 0 else 1.0
            
            # Scoring
            score = 0.0
            supporting = []
            conflicting = []
            
            # Trend component
            if close_price > sma50:
                score += 0.25
                supporting.append(f"Price ({close_price:.2f}) is above 50-day SMA ({sma50:.2f})")
            else:
                score -= 0.25
                conflicting.append(f"Price ({close_price:.2f}) is below 50-day SMA ({sma50:.2f})")
                
            if close_price > sma200:
                score += 0.25
                supporting.append(f"Price is above 200-day SMA ({sma200:.2f})")
            else:
                score -= 0.25
                conflicting.append(f"Price is below 200-day SMA ({sma200:.2f})")
                
            # RSI component
            if rsi_val < 35:
                score += 0.25
                supporting.append(f"RSI is oversold ({rsi_val:.1f})")
            elif rsi_val > 65:
                score -= 0.25
                conflicting.append(f"RSI is overbought ({rsi_val:.1f})")
            else:
                supporting.append(f"RSI is neutral ({rsi_val:.1f})")
                
            # MACD component
            if macd_val > macd_sig:
                score += 0.25
                supporting.append(f"MACD line is above signal line (bullish crossover)")
            else:
                score -= 0.25
                conflicting.append(f"MACD line is below signal line (bearish crossover)")
                
            # Momentum component
            if mom10 > 0:
                score += 0.25
                supporting.append(f"10-Day momentum is positive ({mom10*100:+.2f}%)")
            else:
                score -= 0.25
                conflicting.append(f"10-Day momentum is negative ({mom10*100:+.2f}%)")
                
            normalized_score = float(np.clip(score, -1.0, 1.0))
            
            evidence = {
                "ticker": ticker,
                "close_price": close_price,
                "sma_50": sma50,
                "sma_200": sma200,
                "ema_10": float(latest['10_EMA']),
                "rsi": rsi_val,
                "macd": macd_val,
                "macd_signal": macd_sig,
                "macd_hist": float(latest['MACD_Hist']),
                "boll_middle": float(latest['Boll_Middle']),
                "boll_upper": float(latest['Boll_Upper']),
                "boll_lower": float(latest['Boll_Lower']),
                "atr": atr_val,
                "momentum_10d": mom10,
                "volume_ratio": vol_ratio,
                "obv": float(latest['OBV']),
                "support_20d": float(latest['Support_20D']),
                "resistance_20d": float(latest['Resistance_20D']),
                "rolling_vol": float(latest['Rolling_Vol']),
                "drawdown": float(latest['Drawdown']),
                "technical_score": normalized_score,
                "supporting_indicators": supporting,
                "conflicting_indicators": conflicting
            }
        except Exception as e:
            # Degrade gracefully
            evidence = {
                "ticker": ticker,
                "error": str(e),
                "technical_score": 0.0,
                "supporting_indicators": [],
                "conflicting_indicators": ["Failed to load technical indicators"]
            }
            
        report_str = json.dumps(evidence, indent=2)
        
        return {
            "messages": [AIMessage(content=f"Technical analyst report: {report_str}")],
            "technical_report": report_str,
            "technical_evidence": evidence
        }
        
    return technical_analyst_node
