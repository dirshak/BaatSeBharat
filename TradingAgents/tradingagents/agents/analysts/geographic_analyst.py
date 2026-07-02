import json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from langchain_core.messages import AIMessage

COUNTRY_MAP = {
    ".NS": ("India", "^NSEI"),
    ".BO": ("India", "^BSESN"),
    ".T": ("Japan", "^N225"),
    ".HK": ("Hong Kong", "^HSI"),
    ".L": ("United Kingdom", "^FTSE"),
    ".TO": ("Canada", "^GSPTSE"),
    ".AX": ("Australia", "^AXJO"),
    ".SS": ("China", "000001.SS"),
    ".SZ": ("China", "399001.SZ"),
    "": ("United States", "SPY")
}

def create_geographic_analyst(llm):
    def geographic_analyst_node(state):
        ticker = state["company_of_interest"]
        current_date_str = state["trade_date"]
        current_date = datetime.strptime(current_date_str, "%Y-%m-%d")
        start_date = current_date - timedelta(days=90)
        
        # Determine country and benchmark index based on suffix
        country = "United States"
        bench_ticker = "SPY"
        ticker_upper = ticker.upper()
        
        for suffix, (c, b) in COUNTRY_MAP.items():
            if suffix and ticker_upper.endswith(suffix):
                country = c
                bench_ticker = b
                break
                
        try:
            df = yf.Ticker(bench_ticker).history(start=start_date.strftime("%Y-%m-%d"), end=(current_date + timedelta(days=1)).strftime("%Y-%m-%d"))
            if df.empty or len(df) < 20:
                raise ValueError("Not enough benchmark data.")
                
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df = df[df.index <= current_date]
            
            prices = df['Close'].values
            returns = df['Close'].pct_change().dropna()
            
            # Country benchmark returns
            ret_5d = float((prices[-1] - prices[-min(5, len(prices))]) / prices[-min(5, len(prices))]) if len(prices) >= 5 else 0.0
            ret_20d = float((prices[-1] - prices[0]) / prices[0])
            
            # Volatility shift: 5-day rolling volatility vs 20-day
            vol_5d = float(returns.iloc[-5:].std() * np.sqrt(252)) if len(returns) >= 5 else 0.15
            vol_20d = float(returns.iloc[-20:].std() * np.sqrt(252)) if len(returns) >= 20 else 0.15
            vol_shift = float(vol_5d - vol_20d)
            
            # Index Drawdown
            roll_max = df['Close'].cummax()
            drawdown = float(((df['Close'] - roll_max) / roll_max).iloc[-1])
            
            # Country Risk Score (1 to 10 scale)
            # Baseline is 3; add risk for volatility and drawdown
            risk_score = 3.0
            if vol_5d > 0.20:
                risk_score += 2.0
            if vol_5d > 0.30:
                risk_score += 2.0
            if drawdown < -0.05:
                risk_score += 1.0
            if drawdown < -0.15:
                risk_score += 2.0
            if vol_shift > 0.05:
                risk_score += 1.0
            risk_score = int(np.clip(risk_score, 1, 10))
            
            evidence = {
                "country": country,
                "benchmark": bench_ticker,
                "bench_return_5d": ret_5d,
                "bench_return_20d": ret_20d,
                "volatility_5d": vol_5d,
                "volatility_20d": vol_20d,
                "volatility_shift": vol_shift,
                "drawdown": drawdown,
                "risk_score": risk_score,
                "mapped_exposure": {
                    "domestic_pct": 100.0 if country == "India" else 90.0,
                    "global_pct": 0.0 if country == "India" else 10.0
                }
            }
        except Exception as e:
            evidence = {
                "country": country,
                "benchmark": bench_ticker,
                "error": str(e),
                "risk_score": 5,
                "mapped_exposure": {
                    "domestic_pct": 100.0,
                    "global_pct": 0.0
                }
            }
            
        report_str = json.dumps(evidence, indent=2)
        
        return {
            "messages": [AIMessage(content=f"Geographic analyst report: {report_str}")],
            "geographic_report": report_str,
            "geographic_evidence": evidence
        }
        
    return geographic_analyst_node
