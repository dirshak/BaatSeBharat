import json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from langchain_core.messages import AIMessage

def create_market_analyst(llm):
    def market_analyst_node(state):
        ticker = state["company_of_interest"]
        current_date_str = state["trade_date"]
        current_date = datetime.strptime(current_date_str, "%Y-%m-%d")
        start_date = current_date - timedelta(days=120)
        
        # Indian vs US benchmark index
        is_indian = ticker.endswith(".NS") or ticker.endswith(".BO") or ".NS" in ticker or ".BO" in ticker
        bench_ticker = "^NSEI" if is_indian else "SPY"
        
        try:
            df = yf.Ticker(bench_ticker).history(start=start_date.strftime("%Y-%m-%d"), end=(current_date + timedelta(days=1)).strftime("%Y-%m-%d"))
            if df.empty or len(df) < 20:
                raise ValueError("Not enough index data.")
                
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df = df[df.index <= current_date]
            
            prices = df['Close'].values
            returns = df['Close'].pct_change().dropna()
            
            # Benchmark returns
            ret_5d = float((prices[-1] - prices[-min(5, len(prices))]) / prices[-min(5, len(prices))]) if len(prices) >= 5 else 0.0
            ret_20d = float((prices[-1] - prices[0]) / prices[0])
            
            # Breadth proxy: % of up days in last 20 days
            breadth = float((returns.iloc[-20:] > 0).mean()) if len(returns) >= 20 else 0.50
            
            # Volatility (rolling 20-day index volatility)
            vol = float(returns.rolling(20).std().iloc[-1] * np.sqrt(252)) if len(returns) >= 20 else 0.15
            
            # Drawdown
            roll_max = df['Close'].cummax()
            drawdown = float(((df['Close'] - roll_max) / roll_max).iloc[-1])
            
            # Risk-on/risk-off score: scaled 1 to 10
            # Higher momentum and breadth -> higher score
            base_score = 5.0
            base_score += ret_20d * 20.0
            base_score += (breadth - 0.5) * 6.0
            base_score -= (vol - 0.15) * 4.0
            risk_on_off_score = int(np.clip(base_score, 1, 10))
            
            # Market regime determination
            if vol < 0.15 and ret_20d > 0:
                regime = "Stable"
            elif vol > 0.22 or drawdown < -0.07:
                regime = "Volatile"
            else:
                regime = "Transitional"
                
            evidence = {
                "benchmark_ticker": bench_ticker,
                "benchmark_return_5d": ret_5d,
                "benchmark_return_20d": ret_20d,
                "breadth_proxy": breadth,
                "volatility": vol,
                "drawdown": drawdown,
                "risk_on_off_score": risk_on_off_score,
                "market_regime": regime
            }
        except Exception as e:
            evidence = {
                "benchmark_ticker": bench_ticker,
                "error": str(e),
                "risk_on_off_score": 5,
                "market_regime": "Transitional"
            }
            
        report_str = json.dumps(evidence, indent=2)
        
        return {
            "messages": [AIMessage(content=f"Market analyst report: {report_str}")],
            "market_report": report_str,
            "market_evidence": evidence
        }
        
    return market_analyst_node
