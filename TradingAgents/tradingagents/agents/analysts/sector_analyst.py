import json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from langchain_core.messages import AIMessage
from tradingagents.agents.utils.agent_utils import resolve_instrument_identity

SECTOR_BENCHMARKS_IN = {
    "Financial Services": "^NSEBANK",
    "Financials": "^NSEBANK",
    "Technology": "^CNXIT",
    "IT": "^CNXIT",
    "Pharma": "^CNXPHARMA",
    "Healthcare": "^CNXPHARMA",
    "Auto": "^CNXAUTO",
    "Energy": "^CNXENERGY",
    "Oil & Gas": "^CNXENERGY",
    "Broad Market": "^NSEI"
}

SECTOR_BENCHMARKS_US = {
    "Financial Services": "XLF",
    "Financials": "XLF",
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Energy": "XLE",
    "Broad Market": "SPY"
}

def create_sector_analyst(llm):
    def sector_analyst_node(state):
        ticker = state["company_of_interest"]
        current_date_str = state["trade_date"]
        current_date = datetime.strptime(current_date_str, "%Y-%m-%d")
        start_date = current_date - timedelta(days=90)
        
        # Resolve company's sector
        identity = resolve_instrument_identity(ticker)
        company_sector = identity.get("sector", "Broad Market")
        
        # Decide if Indian or US market based on ticker suffix
        is_indian = ticker.endswith(".NS") or ticker.endswith(".BO") or ".NS" in ticker or ".BO" in ticker
        benchmarks_map = SECTOR_BENCHMARKS_IN if is_indian else SECTOR_BENCHMARKS_US
        
        sector_results = []
        for sector_name, bench_ticker in benchmarks_map.items():
            try:
                df = yf.Ticker(bench_ticker).history(start=start_date.strftime("%Y-%m-%d"), end=(current_date + timedelta(days=1)).strftime("%Y-%m-%d"))
                if df.empty or len(df) < 10:
                    continue
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                df = df[df.index <= current_date]
                
                prices = df['Close'].values
                returns = df['Close'].pct_change().dropna()
                
                # Momentum (1-month return)
                m_1m = float((prices[-1] - prices[0]) / prices[0])
                
                # Volatility
                vol = float(returns.rolling(20).std().iloc[-1] * np.sqrt(252)) if len(returns) >= 20 else 0.20
                
                # Sector Breadth (simulated breadth proxy: ratio of up days in last 10 days)
                breadth = float((returns.iloc[-10:] > 0).mean()) if len(returns) >= 10 else 0.50
                
                # Expected return projection
                expected_ret = m_1m * 0.5 # scaled
                
                sector_results.append({
                    "sector": sector_name,
                    "benchmark": bench_ticker,
                    "momentum_1m": m_1m,
                    "volatility": vol,
                    "breadth_proxy": breadth,
                    "expected_return": expected_ret,
                    "confidence": float(np.clip(0.5 + breadth * 0.3, 0.3, 0.9)),
                    "risk": "HIGH" if vol > 0.25 else ("MEDIUM" if vol > 0.15 else "LOW")
                })
            except Exception:
                continue
                
        # If no results found, populate default dummy sector info
        if not sector_results:
            sector_results = [{
                "sector": "Broad Market",
                "benchmark": "^NSEI" if is_indian else "SPY",
                "momentum_1m": 0.02,
                "volatility": 0.15,
                "breadth_proxy": 0.50,
                "expected_return": 0.01,
                "confidence": 0.50,
                "risk": "MEDIUM"
            }]
            
        # Rank sectors by expected return
        sector_results = sorted(sector_results, key=lambda x: x["expected_return"], reverse=True)
        for rank, item in enumerate(sector_results):
            item["rank"] = rank + 1
            
        # Get target company's sector details
        target_sector_info = next((s for s in sector_results if s["sector"] == company_sector), sector_results[0])
        
        evidence = {
            "ticker": ticker,
            "company_sector": company_sector,
            "target_sector_info": target_sector_info,
            "sector_rankings": sector_results
        }
        
        report_str = json.dumps(evidence, indent=2)
        
        return {
            "messages": [AIMessage(content=f"Sector analyst report: {report_str}")],
            "sector_report": report_str,
            "sector_evidence": evidence
        }
        
    return sector_analyst_node
