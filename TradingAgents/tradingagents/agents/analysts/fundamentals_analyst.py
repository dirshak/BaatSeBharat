import json
import yfinance as yf
from langchain_core.messages import AIMessage

def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        ticker = state["company_of_interest"]
        
        confidence = 0.80
        missing_fields = []
        
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info or {}
            
            # Extract key fundamentals
            pe_ratio = info.get("trailingPE")
            forward_pe = info.get("forwardPE")
            peg_ratio = info.get("pegRatio")
            rev_growth = info.get("revenueGrowth")
            profit_margins = info.get("profitMargins")
            debt_to_equity = info.get("debtToEquity")
            fcf = info.get("freeCashflow")
            roe = info.get("returnOnEquity")
            
            # Check for missing crucial fields and reduce confidence
            fields_to_check = {
                "trailingPE": pe_ratio,
                "revenueGrowth": rev_growth,
                "profitMargins": profit_margins,
                "debtToEquity": debt_to_equity,
                "returnOnEquity": roe
            }
            
            for key, val in fields_to_check.items():
                if val is None:
                    confidence -= 0.10
                    missing_fields.append(key)
            
            confidence = max(0.10, round(confidence, 2))
            
            evidence = {
                "ticker": ticker,
                "pe_ratio": pe_ratio,
                "forward_pe": forward_pe,
                "peg_ratio": peg_ratio,
                "revenue_growth": rev_growth,
                "profit_margins": profit_margins,
                "debt_to_equity": debt_to_equity,
                "free_cash_flow": fcf,
                "return_on_equity": roe,
                "confidence_score": confidence,
                "missing_fields": missing_fields
            }
        except Exception as e:
            evidence = {
                "ticker": ticker,
                "error": str(e),
                "confidence_score": 0.10,
                "missing_fields": ["ALL_FIELDS_FAILED_TO_LOAD"]
            }
            
        report_str = json.dumps(evidence, indent=2)
        
        return {
            "messages": [AIMessage(content=f"Fundamentals analyst report: {report_str}")],
            "fundamentals_report": report_str,
            "fundamentals_evidence": evidence
        }
        
    return fundamentals_analyst_node
