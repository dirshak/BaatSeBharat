import json
import re
from langchain_core.messages import AIMessage

def create_bull_researcher(llm):
    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")
        
        # Retrieve all 6 evidence packets
        market_ev = state.get("market_evidence", {})
        fund_ev = state.get("fundamentals_evidence", {})
        tech_ev = state.get("technical_evidence", {})
        quant_ev = state.get("quant_evidence", {})
        sec_ev = state.get("sector_evidence", {})
        geo_ev = state.get("geographic_evidence", {})
        
        # Assemble collective evidence dict
        evidence = {
            "market": market_ev,
            "fundamentals": fund_ev,
            "technical": tech_ev,
            "quant": quant_ev,
            "sector": sec_ev,
            "geographic": geo_ev
        }
        
        evidence_json = json.dumps(evidence, indent=2)
        
        # Strict Prompt for Ollama
        system_prompt = (
            "You are a Bull Researcher. Your task is to analyze the provided JSON evidence "
            "and present a bullish thesis, invalidation conditions, and a confidence score.\n"
            "CRITICAL: Every statement must be strictly grounded in the provided JSON evidence. "
            "Do NOT invent or assume any facts, news, prices, or indicators. Do NOT refer to external events. "
            "If some fields are missing, explicitly state so.\n\n"
            "Format your response as a valid JSON object matching the following structure:\n"
            "{\n"
            "  \"thesis\": \"A comprehensive explanation of the bullish case based on the evidence.\",\n"
            "  \"invalidation_conditions\": [\"Condition 1\", \"Condition 2\"],\n"
            "  \"confidence\": 0.85\n"
            "}"
        )
        
        user_message = f"Here is the JSON evidence:\n\n{evidence_json}\n\nProduce the JSON response now."
        
        thesis_data = None
        
        # Try calling Ollama/LLM
        try:
            # We construct a prompt list or plain prompt
            response = llm.invoke(f"{system_prompt}\n\n{user_message}")
            content = response.content
            
            # Parse JSON out of content
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                thesis_data = json.loads(match.group(0))
        except Exception:
            pass # Fallback to deterministic template
            
        # Deterministic Template Fallback
        if not thesis_data:
            supporting = tech_ev.get("supporting_indicators", [])
            regime = market_ev.get("market_regime", "Transitional")
            expected_5d = quant_ev.get("horizon_5d", {}).get("expected_return", 0.0)
            
            thesis_text = (
                f"The bullish case is supported by favorable technical and quant indicators. "
                f"Specifically, we observe: {', '.join(supporting[:3]) if supporting else 'neutral momentum'}. "
                f"The market regime is currently {regime}. "
                f"The local quant model predicts 5-day expected return of {expected_5d * 100:+.2f}%. "
                f"Sector rank is {sec_ev.get('target_sector_info', {}).get('rank', 1)} and country risk score is {geo_ev.get('risk_score', 5)}/10."
            )
            
            inval_conds = [
                "Price crosses below the 50-day moving average.",
                "RSI indicator becomes overbought (>70) or momentum shifts negative.",
                f"Market regime transitions to Volatile."
            ]
            
            # Average technical score & quant confidence
            tech_score = tech_ev.get("technical_score", 0.0)
            quant_conf = quant_ev.get("horizon_5d", {}).get("confidence", 0.5)
            confidence = float(max(0.1, min(1.0, (tech_score + 1.0) / 4.0 + quant_conf / 2.0)))
            
            thesis_data = {
                "thesis": thesis_text,
                "invalidation_conditions": inval_conds,
                "confidence": confidence
            }
            
        argument = f"Bull Analyst Thesis:\n{thesis_data['thesis']}\nInvalidation Conditions: {thesis_data['invalidation_conditions']}\nConfidence: {thesis_data['confidence']:.2f}"
        
        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
            # store structured data too
            "bull_thesis": thesis_data
        }
        
        return {"investment_debate_state": new_investment_debate_state}
        
    return bull_node
