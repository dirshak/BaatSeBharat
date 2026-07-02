import json
import re
from langchain_core.messages import AIMessage

def create_bear_researcher(llm):
    def bear_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bear_history = investment_debate_state.get("bear_history", "")
        
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
            "You are a Bear Researcher. Your task is to analyze the provided JSON evidence "
            "and present a bearish thesis, invalidation conditions, and a confidence score.\n"
            "CRITICAL: Every statement must be strictly grounded in the provided JSON evidence. "
            "Do NOT invent or assume any facts, news, prices, or indicators. Do NOT refer to external events. "
            "If some fields are missing, explicitly state so.\n\n"
            "Format your response as a valid JSON object matching the following structure:\n"
            "{\n"
            "  \"thesis\": \"A comprehensive explanation of the bearish case based on the evidence.\",\n"
            "  \"invalidation_conditions\": [\"Condition 1\", \"Condition 2\"],\n"
            "  \"confidence\": 0.85\n"
            "}"
        )
        
        user_message = f"Here is the JSON evidence:\n\n{evidence_json}\n\nProduce the JSON response now."
        
        thesis_data = None
        
        # Try calling Ollama/LLM
        try:
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
            conflicting = tech_ev.get("conflicting_indicators", [])
            regime = market_ev.get("market_regime", "Transitional")
            expected_5d = quant_ev.get("horizon_5d", {}).get("expected_return", 0.0)
            drawdown = tech_ev.get("drawdown", 0.0)
            
            thesis_text = (
                f"The bearish case highlights technical and quant risks. "
                f"Specifically, we observe: {', '.join(conflicting[:3]) if conflicting else 'neutral/weak trend'}. "
                f"The current market drawdown is {drawdown * 100:.2f}%, and the market regime is {regime}. "
                f"The quant model forecasts 5-day return of {expected_5d * 100:+.2f}%. "
                f"Sector expected return is {sec_ev.get('target_sector_info', {}).get('expected_return', 0.0) * 100:+.2f}%."
            )
            
            inval_conds = [
                "Price crosses above the 50-day moving average.",
                "RSI indicator becomes oversold (<30) indicating a potential bounce.",
                f"Market regime transitions to Stable."
            ]
            
            # Average technical score (negative direction) & quant bearishness
            tech_score = tech_ev.get("technical_score", 0.0)
            quant_conf = quant_ev.get("horizon_5d", {}).get("confidence", 0.5)
            confidence = float(max(0.1, min(1.0, (1.0 - tech_score) / 4.0 + quant_conf / 2.0)))
            
            thesis_data = {
                "thesis": thesis_text,
                "invalidation_conditions": inval_conds,
                "confidence": confidence
            }
            
        argument = f"Bear Analyst Thesis:\n{thesis_data['thesis']}\nInvalidation Conditions: {thesis_data['invalidation_conditions']}\nConfidence: {thesis_data['confidence']:.2f}"
        
        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bear_history": bear_history + "\n" + argument,
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
            # store structured data too
            "bear_thesis": thesis_data
        }
        
        return {"investment_debate_state": new_investment_debate_state}
        
    return bear_node
