"""Portfolio Manager: final decision-maker with paper-trading ledger.

Synthesises the Risk Manager's risk review and the Trader's proposal into
the final trading decision.  After every run the decision is written to a
local JSON paper-trading ledger at ``data/simulated_ledger.json`` so that
P&L can be tracked across sessions.

Ledger schema (one entry per trade):
{
    "ticker":          "RELIANCE.NS",
    "date":            "2024-05-10",
    "action":          "Buy",           // rating enum value
    "position_size":   5000.0,          // INR / USD position value
    "entry_price":     null,            // float or null
    "stop_loss":       null,            // float or null
    "rationale":       "...",
    "risk_level":      "LOW",
    "composite_score": 0.312,
    "exit_price":      null,            // filled in on close
    "pnl":             null,            // filled in on close
    "status":          "OPEN"           // OPEN | CLOSED
}
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from tradingagents.agents.schemas import PortfolioDecision, render_pm_decision
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)

logger = logging.getLogger(__name__)

# Default ledger location (relative to project root).
_DEFAULT_LEDGER_PATH = Path("data") / "simulated_ledger.json"


def _load_ledger(path: Path) -> list:
    """Load the ledger from disk; return empty list if file does not exist."""
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read ledger at %s: %s", path, exc)
    return []


def _save_ledger(path: Path, entries: list) -> None:
    """Atomically write the ledger to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
        tmp.replace(path)
    except OSError as exc:
        logger.error("Could not write ledger to %s: %s", path, exc)
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _extract_entry_price(trader_plan: str) -> Optional[float]:
    """Try to parse **Entry Price**: <number> from the trader plan."""
    m = re.search(r"\*\*Entry Price\*\*\s*[:\-]\s*([\d,\.]+)", trader_plan)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass
    return None


def _extract_stop_loss(trader_plan: str) -> Optional[float]:
    """Try to parse **Stop Loss**: <number> from the trader plan."""
    m = re.search(r"\*\*Stop Loss\*\*\s*[:\-]\s*([\d,\.]+)", trader_plan)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass
    return None


def _extract_composite_score(state: dict) -> float:
    """Return the average composite signal score from evidence packets."""
    packets = [
        state.get("market_evidence", {}),
        state.get("fundamentals_evidence", {}),
        state.get("technical_evidence", {}),
        state.get("quant_evidence", {}),
        state.get("sector_evidence", {}),
        state.get("geographic_evidence", {}),
    ]
    scores = []
    for pkt in packets:
        score = pkt.get("score")
        if score is not None:
            try:
                scores.append(float(score))
            except (TypeError, ValueError):
                pass
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def _extract_risk_level(risk_review: str) -> str:
    """Parse **Risk Level**: <value> from the risk review text."""
    m = re.search(r"\*\*Risk Level\*\*\s*[:\-]\s*(\w+)", risk_review)
    return m.group(1).upper() if m else "UNKNOWN"


def _estimate_position_size(state: dict, max_pct: float = 0.05) -> float:
    """Estimate position size based on portfolio value and max allocation.

    Uses the portfolio_value field from the risk review text if available,
    falling back to $100 000 default.
    """
    risk_review = state.get("risk_debate_state", {}).get("history", "")
    m = re.search(
        r"Max position size\s*=\s*([\d,]+)", risk_review
    )
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass
    # Default: 5 % of $100 000
    return 100_000.0 * max_pct


def create_portfolio_manager(
    llm,
    ledger_path: str | Path | None = None,
):
    """Factory for the portfolio manager node.

    Args:
        llm: LLM instance (Ollama / any LangChain-compatible LLM).
        ledger_path: Path to the JSON paper-trading ledger.  Defaults to
            ``data/simulated_ledger.json`` relative to the working directory.
    """
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")
    effective_ledger_path = Path(ledger_path) if ledger_path else _DEFAULT_LEDGER_PATH

    def portfolio_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)

        risk_review = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )

        prompt = (
            f"As the Portfolio Manager, synthesize the Risk Manager's review and "
            f"deliver the final trading decision.\n\n"
            f"{instrument_context}\n\n"
            f"---\n\n"
            f"**Rating Scale** (use exactly one):\n"
            f"- **Buy**: Strong conviction to enter or add to position\n"
            f"- **Overweight**: Favorable outlook, gradually increase exposure\n"
            f"- **Hold**: Maintain current position, no action needed\n"
            f"- **Underweight**: Reduce exposure, take partial profits\n"
            f"- **Sell**: Exit position or avoid entry\n\n"
            f"**Context:**\n"
            f"- Research Manager's investment plan: **{research_plan}**\n"
            f"- Trader's transaction proposal: **{trader_plan}**\n"
            f"{lessons_line}"
            f"**Risk Manager Review:**\n{risk_review}\n\n"
            f"---\n\n"
            f"Be decisive and ground every conclusion in specific evidence from the "
            f"analysts.{get_language_instruction()}"
        )

        final_trade_decision = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_pm_decision,
            "Portfolio Manager",
        )

        new_risk_debate_state = {
            "judge_decision": final_trade_decision,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state.get("aggressive_history", ""),
            "conservative_history": risk_debate_state.get("conservative_history", ""),
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Portfolio Manager",
            "current_aggressive_response": risk_debate_state.get("current_aggressive_response", ""),
            "current_conservative_response": risk_debate_state.get("current_conservative_response", ""),
            "current_neutral_response": risk_debate_state.get("current_neutral_response", ""),
            "count": risk_debate_state["count"],
        }

        # ------------------------------------------------------------------ #
        # Paper-trading ledger persistence
        # ------------------------------------------------------------------ #
        try:
            ticker = state.get("company_of_interest", "UNKNOWN")
            trade_date = state.get("trade_date", datetime.utcnow().strftime("%Y-%m-%d"))
            composite_score = _extract_composite_score(state)
            risk_level = _extract_risk_level(risk_review)
            position_size = _estimate_position_size(state)
            entry_price = _extract_entry_price(trader_plan)
            stop_loss = _extract_stop_loss(trader_plan)

            # Parse the final action from the decision text
            decision_upper = final_trade_decision.upper()
            action = "Hold"
            for a in ("BUY", "OVERWEIGHT", "UNDERWEIGHT", "SELL", "HOLD"):
                if a in decision_upper:
                    action = a.capitalize() if a not in ("BUY", "SELL") else a.capitalize()
                    break

            ledger_entry: Dict[str, Any] = {
                "ticker": ticker,
                "date": str(trade_date),
                "action": action,
                "position_size": position_size,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "rationale": final_trade_decision[:500],  # abbreviated
                "risk_level": risk_level,
                "composite_score": composite_score,
                "exit_price": None,
                "pnl": None,
                "status": "OPEN",
            }

            entries = _load_ledger(effective_ledger_path)
            entries.append(ledger_entry)
            _save_ledger(effective_ledger_path, entries)

            logger.info(
                "Ledger updated → %s %s on %s (score=%.3f, risk=%s)",
                action,
                ticker,
                trade_date,
                composite_score,
                risk_level,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to update paper-trading ledger: %s", exc, exc_info=True)

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
        }

    return portfolio_manager_node
