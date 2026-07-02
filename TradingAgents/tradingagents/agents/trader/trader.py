"""Trader: turns the Research Manager's investment plan into a concrete transaction proposal.

In this offline/deterministic workflow the Trader:
1. Summarises the 6 analyst evidence packets into a composite picture.
2. Computes ATR-based entry / stop-loss levels from technical_evidence when available.
3. Derives a composite numeric score and maps it to BUY / HOLD / SELL.
4. Asks the LLM (Ollama or any LangChain-compatible provider) to fill in
   the TraderProposal schema with the score and context pre-populated.
5. Appends a mandatory research-only disclaimer.

If no LLM is reachable, falls back to a fully deterministic proposal built
solely from the numeric scores.
"""

from __future__ import annotations

import functools
import json
import logging
from typing import Optional

from langchain_core.messages import AIMessage

from tradingagents.agents.schemas import TraderAction, TraderProposal, render_trader_proposal
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)

logger = logging.getLogger(__name__)

_DISCLAIMER = (
    "\n\n---\n*Research-only simulated trade recommendation — not investment advice.*"
)


def _extract_composite_score(state: dict) -> float:
    """Average signal scores across all 6 evidence packets."""
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


def _score_to_action(score: float) -> TraderAction:
    """Map composite score → BUY / HOLD / SELL."""
    if score >= 0.15:
        return TraderAction.BUY
    if score <= -0.15:
        return TraderAction.SELL
    return TraderAction.HOLD


def _extract_atr_levels(state: dict) -> tuple[Optional[float], Optional[float]]:
    """Derive entry price and stop-loss from technical_evidence ATR."""
    tech = state.get("technical_evidence", {})
    close = tech.get("close_price")
    atr = tech.get("atr")
    if close is None or atr is None:
        return None, None
    try:
        close = float(close)
        atr = float(atr)
        entry = round(close, 2)
        stop = round(close - 1.5 * atr, 2)
        return entry, stop
    except (TypeError, ValueError):
        return None, None


def _build_evidence_summary(state: dict) -> str:
    """Render a compact evidence table for the LLM prompt."""
    keys = [
        ("market_evidence", "Market"),
        ("fundamentals_evidence", "Fundamentals"),
        ("technical_evidence", "Technical"),
        ("quant_evidence", "Quant"),
        ("sector_evidence", "Sector"),
        ("geographic_evidence", "Geographic"),
    ]
    lines = []
    for state_key, label in keys:
        pkt = state.get(state_key, {})
        score = pkt.get("score", "N/A")
        summary = pkt.get("summary", pkt.get("signal", ""))
        if len(str(summary)) > 150:
            summary = str(summary)[:147] + "..."
        lines.append(f"- **{label}** (score={score}): {summary}")
    return "\n".join(lines)


def _deterministic_proposal(state: dict, composite_score: float) -> str:
    """Build a TraderProposal entirely from numeric data — no LLM required."""
    action = _score_to_action(composite_score)
    entry, stop = _extract_atr_levels(state)
    ticker = state.get("company_of_interest", "the instrument")
    evidence_summary = _build_evidence_summary(state)

    proposal = TraderProposal(
        action=action,
        reasoning=(
            f"Composite analyst score = {composite_score:+.3f}. "
            f"Signal breakdown:\n{evidence_summary}\n"
            f"Score maps deterministically to {action.value} threshold."
        ),
        entry_price=entry,
        stop_loss=stop,
        position_sizing="≤ 5 % of portfolio (risk-manager confirmed)",
    )
    return render_trader_proposal(proposal) + _DISCLAIMER


def create_trader(llm):
    structured_llm = bind_structured(llm, TraderProposal, "Trader")

    def trader_node(state, name):
        company_name = state["company_of_interest"]
        instrument_context = get_instrument_context_from_state(state)
        investment_plan = state["investment_plan"]

        composite_score = _extract_composite_score(state)
        action = _score_to_action(composite_score)
        entry, stop = _extract_atr_levels(state)
        evidence_summary = _build_evidence_summary(state)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a trading agent converting a research plan and analyst "
                    "evidence into a concrete transaction proposal. "
                    "Anchor your reasoning in the provided scores and analyst summaries. "
                    "Be specific about the action (Buy/Hold/Sell), entry price, and "
                    "stop-loss level."
                    + get_language_instruction()
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Company: {company_name}. {instrument_context}\n\n"
                    f"**Composite analyst score**: {composite_score:+.4f} "
                    f"→ preliminary action = **{action.value}**\n\n"
                    f"**Analyst evidence summary**:\n{evidence_summary}\n\n"
                    f"**Research Manager's investment plan**:\n{investment_plan}\n\n"
                    f"Based on the above, produce a complete transaction proposal. "
                    f"If the composite score strongly disagrees with the research plan, "
                    f"resolve the conflict explicitly in your reasoning."
                    + (
                        f"\n\nSuggested entry ≈ {entry}, stop-loss ≈ {stop}."
                        if entry is not None else ""
                    )
                ),
            },
        ]

        try:
            trader_plan = invoke_structured_or_freetext(
                structured_llm,
                llm,
                messages,
                render_trader_proposal,
                "Trader",
            )
            # Append disclaimer if not already present
            if "not investment advice" not in trader_plan.lower():
                trader_plan += _DISCLAIMER
        except Exception as exc:  # noqa: BLE001
            logger.warning("Trader LLM call failed (%s); using deterministic fallback.", exc)
            trader_plan = _deterministic_proposal(state, composite_score)

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
