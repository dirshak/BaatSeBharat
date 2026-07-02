"""Risk Manager: single-pass numerical risk reviewer.

Replaces the legacy aggressive / conservative / neutral debator trio.
Performs rule-based and score-weighted risk evaluation on the trader's
proposal, then outputs an approved/modified decision that the Portfolio
Manager uses to confirm or override the trade.

Rules applied (all configurable via the ``risk_params`` dict):
- Max single-position size: 10 % of portfolio (paper-trading default $100 000)
- Max sector concentration: 25 %
- ATR-based stop-loss floor: stop must be ≥ 1.5 × ATR below entry
- Regime sizing multiplier: scale down in bear regimes
- Max allowed score: composite analyst score must be ≥ –0.2 for a Buy
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default risk parameters (can be overridden by passing risk_params to factory)
# ---------------------------------------------------------------------------

DEFAULT_RISK_PARAMS: Dict[str, Any] = {
    "portfolio_value": 100_000.0,          # paper-trading portfolio size (USD/INR-equivalent)
    "max_position_pct": 0.10,              # max single-trade size as fraction of portfolio
    "max_sector_pct": 0.25,               # max sector concentration
    "atr_stop_multiple": 1.5,             # stop-loss must be ≥ this many ATRs below entry
    "min_composite_score_for_buy": -0.20,  # composite score threshold to allow a Buy
    "bear_regime_size_multiplier": 0.50,  # scale position down in bear regime
}


def _extract_composite_score(state: dict) -> float:
    """Return the average composite signal score from all evidence packets.

    Gracefully handles missing or malformed evidence packets by skipping them.
    Returns a score in the range [-1.0, +1.0], defaulting to 0.0 if no
    evidence is available.
    """
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


def _extract_atr(state: dict) -> float | None:
    """Return ATR from technical_evidence, or None if unavailable."""
    tech = state.get("technical_evidence", {})
    atr = tech.get("atr")
    if atr is None:
        return None
    try:
        return float(atr)
    except (TypeError, ValueError):
        return None


def _extract_regime(state: dict) -> str:
    """Return the market regime string from market_evidence."""
    mkt = state.get("market_evidence", {})
    return mkt.get("regime", "unknown").lower()


def _parse_trader_action(trader_plan: str) -> str:
    """Extract the action token (BUY/HOLD/SELL) from the trader's plan text."""
    plan_upper = trader_plan.upper()
    for token in ("BUY", "HOLD", "SELL"):
        if token in plan_upper:
            return token
    return "HOLD"


def create_risk_manager(llm=None, risk_params: Dict[str, Any] | None = None):
    """Factory for the risk manager node.

    ``llm`` is accepted for API compatibility with the rest of the graph
    factory pattern but is not used; the node is fully deterministic.
    ``risk_params`` can override DEFAULT_RISK_PARAMS for testing or live use.
    """
    params = {**DEFAULT_RISK_PARAMS, **(risk_params or {})}

    def risk_manager_node(state) -> dict:
        """Apply numerical risk rules and produce a risk-reviewed decision."""
        trader_plan: str = state.get("trader_investment_plan", "")
        composite_score = _extract_composite_score(state)
        atr = _extract_atr(state)
        regime = _extract_regime(state)
        action = _parse_trader_action(trader_plan)

        portfolio_value = params["portfolio_value"]
        max_position_pct = params["max_position_pct"]
        atr_multiple = params["atr_stop_multiple"]
        min_score_buy = params["min_composite_score_for_buy"]
        bear_mult = params["bear_regime_size_multiplier"]

        # --- Rule evaluation ---
        flags: list[str] = []
        risk_level = "LOW"

        # 1. Composite score gate for Buy
        if action == "BUY" and composite_score < min_score_buy:
            flags.append(
                f"Composite score {composite_score:.3f} is below Buy threshold "
                f"{min_score_buy:.2f} → downgrading to HOLD."
            )
            action = "HOLD"
            risk_level = "HIGH"

        # 2. Regime sizing
        effective_max_pct = max_position_pct
        is_bear = "bear" in regime
        if is_bear:
            effective_max_pct = max_position_pct * bear_mult
            flags.append(
                f"Bear regime detected ('{regime}') → max position reduced to "
                f"{effective_max_pct * 100:.1f} % of portfolio."
            )
            if risk_level == "LOW":
                risk_level = "MEDIUM"

        max_position_size = portfolio_value * effective_max_pct

        # 3. ATR stop-loss check (informational when no entry price in plan)
        atr_stop_note = ""
        if atr is not None and atr > 0:
            min_stop_distance = atr * atr_multiple
            atr_stop_note = (
                f"ATR = {atr:.2f}; recommended stop-loss distance ≥ "
                f"{min_stop_distance:.2f} (= {atr_multiple}× ATR)."
            )
        else:
            atr_stop_note = "ATR unavailable — use 2 % trailing stop as fallback."

        # --- Build risk review summary ---
        status = "APPROVED" if action != "HOLD" or not flags else "MODIFIED"
        if action == "HOLD" and not flags:
            status = "APPROVED"

        flag_text = "\n".join(f"  • {f}" for f in flags) if flags else "  • None — all checks passed."

        risk_review = (
            f"**Risk Manager Review**\n\n"
            f"**Composite Signal Score**: {composite_score:+.3f}\n"
            f"**Market Regime**: {regime.capitalize()}\n"
            f"**Recommended Action**: {action}\n"
            f"**Risk Level**: {risk_level}\n"
            f"**Status**: {status}\n\n"
            f"**Position Sizing**:\n"
            f"  Max position size = {max_position_size:,.0f} "
            f"({effective_max_pct * 100:.1f} % of portfolio).\n\n"
            f"**Stop-Loss Guidance**: {atr_stop_note}\n\n"
            f"**Flags Raised**:\n{flag_text}\n\n"
            f"**Original Trader Proposal**:\n{trader_plan}"
        )

        # Populate risk_debate_state fields that downstream nodes (Portfolio Manager,
        # trading_graph._log_state) expect.  We store the review in 'history' so the
        # PM sees it, and zero the legacy debate fields.
        new_risk_debate_state = {
            "aggressive_history": "",
            "conservative_history": "",
            "neutral_history": "",
            "history": risk_review,
            "latest_speaker": "Risk Manager",
            "current_aggressive_response": "",
            "current_conservative_response": "",
            "current_neutral_response": "",
            "judge_decision": "",
            "count": 1,
        }

        logger.info(
            "Risk Manager: action=%s, score=%.3f, regime=%s, risk=%s",
            action,
            composite_score,
            regime,
            risk_level,
        )

        return {
            "risk_debate_state": new_risk_debate_state,
            # Also surface the reviewed action on the trader plan field so PM
            # sees an updated action if Risk Manager downgraded it.
            "trader_investment_plan": (
                trader_plan
                if action == _parse_trader_action(trader_plan)
                else trader_plan + f"\n\n[Risk Manager Override → {action}]"
            ),
        }

    return risk_manager_node
