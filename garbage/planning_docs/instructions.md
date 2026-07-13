## Preserve the complete TradingAgents workflow without paid APIs

Retain the full TradingAgents-style multi-agent decision workflow:

```text
Market/Technical/Fundamental/Sentiment inputs
        ↓
Bull Researcher ↔ Bear Researcher debate
        ↓
Trader Agent proposal
        ↓
Risk Management review
        ↓
Portfolio Manager final approval/rejection
        ↓
Simulated trade recommendation and dashboard output
```

Do not remove the Trader Agent, researcher debate, Risk Manager, Portfolio Manager, or simulated exchange workflow.

However, replace every paid-cloud-LLM dependency and every speech-derived input with free, calculated, reproducible market-data signals.

### Data constraints

* Do not use paid APIs.
* Do not use OpenAI, Claude, Gemini, OpenRouter, Alpha Vantage, or any cloud LLM API key.
* Do not use Mann Ki Baat transcripts, speeches, text sentiment, news sentiment, social-media sentiment, embeddings, keywords, or speech signals.
* Use `yfinance` as the primary market-data source.
* Use `pandas`, `numpy`, `scikit-learn`, `ta`, and local storage/caching.
* Use Ollama locally for agent reasoning and reports when available.
* The system must still work without Ollama by using deterministic scoring and rule-based decision templates.

### Signal agents

Implement these agents as structured Python modules that output JSON-compatible evidence packets:

1. **Market Analyst Agent**

   * Benchmark/index returns, breadth proxy, volatility, drawdown, risk-on/risk-off score, and market regime.

2. **Fundamentals Agent**

   * Use free `yfinance` company fundamentals when available: valuation ratios, revenue/earnings growth, margins, debt, cash flow, and return on equity.
   * If fundamentals are missing, explicitly lower confidence rather than fabricating values.

3. **Technical Analyst Agent**

   * SMA/EMA, RSI, MACD, Bollinger Bands, ATR, momentum, volume trend, OBV, support/resistance, relative strength, rolling volatility, and drawdown.
   * Return a normalized score from -1 to +1 and the strongest supporting/conflicting indicators.

4. **Quant Prediction Agent**

   * Train local walk-forward-validated models using only historical data available before the prediction date.
   * Predict 5-day, 20-day, and 60-day direction, expected return, volatility/risk, and confidence.
   * Use Logistic Regression/Random Forest/Gradient Boosting as appropriate.
   * Persist models and evaluation metrics locally.

5. **Sector Analyst Agent**

   * Sector ETF/index momentum, relative strength, volatility, sector breadth proxy, and company-score aggregation.
   * Rank sectors by expected return, confidence, and risk.

6. **Geographic Impact Agent**

   * Country benchmark/index return, volatility shift, drawdown, risk score, and mapped company/sector exposure.
   * Use structured mappings and calculated market performance only.

### Researcher debate

Retain both researchers.

#### Bull Researcher

* Receives the structured evidence packets.
* Identifies the strongest bullish signals: positive forecast, improving trend, relative strength, favorable fundamentals, supportive market/sector regime, and acceptable risk.
* Produces a bullish thesis, invalidation conditions, and confidence.
* With Ollama enabled, it may write a natural-language thesis using only supplied JSON evidence.
* Without Ollama, create the thesis using deterministic templates.

#### Bear Researcher

* Receives the same evidence packets.
* Identifies downside signals: negative forecast, weakening momentum, high volatility, drawdown, unfavorable regime, poor relative strength, valuation/fundamental risks, and concentration risk.
* Produces a bearish thesis, invalidation conditions, and confidence.
* With Ollama enabled, it may write a natural-language thesis using only supplied JSON evidence.
* Without Ollama, use deterministic templates.

The debate must be evidence-grounded. Agents must never invent prices, news, events, company facts, or indicators.

### Trader Agent

Implement a Trader Agent that receives:

* Market Analyst evidence
* Fundamentals evidence
* Technical evidence
* Quant prediction output
* Sector analysis
* Geographic risk output
* Bull Researcher thesis
* Bear Researcher thesis
* Current simulated portfolio state

The Trader Agent must produce a structured trade proposal:

```json
{
  "ticker": "RELIANCE.NS",
  "action": "BUY | HOLD | SELL | AVOID",
  "time_horizon": "5D | 20D | 60D",
  "conviction": 0.0,
  "expected_return_pct": 0.0,
  "estimated_downside_pct": 0.0,
  "suggested_position_size_pct": 0.0,
  "entry_zone": {
    "low": 0.0,
    "high": 0.0
  },
  "stop_loss": 0.0,
  "take_profit": 0.0,
  "risk_reward_ratio": 0.0,
  "supporting_evidence": [],
  "conflicting_evidence": [],
  "conditions_to_invalidate": [],
  "reasoning_summary": ""
}
```

Use transparent decision logic for the recommendation:

* `BUY` only when weighted evidence is positive, confidence exceeds a configurable threshold, and risk limits permit it.
* `SELL` when downside evidence dominates or a risk exit is triggered.
* `HOLD` when signals are mixed or there is already a valid simulated position.
* `AVOID` when data quality is poor, uncertainty is high, or risk is too high.
* Calculate entry zone from current price, support/resistance, ATR, and moving averages.
* Calculate stop loss and take profit from ATR, support/resistance, predicted downside/upside, and a minimum configurable risk-reward ratio.
* Calculate position size using volatility targeting and portfolio risk limits, not arbitrary values.

The recommendation must be labelled **“Research-only simulated trade recommendation — not investment advice.”**

### Risk Management Agent

Retain the Risk Management Agent. It must review every Trader Agent proposal against:

* Maximum position size
* Maximum portfolio exposure
* Maximum sector exposure
* Maximum country exposure
* Volatility-adjusted position sizing
* ATR-based stop-loss distance
* Drawdown threshold
* Correlation/concentration proxy
* Market-regime risk multiplier
* Minimum risk-reward ratio
* Data quality and model-confidence threshold

Output:

```json
{
  "approved": true,
  "risk_level": "LOW | MEDIUM | HIGH",
  "adjusted_position_size_pct": 0.0,
  "required_stop_loss": 0.0,
  "risk_flags": [],
  "required_changes": [],
  "risk_summary": ""
}
```

The Risk Manager may reduce position size, require a tighter stop loss, convert `BUY` to `HOLD`, or reject the proposal.

### Portfolio Manager Agent

Retain the Portfolio Manager as the final decision-maker.

It must:

* Review the Trader proposal and Risk Manager output.
* Approve, reject, or modify the recommendation.
* Check portfolio-level constraints, existing holdings, cash allocation, sector/country concentration, and total risk.
* Send only approved actions to a simulated paper-trading ledger.
* Track entry price, quantity, stop loss, take profit, holding horizon, realized/unrealized P&L, and decision rationale.
* Generate a final decision record and dashboard card.

Final output:

```json
{
  "final_action": "BUY | HOLD | SELL | AVOID",
  "approved": true,
  "ticker": "RELIANCE.NS",
  "position_size_pct": 0.0,
  "entry_zone": [0.0, 0.0],
  "stop_loss": 0.0,
  "take_profit": 0.0,
  "portfolio_risk_after_trade": 0.0,
  "decision_summary": "",
  "disclaimer": "Research-only simulated trade recommendation — not investment advice."
}
```

### Ollama behavior

* Use Ollama as the local reasoning/writing engine for the Bull Researcher, Bear Researcher, Trader Agent, Risk Manager, and Portfolio Manager when installed.
* Every Ollama prompt must include only calculated evidence packets and explicitly prohibit invented information.
* Require valid JSON output and validate it with Pydantic schemas.
* If Ollama is unavailable or returns invalid output, fall back to deterministic scoring/rules so the complete workflow still produces a recommendation.
* Ollama enhances explanation and structured deliberation; it must not be the sole source of numerical predictions, price levels, risk limits, or trade execution decisions.
