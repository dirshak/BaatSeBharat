"""Quick offline test - directly tests the scoring math."""
import sys, numpy as np
sys.path.insert(0, 'src')
from prediction_engine import _composite_score, _score_to_signal, COMPANY_PROFILES, SECTOR_PROFILES

print("=== Score Diversity Test (m5=m10=0, simulating yfinance failure) ===")
sentiment = 0.2
topic = 0.6
regime = 'Bull'
hist_ret = 0.01

scores = {}
for company, (beta, rhet_sens, base_drift) in COMPANY_PROFILES.items():
    score = _composite_score(
        sentiment_score=sentiment,
        topic_strength=topic,
        regime_label=regime,
        momentum_5d=0.0,
        momentum_10d=0.0,
        historical_return=hist_ret,
        beta=beta,
        rhetoric_sensitivity=rhet_sens,
        base_drift=base_drift,
    )
    signal, emoji = _score_to_signal(score)
    scores[company] = score
    print(f"  {company:25s} | {emoji} {signal:8s} | score={score:+.4f} | drift={base_drift:+.2f}")

unique_scores = len(set(round(s, 4) for s in scores.values()))
unique_signals = len(set(_score_to_signal(s)[0] for s in scores.values()))
print(f"\nResult: {unique_scores} unique scores, {unique_signals} unique signal types")

if unique_scores > 1:
    print("PASS: Predictions are diversified even without live data!")
else:
    print("FAIL: All scores are identical")
