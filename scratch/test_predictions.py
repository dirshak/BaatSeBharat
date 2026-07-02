import sys
sys.path.insert(0, 'src')
from prediction_engine import get_all_company_predictions, COMPANY_PROFILES, SECTOR_PROFILES, get_all_sector_predictions

print("=== Testing AI Predictions Diversity ===")
preds = get_all_company_predictions(0.2, 0.6, 'Bull', 0.01)
signals_seen = set()
for p in preds:
    signals_seen.add(p["signal"])
    print(f"  {p['company']:25s} | {p['signal']:8s} | score={p['score']:+.3f} | conf={p['confidence']:.0f}%")

print(f"\nUnique signals: {signals_seen}")
assert len(preds) > 0, "No predictions returned!"
# Check that scores are NOT all identical
scores = [p["score"] for p in preds]
assert len(set(scores)) > 1, f"All scores are the same: {scores[0]} — BUG!"
print(f"\nScore diversity: {len(set(scores))} unique scores across {len(scores)} companies. ✅")

print("\n=== Testing Sector Predictions ===")
sec_preds = get_all_sector_predictions(0.2, 0.6)
for sp in sec_preds:
    print(f"  {sp['sector']:15s} | {sp['signal']:8s} | score={sp['score']:+.3f}")

print("\nALL TESTS PASSED ✅")
