/**
 * Client-side port of src/prediction_engine.py's composite-score formula.
 *
 * This is a direct port of _composite_score() / _score_to_signal() /
 * _confidence_from_score() / _forecast_return() (prediction_engine.py
 * lines 716-796) plus the input-blending each of the THREE real call
 * paths performs before reaching them. Those three paths are NOT
 * identical -- traced exactly against backend/routers/predictions.py:
 *
 *  1. Company DETAIL view (one company, the dropdown card):
 *     sentiment & topic ARE slider-driven. regime and historical-return
 *     are NOT -- company_prediction() always overrides regime with that
 *     company's own resolved CSV regime before calling
 *     get_company_prediction(), and historical-return comes from a
 *     ticker-matched DB row (or the live cross-sectional average as
 *     fallback), never from a slider. Both are precomputed server-side
 *     at export time (companies.json's `actualRegimeDetail` /
 *     `actualHistoricalReturnDetail`) and used as-is here.
 *
 *  2. Company BULK table (all 15 companies at once):
 *     sentiment & topic are slider-driven, same as detail. Regime IS
 *     slider-driven here (get_all_company_predictions() passes the raw
 *     slider value to every company uniformly) -- the ONLY place in the
 *     whole app where the Regime control has any effect. Historical
 *     return uses one global constant (live_hist_ret) for every company,
 *     never a slider -- precomputed as `actualHistoricalReturnBulk`.
 *
 *  3. Sector predictions (cards, multi-horizon bar, and the sector map):
 *     ONLY sentiment & topic are slider-driven. regime is not even
 *     accepted as a parameter by get_all_sector_predictions() server-side
 *     (sector_predictions() silently drops it) and historical-return's
 *     sector-name lookup never matches its ticker-keyed source table, so
 *     both always fall through to each sector's own baseline. Used as-is
 *     from sectors.json's `regimeBase` / `historicalReturn5dBase` /
 *     `historicalReturn10dBase`.
 *
 * Nothing here is new modeling logic -- every constant/formula matches
 * prediction_engine.py exactly. See scripts/verify_prediction_math.mjs
 * for a numeric diff against the real Python functions across many
 * companies/sectors and slider combinations.
 */

const clip = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const round2 = (v) => Math.round(v * 100) / 100;
const round3 = (v) => Math.round(v * 1000) / 1000;

export const HORIZONS = { 1: "1-Day", 5: "1-Week", 10: "10-Day" };

/** Direct port of _composite_score() (prediction_engine.py:716-745). */
export function compositeScore({
  sentiment, topic, regime, momentum5d, momentum10d, historicalReturn,
  beta, rhetoricSensitivity, baseDrift, regimeMultiplier,
}) {
  const regimeMul = regimeMultiplier[regime] ?? 0.0;

  const score =
    0.28 * clip(sentiment, -1, 1) * rhetoricSensitivity +
    0.18 * regimeMul * beta +
    0.16 * clip(momentum5d * 10, -1, 1) * beta +
    0.14 * clip(momentum10d * 10, -1, 1) * beta +
    0.10 * clip(historicalReturn * 20, -1, 1) +
    0.08 * (topic - 0.5) * 2 * rhetoricSensitivity +
    0.06 * baseDrift;

  return clip(score, -1, 1);
}

/** Direct port of _score_to_signal() (prediction_engine.py:748-755). */
export function scoreToSignal(score) {
  if (score > 0.25) return { signal: "Bullish", emoji: "🟢" };
  if (score < -0.25) return { signal: "Bearish", emoji: "🔴" };
  return { signal: "Neutral", emoji: "⚪" };
}

/** Direct port of _confidence_from_score() (prediction_engine.py:758-760). */
export function confidenceFromScore(score) {
  return Math.min(100.0, Math.abs(score) * 120 + 30);
}

/** Direct port of _forecast_return() (prediction_engine.py:763-796). */
export function forecastReturn(score, horizon, historicalReturn, currentPrice) {
  const annualReturnPct = score * 25.0;
  const horizonReturnPct = annualReturnPct * (horizon / 252.0);
  const ciHalf = Math.abs(score) * 3.0 * Math.sqrt(horizon / 10.0);

  const low = horizonReturnPct - ciHalf;
  const high = horizonReturnPct + ciHalf;

  const result = {
    return_pct: round2(horizonReturnPct),
    return_low: round2(low),
    return_high: round2(high),
  };

  if (currentPrice && currentPrice > 0) {
    result.price_mid = round2(currentPrice * (1 + horizonReturnPct / 100));
    result.price_low = round2(currentPrice * (1 + low / 100));
    result.price_high = round2(currentPrice * (1 + high / 100));
  }
  return result;
}

function buildPredictions(score, historicalReturnByHorizon, currentPrice, horizons) {
  const predictions = {};
  for (const h of horizons) {
    const hr = typeof historicalReturnByHorizon === "function"
      ? historicalReturnByHorizon(h)
      : historicalReturnByHorizon;
    predictions[h] = {
      label: HORIZONS[h] ?? `${h}-Day`,
      ...forecastReturn(score, h, hr, currentPrice),
    };
  }
  return predictions;
}

/**
 * Company DETAIL view -- mirrors company_prediction() + the rule-based
 * path of get_company_prediction() (prediction_engine.py:880-917).
 * `sliders` only ever needs {sentiment, topic}: regime and historical
 * return are pulled from the precomputed `baseline.actualRegimeDetail` /
 * `baseline.actualHistoricalReturnDetail` fields, never from a slider
 * (see file docstring).
 */
export function predictCompanyDetail(baseline, sliders, regimeMultiplier, horizons = [1, 5, 10]) {
  const actualSentiment = clip(baseline.sentimentBase + sliders.sentiment * 0.5, -1, 1);
  const actualTopic = clip(baseline.topicBase + (sliders.topic - 0.5), 0, 1);
  const actualRegime = baseline.actualRegimeDetail;
  const actualHistRet = baseline.actualHistoricalReturnDetail;

  const score = compositeScore({
    sentiment: actualSentiment, topic: actualTopic, regime: actualRegime,
    momentum5d: baseline.momentum5d, momentum10d: baseline.momentum10d,
    historicalReturn: actualHistRet,
    beta: baseline.beta, rhetoricSensitivity: baseline.rhetoricSensitivity,
    baseDrift: baseline.baseDrift, regimeMultiplier,
  });

  const { signal, emoji } = scoreToSignal(score);

  return {
    company: baseline.company,
    ticker: baseline.ticker,
    signal, emoji,
    confidence: confidenceFromScore(score),
    score: round3(score),
    mode: "rule-based",
    predictions: buildPredictions(score, actualHistRet, baseline.currentPrice, horizons),
    current_price: baseline.currentPrice,
    inputs: {
      sentiment: round3(actualSentiment),
      rhetoric_signal: round3(actualTopic),
      regime: actualRegime,
      momentum_5d_pct: round2(baseline.momentum5d * 100),
      historical_return_pct: round2(actualHistRet * 100),
      llm_strength: baseline.llmStrength,
      llm_sentiment: baseline.llmSentiment,
    },
  };
}

/**
 * Company BULK table (all companies) -- mirrors
 * get_all_company_predictions() (prediction_engine.py:1119-1134).
 * `sliders` needs {sentiment, topic, regime}: regime IS slider-driven
 * here (the one place in the app where it matters) -- blended exactly
 * like get_company_prediction()'s own internal rule.
 */
export function predictCompanyBulk(baseline, sliders, regimeMultiplier, horizons = [1, 5, 10]) {
  const actualSentiment = clip(baseline.sentimentBase + sliders.sentiment * 0.5, -1, 1);
  const actualTopic = clip(baseline.topicBase + (sliders.topic - 0.5), 0, 1);
  const actualRegime =
    sliders.regime === "Neutral" || sliders.regime === "Stable"
      ? baseline.regimeBase
      : sliders.regime;
  const actualHistRet = baseline.actualHistoricalReturnBulk;

  const score = compositeScore({
    sentiment: actualSentiment, topic: actualTopic, regime: actualRegime,
    momentum5d: baseline.momentum5d, momentum10d: baseline.momentum10d,
    historicalReturn: actualHistRet,
    beta: baseline.beta, rhetoricSensitivity: baseline.rhetoricSensitivity,
    baseDrift: baseline.baseDrift, regimeMultiplier,
  });

  const { signal, emoji } = scoreToSignal(score);

  return {
    company: baseline.company,
    signal,
    confidence: confidenceFromScore(score),
    score: round3(score),
    predictions: buildPredictions(score, actualHistRet, baseline.currentPrice, horizons),
  };
}

/**
 * Sector predictions (cards, multi-horizon bar, sector map) -- mirrors
 * get_sector_prediction() (prediction_engine.py:924-1030). `sliders` only
 * ever needs {sentiment, topic}: regime and both historical-return
 * baselines are used as-is (see file docstring) -- the Regime control has
 * no effect on this view at all, matching the live app exactly.
 */
export function predictSector(baseline, sliders, regimeMultiplier, horizons = [1, 5, 10]) {
  const actualSentiment = clip(baseline.sentimentBase + sliders.sentiment * 0.5, -1, 1);
  const actualTopic = clip(baseline.topicBase + (sliders.topic - 0.5), 0, 1);
  const actualRegime = baseline.regimeBase;

  const score = compositeScore({
    sentiment: actualSentiment, topic: actualTopic, regime: actualRegime,
    momentum5d: baseline.momentum5d, momentum10d: baseline.momentum10d,
    historicalReturn: baseline.historicalReturn5dBase,
    beta: baseline.beta, rhetoricSensitivity: baseline.rhetoricSensitivity,
    baseDrift: baseline.baseDrift, regimeMultiplier,
  });

  const { signal, emoji } = scoreToSignal(score);

  const historicalReturnByHorizon = (h) =>
    h <= 5 ? baseline.historicalReturn5dBase : baseline.historicalReturn10dBase;

  return {
    sector: baseline.sector,
    signal, emoji,
    confidence: confidenceFromScore(score),
    score: round3(score),
    mode: "rule-based",
    predictions: buildPredictions(score, historicalReturnByHorizon, null, horizons),
    constituent_prices: baseline.constituentPrices,
  };
}
