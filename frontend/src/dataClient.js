/**
 * Static data layer -- replaces api.js's fetch('/api/...') calls against
 * a live FastAPI backend with fetches against pre-exported JSON files
 * (written by scripts/export_static_data.py). No backend needed in
 * production.
 *
 * Base path is configurable via VITE_DATA_BASE_URL (set at build time) so
 * the same code works whether the JSON is bundled with the site (default,
 * same-origin /data/...) or served from an external CDN/R2 bucket.
 */

const DATA_BASE = (import.meta.env.VITE_DATA_BASE_URL || "/data").replace(/\/$/, "");

let manifestPromise = null;

async function fetchJson(path) {
  const res = await fetch(`${DATA_BASE}/${path}`);
  if (!res.ok) throw new Error(`${path} failed: ${res.status}`);
  return res.json();
}

export function getManifest() {
  if (!manifestPromise) manifestPromise = fetchJson("manifest.json");
  return manifestPromise;
}

export const dataClient = {
  status: () => fetchJson("status.json"),
  overview: () => fetchJson("stage0_overview.json"),

  // Stage 1
  ingestionIndex: () => fetchJson("stage1_ingestion/index.json"),
  ingestionSpeechText: (id) => fetchJson(`stage1_ingestion/speech/${id}.json`),
  ingestionMarket: () => fetchJson("stage1_ingestion/market.json"),

  // Stage 2
  async nlpTopics(model) {
    const manifest = await getManifest();
    const slug = manifest.stage2_nlp.slugs[model];
    return fetchJson(`stage2_nlp/${slug}.json`);
  },
  nlpModelList: async () => (await getManifest()).stage2_nlp.models,

  // Stage 3 -- full unfiltered event table; caller filters by source client-side
  marketImpactTickers: async () => (await getManifest()).stage3_market_impact.tickers,
  async marketImpact(ticker) {
    const manifest = await getManifest();
    const slug = manifest.stage3_market_impact.slugs[ticker];
    return fetchJson(`stage3_market_impact/${slug}.json`);
  },

  // Stage 4
  regimeTickers: async () => (await getManifest()).stage4_regime.tickers,
  async regime(ticker) {
    const manifest = await getManifest();
    const slug = manifest.stage4_regime.slugs[ticker];
    return fetchJson(`stage4_regime/${slug}.json`);
  },

  // Stage 5
  companyAnalyticsCompanies: async () => (await getManifest()).stage5_company_analytics.companies,
  async companyAnalytics(company) {
    const manifest = await getManifest();
    const slug = manifest.stage5_company_analytics.slugs[company];
    return fetchJson(`stage5_company_analytics/${slug}.json`);
  },

  // Stage 6 -- baselines only; predictions computed client-side (see predictionMath.js)
  predictionCompanies: () => fetchJson("stage6_predictions/companies.json"),
  predictionSectors: () => fetchJson("stage6_predictions/sectors.json"),
  predictionConstants: () => fetchJson("stage6_predictions/constants.json"),
  companyLocations: () => fetchJson("stage6_predictions/company_locations.json"),

  // Stage 7
  geoCountryRisk: () => fetchJson("stage7_geo/country_risk.json"),
  geoShocks: () => fetchJson("stage7_geo/shocks.json"),
  geoCompanyMap: () => fetchJson("stage7_geo/company_map.json"),
  geoIndicatorsList: () => fetchJson("stage7_geo/indicators.json"),
  async geoIndicator(name) {
    const manifest = await getManifest();
    const slug = manifest.stage7_geo.indicatorSlugs[name];
    return fetchJson(`stage7_geo/indicator/${slug}.json`);
  },

  // Stage 8 -- full unfiltered dataset; caller filters + recomputes summary client-side
  previewDetail: () => fetchJson("stage8_preview/detail.json"),
};
