const BASE = '/api'

async function request(path, params) {
  let url = BASE + path
  if (params) {
    const qs = new URLSearchParams(
      Object.fromEntries(Object.entries(params).filter(([, v]) => v !== undefined && v !== null))
    ).toString()
    if (qs) url += '?' + qs
  }
  const res = await fetch(url)
  if (!res.ok) {
    throw new Error(`${path} failed: ${res.status}`)
  }
  return res.json()
}

export const api = {
  status: () => request('/status'),
  overview: () => request('/overview'),
  runPipeline: () => fetch(BASE + '/run-pipeline', { method: 'POST' }).then((r) => r.json()),

  ingestionSpeeches: (source) => request('/ingestion/speeches', { source }),
  ingestionSpeechText: (id) => request(`/ingestion/speeches/${id}`),
  ingestionMarket: () => request('/ingestion/market'),

  nlpModels: () => request('/nlp/models'),
  nlpTopics: (model) => request('/nlp/topics', { model }),

  marketImpactTickers: () => request('/market-impact/tickers'),
  marketImpact: (ticker, sources) => request('/market-impact', { ticker, sources }),

  regimeTickers: () => request('/regime/tickers'),
  regime: (ticker) => request('/regime', { ticker }),

  companyAnalyticsCompanies: () => request('/company-analytics/companies'),
  companyAnalytics: (company) => request('/company-analytics', { company }),

  predictionDefaults: () => request('/predictions/defaults'),
  predictionCompany: (params) => request('/predictions/company', params),
  predictionCompanyAll: (params) => request('/predictions/company/all', params),
  predictionSector: (params) => request('/predictions/sector', params),
  predictionSectorMap: (params) => request('/predictions/sector-map', params),

  geoCountryRisk: () => request('/geo/country-risk'),
  geoShocks: (shock_type) => request('/geo/shocks', { shock_type }),
  geoCompanyMap: () => request('/geo/company-map'),
  geoIndicators: () => request('/geo/indicators'),
  geoIndicator: (name, countries) => request('/geo/indicator', { name, countries }),

  previewCompanies: () => request('/preview/companies'),
  preview: (source, company) => request('/preview', { source, company }),
}
