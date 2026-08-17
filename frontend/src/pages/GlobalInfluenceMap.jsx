import { useEffect, useState } from 'react'
import { api } from '../api.js'
import Tabs from '../components/Tabs.jsx'
import PlotlyChart from '../components/PlotlyChart.jsx'
import DataTable from '../components/DataTable.jsx'
import Alert from '../components/Alert.jsx'

function CountryRiskTab() {
  const [data, setData] = useState(null)
  const [showTable, setShowTable] = useState(false)

  useEffect(() => {
    api.geoCountryRisk().then((d) => setData(d.geoOk ? d : null))
  }, [])

  if (!data) return <p className="spinner-note">Loading country risk data…</p>

  return (
    <div>
      <h3>🌐 Country Risk Layer — Bull / Neutral / Bear</h3>
      <p className="caption">
        Computed from World Bank GDP growth and inflation data. Green = bullish macro environment, Red = bearish
        macro stress.
      </p>
      <PlotlyChart fig={data.fig} height={480} />
      <div className="grid-3">
        <div className="stat-card">
          <div className="stat-label">🟢 Bull Countries</div>
          <div className="stat-value">{data.counts.bull}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">⚪ Neutral Countries</div>
          <div className="stat-value">{data.counts.neutral}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">🔴 Bear Countries</div>
          <div className="stat-value">{data.counts.bear}</div>
        </div>
      </div>
      <div style={{ marginTop: '1rem' }}>
        <button onClick={() => setShowTable((v) => !v)}>📋 Country Detail Table</button>
        {showTable ? <DataTable rows={data.table} /> : null}
      </div>
    </div>
  )
}

const SHOCK_TYPES = [
  { key: 'Inflation', label: '🔥 Inflation Shock' },
  { key: 'Policy', label: '⚙️ Policy Shock' },
  { key: 'Banking', label: '🏦 Banking Stress' },
  { key: 'Geopolitical', label: '⚡ Geopolitical Risk' },
]

function ShocksTab() {
  const [shockType, setShockType] = useState('Inflation')
  const [data, setData] = useState(null)
  const [showTable, setShowTable] = useState(false)

  useEffect(() => {
    setData(null)
    api.geoShocks(shockType).then((d) => setData(d.geoOk ? d : null))
  }, [shockType])

  return (
    <div>
      <h3>⚡ Market Shock Layer</h3>
      <p className="caption">
        Heatmaps showing severity (0 = no shock, 10 = extreme shock) of inflation, policy, banking, and geopolitical
        shocks per country.
      </p>
      <div className="field">
        <label>Select Shock Type</label>
        <select value={shockType} onChange={(e) => setShockType(e.target.value)}>
          {SHOCK_TYPES.map((s) => (
            <option key={s.key} value={s.key}>{s.label}</option>
          ))}
        </select>
      </div>

      {!data ? (
        <p className="spinner-note">Computing shock matrix…</p>
      ) : (
        <>
          <PlotlyChart fig={data.selectedFig} height={480} />

          <h3 style={{ marginTop: '1rem' }}>Shock Comparison — All Types</h3>
          <div className="grid-2">
            <div>
              <PlotlyChart fig={data.allFigs.Inflation} height={420} />
              <PlotlyChart fig={data.allFigs.Banking} height={420} />
            </div>
            <div>
              <PlotlyChart fig={data.allFigs.Policy} height={420} />
              <PlotlyChart fig={data.allFigs.Geopolitical} height={420} />
            </div>
          </div>

          <div style={{ marginTop: '1rem' }}>
            <button onClick={() => setShowTable((v) => !v)}>📋 Shock Data Table</button>
            {showTable ? <DataTable rows={data.table} /> : null}
          </div>
        </>
      )}
    </div>
  )
}

function CompanyLocationsTab() {
  const [data, setData] = useState(null)

  useEffect(() => {
    api.geoCompanyMap().then((d) => setData(d.geoOk ? d : null))
  }, [])

  return (
    <div>
      <h3>📍 Company Geo-Intelligence Map</h3>
      <p className="caption">
        Company headquarters plotted on the map with signal overlays. Marker size ∝ prediction confidence.
      </p>
      {!data ? <p className="spinner-note">Loading…</p> : <PlotlyChart fig={data.fig} height={420} />}
    </div>
  )
}

function MacroIndicatorsTab() {
  const [indicators, setIndicators] = useState([])
  const [indicator, setIndicator] = useState(null)
  const [data, setData] = useState(null)
  const [selectedCountries, setSelectedCountries] = useState([])
  const [showTrend, setShowTrend] = useState(false)
  const [showRaw, setShowRaw] = useState(false)

  useEffect(() => {
    api.geoIndicators().then((d) => {
      if (d.geoOk) {
        setIndicators(d.indicators)
        if (d.indicators.length > 0) setIndicator(d.indicators[0])
      }
    })
  }, [])

  useEffect(() => {
    if (!indicator) return
    setData(null)
    api.geoIndicator(indicator).then((d) => {
      setData(d)
      if (d.selectedCountries) setSelectedCountries(d.selectedCountries)
    })
  }, [indicator])

  useEffect(() => {
    if (!indicator || selectedCountries.length === 0) return
    api.geoIndicator(indicator, selectedCountries.join(',')).then(setData)
  }, [selectedCountries])

  return (
    <div>
      <h3>📊 World Bank Macro Indicator Explorer</h3>
      <p className="caption">Select a macro indicator to view choropleth (2018–2023).</p>

      <div className="field">
        <label>Select Indicator</label>
        <select value={indicator ?? ''} onChange={(e) => setIndicator(e.target.value)}>
          {indicators.map((i) => (
            <option key={i}>{i}</option>
          ))}
        </select>
      </div>

      {!data ? (
        <p className="spinner-note">Fetching from World Bank…</p>
      ) : data.empty ? (
        <Alert type="warning">No data available for {indicator}.</Alert>
      ) : (
        <>
          {data.simulated ? (
            <Alert type="warning">
              ⚠️ Showing simulated data — live World Bank fetch unavailable. These values are synthetic placeholders,
              not real economic data.
            </Alert>
          ) : null}

          <div className="field">
            <label>Filter Countries</label>
            <select
              multiple
              value={selectedCountries}
              onChange={(e) => setSelectedCountries(Array.from(e.target.selectedOptions, (o) => o.value))}
              style={{ height: 140 }}
            >
              {data.allCountries.map((c) => (
                <option key={c}>{c}</option>
              ))}
            </select>
          </div>

          <PlotlyChart fig={data.fig} height={500} />

          <div style={{ marginTop: '0.5rem' }}>
            <button onClick={() => setShowTrend((v) => !v)}>📈 Trend Lines</button>
            {showTrend ? (
              data.trend ? (
                <DataTable rows={data.trend} />
              ) : (
                <Alert type="info">Select ≤15 countries to view trend lines.</Alert>
              )
            ) : null}
          </div>
          <div style={{ marginTop: '0.5rem' }}>
            <button onClick={() => setShowRaw((v) => !v)}>Show Raw Data</button>
            {showRaw ? <DataTable rows={data.rawData} /> : null}
          </div>
        </>
      )}
    </div>
  )
}

export default function GlobalInfluenceMap() {
  const [geoOk, setGeoOk] = useState(true)
  const [err, setErr] = useState(null)

  useEffect(() => {
    api.geoCountryRisk().then((d) => {
      if (!d.geoOk) {
        setGeoOk(false)
        setErr(d.error)
      }
    })
  }, [])

  if (!geoOk) {
    return (
      <div>
        <div className="stage-eyebrow">Stage 07</div>
        <h1>Global Influence Map</h1>
        <Alert type="error">GeoDashboard module failed to load: <code>{err}</code></Alert>
        <Alert type="info">Ensure `src/geo_dashboard.py` exists and `wbdata` is installed.</Alert>
      </div>
    )
  }

  return (
    <div>
      <div className="stage-eyebrow">Stage 07</div>
      <h1>Global Influence Map</h1>
      <p className="stage-subtitle">Country-level market state, macro shocks, and company geo-intelligence.</p>

      <Tabs
        tabs={[
          { label: '🗺️ Country Market State', content: <CountryRiskTab /> },
          { label: '⚡ Market Shock Heatmaps', content: <ShocksTab /> },
          { label: '📍 Company Locations', content: <CompanyLocationsTab /> },
          { label: '📊 Macro Indicators', content: <MacroIndicatorsTab /> },
        ]}
      />
    </div>
  )
}
