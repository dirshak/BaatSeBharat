import { useEffect, useMemo, useState } from 'react'
import { dataClient } from '../dataClient.js'
import { predictCompanyDetail, predictCompanyBulk, predictSector } from '../predictionMath.js'
import { buildCompanyBarFig, buildCompanyScatterFig, buildSectorBarFig, buildSectorMapFig } from '../chartBuilders.js'
import StageHeader from '../components/StageHeader.jsx'
import Tabs from '../components/Tabs.jsx'
import PlotlyChart from '../components/PlotlyChart.jsx'
import DataTable from '../components/DataTable.jsx'
import Alert from '../components/Alert.jsx'

const SIGNAL_COLORS = { Bullish: '#2F6F4E', Bearish: '#A6503A', Neutral: '#9AA3B5' }
const REGIME_OPTIONS = ['Bull', 'Neutral', 'Bear', 'Stable', 'Transitional', 'Volatile']

function CompanyTab({ sentiment, topic, regime, companies, constants }) {
  const [selectedCompany, setSelectedCompany] = useState(companies[0]?.company ?? null)
  const [showInputs, setShowInputs] = useState(false)
  const [showTable, setShowTable] = useState(false)

  const baseline = companies.find((c) => c.company === selectedCompany)
  const pred = baseline
    ? predictCompanyDetail(baseline, { sentiment, topic }, constants.regimeMultiplier)
    : null

  const bulkPredictions = useMemo(
    () => companies.map((b) => predictCompanyBulk(b, { sentiment, topic, regime }, constants.regimeMultiplier)),
    [companies, sentiment, topic, regime, constants]
  )
  const missingMarketData = companies.filter((c) => c.currentPrice === null).map((c) => c.company)
  const barFig = useMemo(() => buildCompanyBarFig(bulkPredictions), [bulkPredictions])
  const scatterFig = useMemo(() => buildCompanyScatterFig(bulkPredictions), [bulkPredictions])

  return (
    <div>
      <h3>🏢 Company-Level Predictions</h3>
      <p className="caption">Signals: FinBERT sentiment · topic strength · regime · yfinance price momentum</p>

      <div className="field">
        <label>Select Company for Detail View</label>
        <select value={selectedCompany ?? ''} onChange={(e) => setSelectedCompany(e.target.value)}>
          {companies.map((c) => (
            <option key={c.company}>{c.company}</option>
          ))}
        </select>
      </div>

      {pred ? (
        <>
          <div className="signal-banner" style={{ border: `1px solid ${SIGNAL_COLORS[pred.signal]}`, borderLeft: `3px solid ${SIGNAL_COLORS[pred.signal]}` }}>
            <span className="signal-dot" style={{ background: SIGNAL_COLORS[pred.signal] }}></span>
            <span className="signal-title" style={{ color: SIGNAL_COLORS[pred.signal] }}>{pred.signal.toUpperCase()}</span>
            <span className="signal-meta">
              {selectedCompany} &middot; <b>{pred.ticker || 'N/A'}</b> &middot; Confidence:{' '}
              <b style={{ color: SIGNAL_COLORS[pred.signal] }}>{pred.confidence.toFixed(0)}%</b> &middot; Mode:{' '}
              {pred.mode?.toUpperCase()}
            </span>
          </div>

          {pred.current_price ? (
            <div className="stat-card" style={{ display: 'inline-block', marginBottom: '1rem', textAlign: 'left' }}>
              <div className="stat-label">Current Market Price ({pred.ticker})</div>
              <div className="stat-value">₹{Number(pred.current_price).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</div>
            </div>
          ) : null}

          <h3 style={{ marginTop: '1rem' }}>📅 Forecast Horizons</h3>
          <div className="grid-3">
            {[1, 5, 10].map((h) => {
              const fc = pred.predictions?.[h] || {}
              const ret = fc.return_pct || 0
              const rc = ret > 0 ? '#2F6F4E' : ret < 0 ? '#A6503A' : '#9AA3B5'
              return (
                <div className="forecast-card" key={h}>
                  <div className="fc-label">{fc.label || `${h}D`}</div>
                  <div className="fc-value" style={{ color: rc }}>
                    {ret >= 0 ? '+' : ''}
                    {ret.toFixed(2)}%
                  </div>
                  <div className="fc-range">
                    Range: {(fc.return_low || 0).toFixed(1)}% → {(fc.return_high || 0).toFixed(1)}%
                  </div>
                  {fc.price_mid ? (
                    <div style={{ fontSize: '0.78rem', color: '#475569', marginTop: 4 }}>
                      ₹{Math.round(fc.price_low).toLocaleString()} – ₹{Math.round(fc.price_high).toLocaleString()}
                    </div>
                  ) : null}
                </div>
              )
            })}
          </div>

          <div style={{ marginTop: '1rem' }}>
            <button onClick={() => setShowInputs((v) => !v)}>🔍 Signal Inputs Used</button>
            {showInputs ? (
              <DataTable
                rows={[
                  { Signal: 'FinBERT Sentiment', Value: `${(pred.inputs?.sentiment ?? 0).toFixed(4)}` },
                  { Signal: 'Rhetoric Signal', Value: `${(pred.inputs?.rhetoric_signal ?? 0).toFixed(4)}` },
                  { Signal: 'Market Regime', Value: pred.inputs?.regime || 'N/A' },
                  { Signal: '5-Day Price Momentum', Value: `${(pred.inputs?.momentum_5d_pct ?? 0).toFixed(2)}%` },
                  { Signal: 'Hist. Avg Return (5D)', Value: `${(pred.inputs?.historical_return_pct ?? 0).toFixed(4)}%` },
                  { Signal: 'Groq Topic Strength', Value: pred.inputs?.llm_strength != null ? pred.inputs.llm_strength.toFixed(4) : 'N/A — not yet classified' },
                  { Signal: 'Groq Sentiment', Value: pred.inputs?.llm_sentiment != null ? pred.inputs.llm_sentiment.toFixed(4) : 'N/A — not yet classified' },
                ]}
                columns={['Signal', 'Value']}
              />
            ) : null}
          </div>
        </>
      ) : null}

      <hr />
      <h3>📋 All Companies — 5-Day Forecast</h3>
      <p className="caption">Recomputed instantly in your browser as you move the sliders above — no server round-trip.</p>

      {missingMarketData.length > 0 ? (
        <p className="caption">
          ⚠️ No market_data downloaded yet for: {missingMarketData.join(', ')}. Their predictions fall back
          to profile-only baselines (no live momentum or historical-return signal) until yfinance data is
          fetched for them.
        </p>
      ) : null}

      <PlotlyChart fig={barFig} height={370} />
      <PlotlyChart fig={scatterFig} height={330} />
      <div style={{ marginTop: '0.5rem' }}>
        <button onClick={() => setShowTable((v) => !v)}>📋 Full Table</button>
        {showTable ? (
          <DataTable
            rows={[...bulkPredictions]
              .sort((a, b) => b.score - a.score)
              .map((p) => ({
                Company: p.company,
                Signal: { Bullish: '🟢 Bullish', Neutral: '⚪ Neutral', Bearish: '🔴 Bearish' }[p.signal] || p.signal,
                Confidence: `${p.confidence.toFixed(0)}%`,
                '1D %': `${p.predictions[1].return_pct >= 0 ? '+' : ''}${p.predictions[1].return_pct.toFixed(2)}%`,
                '5D %': `${p.predictions[5].return_pct >= 0 ? '+' : ''}${p.predictions[5].return_pct.toFixed(2)}%`,
                '10D %': `${p.predictions[10].return_pct >= 0 ? '+' : ''}${p.predictions[10].return_pct.toFixed(2)}%`,
              }))}
            columns={['Company', 'Signal', 'Confidence', '1D %', '5D %', '10D %']}
          />
        ) : null}
      </div>
    </div>
  )
}

function SectorTab({ sentiment, topic, sectors, constants, companyLocations }) {
  const sectorPredictions = useMemo(
    () => sectors.map((b) => predictSector(b, { sentiment, topic }, constants.regimeMultiplier)),
    [sectors, sentiment, topic, constants]
  )
  const sorted = [...sectorPredictions].sort((a, b) => b.score - a.score)
  const barFig = useMemo(() => buildSectorBarFig(sectorPredictions), [sectorPredictions])
  const { fig: mapFig, skippedSectors } = useMemo(
    () => buildSectorMapFig(sectorPredictions, sectors, companyLocations),
    [sectorPredictions, sectors, companyLocations]
  )

  return (
    <div>
      <h3>📦 Sector-Level Predictions</h3>
      <p className="caption">Aggregated from constituent company momentum + BaatSeBharat regime data · recomputed instantly client-side</p>

      {sorted.map((row, i) =>
        i % 3 === 0 ? (
          <div className="grid-3" key={i}>
            {sorted.slice(i, i + 3).map((r) => {
              const sc = SIGNAL_COLORS[r.signal]
              const ret5 = r.predictions[5].return_pct
              const rc = ret5 > 0 ? '#2F6F4E' : ret5 < 0 ? '#A6503A' : '#9AA3B5'
              return (
                <div className="sector-card" key={r.sector} style={{ border: `1px solid ${sc}`, borderLeft: `3px solid ${sc}` }}>
                  <div className="sc-title" style={{ color: sc }}>
                    <span className="signal-dot" style={{ background: sc, width: 8, height: 8 }}></span>
                    {r.sector}
                  </div>
                  <div className="sc-meta">
                    Confidence: <span style={{ fontFamily: 'IBM Plex Mono, monospace' }}>{r.confidence.toFixed(0)}%</span>
                  </div>
                  <table>
                    <tbody>
                      <tr><td>1D</td><td style={{ color: rc }}>{r.predictions[1].return_pct >= 0 ? '+' : ''}{r.predictions[1].return_pct.toFixed(2)}%</td></tr>
                      <tr><td>1W</td><td style={{ color: rc }}>{ret5 >= 0 ? '+' : ''}{ret5.toFixed(2)}%</td></tr>
                      <tr><td>10D</td><td style={{ color: rc }}>{r.predictions[10].return_pct >= 0 ? '+' : ''}{r.predictions[10].return_pct.toFixed(2)}%</td></tr>
                    </tbody>
                  </table>
                </div>
              )
            })}
          </div>
        ) : null
      )}

      <PlotlyChart fig={barFig} height={370} />

      <h3 style={{ marginTop: '1.5rem' }}>🗺️ Sector Prediction Map</h3>
      <p className="caption">
        Constituent-company HQ locations colored by that sector's predicted effect — green = strongest bullish,
        red = most bearish.
      </p>
      {mapFig ? (
        <>
          <PlotlyChart fig={mapFig} height={460} />
          {skippedSectors?.length ? (
            <p className="caption">
              Not plotted: {skippedSectors.join(', ')} — "Broad Market" spans every company (would overlap
              every other sector's markers) and sectors with no mapped constituent companies have no real
              location to plot.
            </p>
          ) : null}
        </>
      ) : (
        <Alert type="info">No location data available for the current sector predictions.</Alert>
      )}
    </div>
  )
}

export default function Predictions() {
  const [loaded, setLoaded] = useState(null)
  const [sentiment, setSentiment] = useState(0)
  const [topic, setTopic] = useState(0.5)
  const [regime, setRegime] = useState('Neutral')

  useEffect(() => {
    Promise.all([
      dataClient.predictionCompanies(),
      dataClient.predictionSectors(),
      dataClient.predictionConstants(),
      dataClient.companyLocations(),
    ]).then(([companiesResp, sectorsResp, constants, companyLocations]) => {
      setLoaded({ companies: companiesResp.companies, sectors: sectorsResp.sectors, constants, companyLocations })
      setSentiment(constants.liveDefaults.sentiment)
      setTopic(constants.liveDefaults.topicStrength)
      setRegime(REGIME_OPTIONS.includes(constants.liveDefaults.regime) ? constants.liveDefaults.regime : 'Neutral')
    })
  }, [])

  if (!loaded) {
    return (
      <div>
        <StageHeader number="06" title="AI Market Predictions" subtitle="Company & sector forecasts driven by BaatSeBharat NLP signals." />
        <p className="spinner-note">Loading…</p>
      </div>
    )
  }

  return (
    <div>
      <StageHeader number="06" title="AI Market Predictions" subtitle="Company & sector forecasts driven by BaatSeBharat NLP signals." />

      <hr />
      <h3>⚙️ Signal Overrides</h3>
      <p className="caption">
        Defaults reflect the most recent scheduled pipeline run. Adjust to run what-if scenarios — recomputed
        instantly in your browser, no server round-trip.
      </p>

      <div className="field-row">
        <div className="field">
          <label>FinBERT Sentiment: {sentiment.toFixed(2)}</label>
          <input type="range" min={-1} max={1} step={0.01} value={sentiment} onChange={(e) => setSentiment(Number(e.target.value))} />
        </div>
        <div className="field">
          <label>Rhetoric Signal: {topic.toFixed(2)}</label>
          <input type="range" min={0} max={1} step={0.01} value={topic} onChange={(e) => setTopic(Number(e.target.value))} />
        </div>
        <div className="field">
          <label>Market Regime</label>
          <select value={regime} onChange={(e) => setRegime(e.target.value)}>
            {REGIME_OPTIONS.map((r) => (
              <option key={r}>{r}</option>
            ))}
          </select>
          <div className="caption" style={{ marginTop: '0.3rem' }}>
            Only affects the "All Companies" table below — the single-company detail card and Sector Predictions
            each use their own resolved regime, same as the live pipeline.
          </div>
        </div>
      </div>

      <hr />

      <Tabs
        tabs={[
          {
            label: '🏢 Company Predictions',
            content: (
              <CompanyTab
                sentiment={sentiment} topic={topic} regime={regime}
                companies={loaded.companies} constants={loaded.constants}
              />
            ),
          },
          {
            label: '📦 Sector Predictions',
            content: (
              <SectorTab
                sentiment={sentiment} topic={topic}
                sectors={loaded.sectors} constants={loaded.constants} companyLocations={loaded.companyLocations}
              />
            ),
          },
        ]}
      />

      <Alert type="info">
        💡 <strong>Data freshness:</strong> Sentiment, topic strength, regime, and momentum baselines refresh
        whenever the scheduled pipeline runs (see status bar above for the last update). Moving the sliders
        recomputes forecasts instantly from those baselines — no server round-trip.
      </Alert>
    </div>
  )
}
