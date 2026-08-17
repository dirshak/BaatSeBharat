import { useEffect, useState } from 'react'
import { api } from '../api.js'
import StageHeader from '../components/StageHeader.jsx'
import Tabs from '../components/Tabs.jsx'
import PlotlyChart from '../components/PlotlyChart.jsx'
import DataTable from '../components/DataTable.jsx'
import Alert from '../components/Alert.jsx'

const SIGNAL_COLORS = { Bullish: '#2F6F4E', Bearish: '#A6503A', Neutral: '#9AA3B5' }
const REGIME_OPTIONS = ['Bull', 'Neutral', 'Bear', 'Stable', 'Transitional', 'Volatile']

function CompanyTab({ sentiment, topic, regime, useLlm }) {
  const [companies, setCompanies] = useState([])
  const [selectedCompany, setSelectedCompany] = useState(null)
  const [pred, setPred] = useState(null)
  const [all, setAll] = useState(null)
  const [showInputs, setShowInputs] = useState(false)
  const [showLlm, setShowLlm] = useState(false)
  const [showTable, setShowTable] = useState(false)

  useEffect(() => {
    api.predictionDefaults().then((d) => {
      if (d.predOk) {
        setCompanies(d.companies)
        if (d.companies.length > 0) setSelectedCompany(d.companies[0])
      }
    })
  }, [])

  useEffect(() => {
    if (!selectedCompany) return
    setPred(null)
    api
      .predictionCompany({ company: selectedCompany, sentiment, topic, regime, use_llm: useLlm })
      .then((d) => setPred(d.predOk ? d.prediction : null))
  }, [selectedCompany, sentiment, topic, regime, useLlm])

  useEffect(() => {
    setAll(null)
    api.predictionCompanyAll({ sentiment, topic, regime }).then((d) => setAll(d.predOk ? d : null))
  }, [sentiment, topic, regime])

  return (
    <div>
      <h3>🏢 Company-Level Predictions</h3>
      <p className="caption">Signals: FinBERT sentiment · topic strength · regime · yfinance price momentum</p>

      <div className="field">
        <label>Select Company for Detail View</label>
        <select value={selectedCompany ?? ''} onChange={(e) => setSelectedCompany(e.target.value)}>
          {companies.map((c) => (
            <option key={c}>{c}</option>
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
                  { Signal: 'Rhetoric Signal', Value: `${(pred.inputs?.rhetoric_signal ?? pred.inputs?.topic_strength ?? 0).toFixed(4)}` },
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

          {useLlm && pred.llm_decision ? (
            <div style={{ marginTop: '0.5rem' }}>
              <button onClick={() => setShowLlm((v) => !v)}>🤖 LLM Reasoning</button>
              {showLlm ? <pre style={{ whiteSpace: 'pre-wrap', color: 'var(--ink-dim)' }}>{pred.llm_decision}</pre> : null}
            </div>
          ) : null}
        </>
      ) : (
        <p className="spinner-note">Computing prediction…</p>
      )}

      <hr />
      <h3>📋 All Companies — 5-Day Forecast</h3>
      <p className="caption">Cached for 30 min. Adjust any slider above to invalidate cache.</p>

      {!all ? (
        <p className="spinner-note">Loading cached bulk predictions…</p>
      ) : (
        <>
          {all.missingMarketData.length > 0 ? (
            <p className="caption">
              ⚠️ No market_data downloaded yet for: {all.missingMarketData.join(', ')}. Their predictions fall back
              to profile-only baselines (no live momentum or historical-return signal) until yfinance data is
              fetched for them.
            </p>
          ) : null}

          {all.predictions.length > 0 ? (
            <>
              <PlotlyChart fig={all.barFig} height={370} />
              <PlotlyChart fig={all.scatterFig} height={330} />
              <div style={{ marginTop: '0.5rem' }}>
                <button onClick={() => setShowTable((v) => !v)}>📋 Full Table</button>
                {showTable ? (
                  <DataTable
                    rows={all.predictions.map((p) => ({
                      Company: p.Company,
                      Signal: { Bullish: '🟢 Bullish', Neutral: '⚪ Neutral', Bearish: '🔴 Bearish' }[p.Signal] || p.Signal,
                      Confidence: `${p.Confidence.toFixed(0)}%`,
                      '1D %': `${p['1D %'] >= 0 ? '+' : ''}${p['1D %'].toFixed(2)}%`,
                      '5D %': `${p['5D %'] >= 0 ? '+' : ''}${p['5D %'].toFixed(2)}%`,
                      '10D %': `${p['10D %'] >= 0 ? '+' : ''}${p['10D %'].toFixed(2)}%`,
                    }))}
                    columns={['Company', 'Signal', 'Confidence', '1D %', '5D %', '10D %']}
                  />
                ) : null}
              </div>
            </>
          ) : null}
        </>
      )}
    </div>
  )
}

function SectorTab({ sentiment, topic, regime }) {
  const [sec, setSec] = useState(null)
  const [map, setMap] = useState(undefined)

  useEffect(() => {
    setSec(null)
    api.predictionSector({ sentiment, topic, regime }).then((d) => setSec(d.predOk ? d : null))
  }, [sentiment, topic, regime])

  useEffect(() => {
    setMap(undefined)
    api.predictionSectorMap({ sentiment, topic, regime }).then((d) => setMap(d.predOk ? d : null))
  }, [sentiment, topic, regime])

  return (
    <div>
      <h3>📦 Sector-Level Predictions</h3>
      <p className="caption">Aggregated from constituent company momentum + BaatSeBharat regime data · cached 30 min</p>

      {!sec ? (
        <p className="spinner-note">Loading cached sector predictions…</p>
      ) : (
        <>
          {sec.sectors.map((row, i) => i % 3 === 0 ? (
            <div className="grid-3" key={i}>
              {sec.sectors.slice(i, i + 3).map((r) => {
                const sc = SIGNAL_COLORS[r.Signal]
                const rc = r['5D %'] > 0 ? '#2F6F4E' : r['5D %'] < 0 ? '#A6503A' : '#9AA3B5'
                return (
                  <div className="sector-card" key={r.Sector} style={{ border: `1px solid ${sc}`, borderLeft: `3px solid ${sc}` }}>
                    <div className="sc-title" style={{ color: sc }}>
                      <span className="signal-dot" style={{ background: sc, width: 8, height: 8 }}></span>
                      {r.Sector}
                    </div>
                    <div className="sc-meta">
                      Confidence: <span style={{ fontFamily: 'IBM Plex Mono, monospace' }}>{r.Conf.toFixed(0)}%</span> &nbsp;·&nbsp; Regime: {regime}
                    </div>
                    <table>
                      <tbody>
                        <tr><td>1D</td><td style={{ color: rc }}>{r['1D %'] >= 0 ? '+' : ''}{r['1D %'].toFixed(2)}%</td></tr>
                        <tr><td>1W</td><td style={{ color: rc }}>{r['5D %'] >= 0 ? '+' : ''}{r['5D %'].toFixed(2)}%</td></tr>
                        <tr><td>10D</td><td style={{ color: rc }}>{r['10D %'] >= 0 ? '+' : ''}{r['10D %'].toFixed(2)}%</td></tr>
                      </tbody>
                    </table>
                  </div>
                )
              })}
            </div>
          ) : null)}

          <PlotlyChart fig={sec.fig} height={370} />

          <h3 style={{ marginTop: '1.5rem' }}>🗺️ Sector Prediction Map</h3>
          <p className="caption">
            Constituent-company HQ locations colored by that sector's predicted effect — green = strongest bullish,
            red = most bearish.
          </p>
          {map === undefined ? (
            <p className="spinner-note">Loading map…</p>
          ) : map?.fig ? (
            <>
              <PlotlyChart fig={map.fig} height={460} />
              {map.skippedSectors?.length ? (
                <p className="caption">
                  Not plotted: {map.skippedSectors.join(', ')} — "Broad Market" spans every company (would overlap
                  every other sector's markers) and sectors with no mapped constituent companies have no real
                  location to plot.
                </p>
              ) : null}
            </>
          ) : (
            <Alert type="info">No location data available for the current sector predictions.</Alert>
          )}
        </>
      )}
    </div>
  )
}

export default function Predictions() {
  const [defaults, setDefaults] = useState(null)
  const [sentiment, setSentiment] = useState(0)
  const [topic, setTopic] = useState(0.5)
  const [regime, setRegime] = useState('Neutral')
  const [useLlm, setUseLlm] = useState(false)

  useEffect(() => {
    api.predictionDefaults().then((d) => {
      setDefaults(d)
      if (d.predOk) {
        setSentiment(d.sentiment)
        setTopic(d.topicStrength)
        setRegime(REGIME_OPTIONS.includes(d.regime) ? d.regime : 'Neutral')
        setUseLlm(d.llmModeAvailable)
      }
    })
  }, [])

  if (defaults && !defaults.predOk) {
    return (
      <div>
        <StageHeader number="06" title="AI Market Predictions" subtitle="Company & sector forecasts driven by BaatSeBharat NLP signals." />
        <Alert type="error">Prediction engine could not load: <code>{defaults.error}</code></Alert>
        <Alert type="info">Ensure `src/prediction_engine.py` is present and `yfinance` is installed.</Alert>
      </div>
    )
  }

  return (
    <div>
      <StageHeader number="06" title="AI Market Predictions" subtitle="Company & sector forecasts driven by BaatSeBharat NLP signals." />

      <hr />
      <h3>⚙️ Signal Overrides</h3>
      <p className="caption">Defaults are read live from the DB and processed files. Adjust to run what-if scenarios.</p>

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
        </div>
        <div className="field">
          <label>
            <input
              type="checkbox"
              checked={useLlm}
              disabled={!defaults?.llmModeAvailable}
              onChange={(e) => setUseLlm(e.target.checked)}
            />{' '}
            LLM Mode
          </label>
        </div>
      </div>

      <hr />

      <Tabs
        tabs={[
          { label: '🏢 Company Predictions', content: <CompanyTab sentiment={sentiment} topic={topic} regime={regime} useLlm={useLlm} /> },
          { label: '📦 Sector Predictions', content: <SectorTab sentiment={sentiment} topic={topic} regime={regime} /> },
        ]}
      />

      <Alert type="info">
        💡 <strong>Caching:</strong> Predictions refresh every 30 minutes, or immediately when you adjust the signal
        sliders. Price momentum is fetched live from yfinance.
      </Alert>
    </div>
  )
}
