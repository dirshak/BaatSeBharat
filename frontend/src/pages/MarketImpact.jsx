import { useEffect, useState } from 'react'
import { api } from '../api.js'
import StageHeader from '../components/StageHeader.jsx'
import PlotlyChart from '../components/PlotlyChart.jsx'
import DataTable from '../components/DataTable.jsx'
import Alert from '../components/Alert.jsx'

export default function MarketImpact() {
  const [tickers, setTickers] = useState([])
  const [ticker, setTicker] = useState(null)
  const [sourceColors, setSourceColors] = useState({})
  const [sourceFilter, setSourceFilter] = useState([])
  const [data, setData] = useState(null)

  useEffect(() => {
    api.marketImpactTickers().then((d) => {
      setTickers(d.tickers)
      if (d.tickers.length > 0) setTicker(d.tickers[0])
    })
  }, [])

  useEffect(() => {
    if (!ticker) return
    api.marketImpact(ticker, sourceFilter.length ? sourceFilter.join(',') : undefined).then((d) => {
      setData(d)
      if (d.sourceColors && sourceFilter.length === 0) {
        setSourceFilter(Object.keys(d.sourceColors))
      }
    })
  }, [ticker])

  useEffect(() => {
    if (!ticker || sourceFilter.length === 0) return
    api.marketImpact(ticker, sourceFilter.join(',')).then(setData)
  }, [sourceFilter])

  return (
    <div>
      <StageHeader number="03" title="Speech Impact on Markets" />

      {!data ? (
        <p className="spinner-note">Loading…</p>
      ) : data.empty ? (
        <Alert type="warning">No impact data yet. Click "Run Pipeline" above to populate the database.</Alert>
      ) : (
        <>
          <h3>Market Performance with Speech Events</h3>
          <div className="field">
            <label>Select Ticker</label>
            <select value={ticker} onChange={(e) => setTicker(e.target.value)}>
              {tickers.map((t) => (
                <option key={t}>{t}</option>
              ))}
            </select>
          </div>

          <div className="grid-3">
            <div className="stat-card" style={{ borderColor: data.signal.color }}>
              <div className="stat-label">Overall Market Signal</div>
              <div className="stat-value" style={{ color: data.signal.color }}>
                {data.signal.emoji} {data.signal.label}
              </div>
              <div className="stat-sub">Based on 5-Day Fwd Returns</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Signal Breakdown</div>
              <div style={{ fontSize: '0.92rem', marginTop: 4 }}>
                <span style={{ color: '#2F6F4E' }}>🟢 {data.breakdown.bullish} Bull</span>&nbsp;
                <span style={{ color: '#9AA3B5' }}>⚪ {data.breakdown.neutral} Neutral</span>&nbsp;
                <span style={{ color: '#A6503A' }}>🔴 {data.breakdown.bearish} Bear</span>
              </div>
              <div className="stat-sub">Across {data.breakdown.total} events</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Signal Confidence</div>
              <div className="stat-value" style={{ color: data.signal.color }}>
                {data.confidence.pct.toFixed(0)}%
              </div>
              <div className="stat-sub">Avg Abnormal: {data.confidence.avgAbnormalPct >= 0 ? '+' : ''}{data.confidence.avgAbnormalPct.toFixed(3)}%</div>
            </div>
          </div>

          <PlotlyChart fig={data.priceFig} height={450} />

          <h3>Speech Event Forward Returns</h3>
          <div className="field">
            <label>Filter by Source</label>
            <div>
              {Object.keys(data.sourceColors).map((src) => (
                <label key={src} style={{ marginRight: '1rem', display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
                  <input
                    type="checkbox"
                    checked={sourceFilter.includes(src)}
                    onChange={(e) => {
                      setSourceFilter((prev) =>
                        e.target.checked ? [...prev, src] : prev.filter((s) => s !== src)
                      )
                    }}
                  />
                  {src}
                </label>
              ))}
            </div>
          </div>
          <DataTable rows={data.eventTable} />

          <h3 style={{ marginTop: '1.5rem' }}>Average 5-Day Abnormal Return by Source</h3>
          <PlotlyChart fig={data.avgBySourceFig} height={380} />
          <Alert type="info">
            💡 <strong>Interpretation:</strong> A positive abnormal return means speeches from this source tend to
            coincide with above-average 5-day forward returns.
          </Alert>

          <hr />
          <h3>🎯 Topic-Market Correlation Analysis</h3>
          <p>
            This section aligns leadership rhetoric (topics) with market performance to identify which themes drive
            the highest returns.
          </p>
          {data.topicCorrelationFig ? (
            <>
              <PlotlyChart fig={data.topicCorrelationFig} height={440} />
              {data.alphaDriver ? (
                <Alert type="success">
                  <strong>Alpha Driver:</strong> <strong>{data.alphaDriver.topicLabel}</strong> is the most impactful
                  theme for {ticker} — Signal: <strong>{data.alphaDriver.signal}</strong> · Avg 5-Day Abnormal Return:{' '}
                  <strong>{(data.alphaDriver.avgAbnormal * 100).toFixed(2)}%</strong>
                </Alert>
              ) : null}
            </>
          ) : (
            <Alert type="warning">No topic-alignment data available. Run the pipeline first.</Alert>
          )}
        </>
      )}
    </div>
  )
}
