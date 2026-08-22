import { useEffect, useState } from 'react'
import { dataClient } from '../dataClient.js'
import LedgerRow from '../components/LedgerRow.jsx'
import PlotlyChart from '../components/PlotlyChart.jsx'
import DataTable from '../components/DataTable.jsx'
import Alert from '../components/Alert.jsx'

const SOURCE_DOT_COLORS = { 'Mann Ki Baat': '#C97A2B', ECB: '#5A7A9A', Fed: '#2F6F4E' }

export default function Overview() {
  const [data, setData] = useState(null)

  useEffect(() => {
    dataClient.overview().then(setData).catch(() => setData(null))
  }, [])

  if (!data) return <p className="spinner-note">Loading…</p>

  const hitSub =
    data.hitRateDeltaPp !== null && data.hitRateDeltaPp !== undefined
      ? `<span style="color:${data.hitRateDeltaPp > 0 ? '#2F6F4E' : data.hitRateDeltaPp < 0 ? '#A6503A' : '#9AA3B5'}">${data.hitRateDeltaPp >= 0 ? '+' : ''}${data.hitRateDeltaPp.toFixed(1)}pp vs. random</span>`
      : null
  const hitTip =
    data.hitRatePct !== null && data.hitRatePct !== undefined
      ? `Out-of-sample backtest: topic-to-return bias learned on speeches before ${data.backtestCutoffDate}, tested on ${data.backtestNEvents} events after. 50% = no better than a coin flip.`
      : null

  return (
    <div>
      <div className="stage-eyebrow">BaatSeBharat · Research Console</div>
      <h1>Leadership Rhetoric Driven Market Intelligence</h1>
      <p>Quantifying the impact of leadership narrative on market volatility.</p>

      <LedgerRow
        items={[
          { label: 'Processed Speeches', value: data.speechCount.toLocaleString() },
          { label: 'Market Data Points', value: data.marketCount.toLocaleString() },
          { label: 'Active Topics', value: data.activeTopics || '—' },
          {
            label: 'Directional Hit Rate (OOS)',
            value: data.hitRatePct !== null && data.hitRatePct !== undefined ? `${data.hitRatePct.toFixed(1)}%` : '—',
            sub: hitSub,
            tooltip: hitTip,
          },
        ]}
      />

      <hr />

      {data.sourceBreakdown.length > 0 ? (
        <div className="grid-2" style={{ alignItems: 'center' }}>
          <div>
            <h3>Speech Sources</h3>
            {data.sourceBreakdown.map((s) => (
              <p key={s.source}>
                <span style={{ color: SOURCE_DOT_COLORS[s.source] || '#9AA3B5' }}>●</span>{' '}
                <strong>{s.source}</strong>: {s.count} speeches
              </p>
            ))}
          </div>
          <div>
            <PlotlyChart fig={data.sourcePieFig} height={340} />
          </div>
        </div>
      ) : null}

      <hr />
      <h3>Live Pipeline Feed</h3>
      {data.speechCount > 0 ? (
        <DataTable rows={data.recentSpeeches} columns={['date', 'source', 'speaker', 'title']} />
      ) : (
        <Alert type="warning">No data found. Waiting on the next scheduled pipeline run.</Alert>
      )}
    </div>
  )
}
