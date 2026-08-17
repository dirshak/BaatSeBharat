import { useEffect, useMemo, useState } from 'react'
import { dataClient } from '../dataClient.js'
import { buildPredVsActualScatterFig, summarizePreview } from '../chartBuilders.js'
import StageHeader from '../components/StageHeader.jsx'
import LedgerRow from '../components/LedgerRow.jsx'
import PlotlyChart from '../components/PlotlyChart.jsx'
import DataTable from '../components/DataTable.jsx'
import Alert from '../components/Alert.jsx'

export default function GlobalPreview() {
  const [source, setSource] = useState('All')
  const [company, setCompany] = useState('All')
  const [full, setFull] = useState(undefined) // full unfiltered detail.json
  const [notOk, setNotOk] = useState(null)

  useEffect(() => {
    dataClient.previewDetail().then((d) => {
      if (!d.predHistOk) {
        setNotOk(d.error)
        return
      }
      setFull(d)
    })
  }, [])

  const companies = useMemo(() => {
    if (!full || full.empty) return []
    return [...new Set(full.detail.map((r) => r.company))].sort()
  }, [full])

  // detail.json was exported unfiltered; the Source/Company dropdowns
  // filter it (and recompute the summary stats over the filtered subset,
  // matching the original server-side behavior where filters were applied
  // before summarize() ran) client-side instead of re-fetching.
  const filteredRows = useMemo(() => {
    if (!full || full.empty) return []
    return full.detail.filter(
      (r) => (source === 'All' || r.source === source) && (company === 'All' || r.company === company)
    )
  }, [full, source, company])

  const summary = useMemo(() => summarizePreview(filteredRows), [filteredRows])
  const scatterFig = useMemo(() => buildPredVsActualScatterFig(filteredRows), [filteredRows])

  if (notOk) {
    return (
      <div>
        <StageHeader number="08" title="Global Preview" subtitle="Past speeches: what the pipeline predicted vs. what actually happened." />
        <Alert type="error">Prediction history data failed to load: <code>{notOk}</code></Alert>
      </div>
    )
  }

  return (
    <div>
      <StageHeader number="08" title="Global Preview" subtitle="Past speeches: what the pipeline predicted vs. what actually happened." />

      <div className="field-row">
        <div className="field">
          <label>Source</label>
          <select value={source} onChange={(e) => setSource(e.target.value)}>
            <option>All</option>
            <option>Mann Ki Baat</option>
            <option>ECB</option>
            <option>Fed</option>
          </select>
        </div>
        <div className="field">
          <label>Company</label>
          <select value={company} onChange={(e) => setCompany(e.target.value)}>
            <option>All</option>
            {companies.map((c) => (
              <option key={c}>{c}</option>
            ))}
          </select>
        </div>
      </div>

      {full === undefined ? (
        <p className="spinner-note">Loading…</p>
      ) : full.empty || filteredRows.length === 0 ? (
        <Alert type="info">
          Not enough data for this filter — needs speeches with computed market impact (speech_market_impact).
        </Alert>
      ) : (
        <>
          <LedgerRow
            items={[
              { label: 'Directional Hit Rate (5D)', value: summary.overallHitRatePct != null ? `${summary.overallHitRatePct.toFixed(1)}%` : 'N/A', sub: 'sign(predicted) == sign(actual)' },
              { label: 'Mean Abs. Error (1D)', value: `${summary.meanAbsError1d.toFixed(2)}%` },
              { label: 'Mean Abs. Error (5D)', value: `${summary.meanAbsError5d.toFixed(2)}%` },
              { label: 'Speeches Covered', value: summary.nEvents.toLocaleString() },
            ]}
          />

          <h3>Predicted vs. Actual Return (5-Day)</h3>
          {scatterFig ? (
            <PlotlyChart fig={scatterFig} height={420} />
          ) : (
            <p className="caption">No events with a non-zero actual return to plot yet.</p>
          )}

          <h3>Speech-Level Detail</h3>
          <DataTable rows={filteredRows} />

          <div className="grid-2" style={{ marginTop: '1rem' }}>
            {summary.perCompany.length > 0 ? (
              <div>
                <h3>Accuracy by Company</h3>
                <DataTable rows={summary.perCompany} />
              </div>
            ) : null}
            {summary.perSource.length > 0 ? (
              <div>
                <h3>Accuracy by Source</h3>
                <DataTable rows={summary.perSource} />
              </div>
            ) : null}
          </div>
        </>
      )}
    </div>
  )
}
