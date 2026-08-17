import { useEffect, useState } from 'react'
import { api } from '../api.js'
import StageHeader from '../components/StageHeader.jsx'
import LedgerRow from '../components/LedgerRow.jsx'
import PlotlyChart from '../components/PlotlyChart.jsx'
import DataTable from '../components/DataTable.jsx'
import Alert from '../components/Alert.jsx'

export default function GlobalPreview() {
  const [companies, setCompanies] = useState([])
  const [source, setSource] = useState('All')
  const [company, setCompany] = useState('All')
  const [data, setData] = useState(undefined)
  const [notOk, setNotOk] = useState(null)

  useEffect(() => {
    api.previewCompanies().then((d) => {
      if (d.predHistOk) setCompanies(d.companies)
      else setNotOk(d.error)
    })
  }, [])

  useEffect(() => {
    setData(undefined)
    api.preview(source, company).then((d) => {
      if (!d.predHistOk) {
        setNotOk(d.error)
        return
      }
      setData(d)
    })
  }, [source, company])

  if (notOk) {
    return (
      <div>
        <StageHeader number="08" title="Global Preview" subtitle="Past speeches: what the pipeline predicted vs. what actually happened." />
        <Alert type="error">Prediction history module failed to load: <code>{notOk}</code></Alert>
        <Alert type="info">Ensure `src/prediction_history.py` is present.</Alert>
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

      {data === undefined ? (
        <p className="spinner-note">Replaying past predictions against realized outcomes…</p>
      ) : data.empty ? (
        <Alert type="info">
          Not enough data yet to build a Global Preview for this filter — needs speeches with computed market impact
          (speech_market_impact).
        </Alert>
      ) : (
        <>
          <LedgerRow
            items={[
              { label: 'Directional Hit Rate (5D)', value: data.summary.overallHitRatePct != null ? `${data.summary.overallHitRatePct.toFixed(1)}%` : 'N/A', sub: 'sign(predicted) == sign(actual)' },
              { label: 'Mean Abs. Error (1D)', value: `${data.summary.meanAbsError1d.toFixed(2)}%` },
              { label: 'Mean Abs. Error (5D)', value: `${data.summary.meanAbsError5d.toFixed(2)}%` },
              { label: 'Speeches Covered', value: data.summary.nEvents.toLocaleString() },
            ]}
          />

          <h3>Predicted vs. Actual Return (5-Day)</h3>
          {data.scatterFig ? (
            <PlotlyChart fig={data.scatterFig} height={420} />
          ) : (
            <p className="caption">No events with a non-zero actual return to plot yet.</p>
          )}

          <h3>Speech-Level Detail</h3>
          <DataTable rows={data.detail} />

          <div className="grid-2" style={{ marginTop: '1rem' }}>
            {data.perCompany.length > 0 ? (
              <div>
                <h3>Accuracy by Company</h3>
                <DataTable rows={data.perCompany} />
              </div>
            ) : null}
            {data.perSource.length > 0 ? (
              <div>
                <h3>Accuracy by Source</h3>
                <DataTable rows={data.perSource} />
              </div>
            ) : null}
          </div>
        </>
      )}
    </div>
  )
}
