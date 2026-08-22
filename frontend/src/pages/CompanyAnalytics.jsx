import { useEffect, useState } from 'react'
import { dataClient } from '../dataClient.js'
import StageHeader from '../components/StageHeader.jsx'
import PlotlyChart from '../components/PlotlyChart.jsx'
import Alert from '../components/Alert.jsx'

export default function CompanyAnalytics() {
  const [companies, setCompanies] = useState([])
  const [company, setCompany] = useState(null)
  const [data, setData] = useState(null)

  useEffect(() => {
    dataClient.companyAnalyticsCompanies().then((cs) => {
      setCompanies(cs)
      if (cs.length > 0) setCompany(cs[0])
    })
  }, [])

  useEffect(() => {
    if (!company) return
    setData(null)
    dataClient.companyAnalytics(company).then(setData)
  }, [company])

  return (
    <div>
      <StageHeader
        number="05"
        title="Company Specific Returns vs. Rhetoric"
        subtitle="Analyzing how leadership topics impact individual company performance."
      />

      <div className="field">
        <label>Select Company</label>
        <select value={company ?? ''} onChange={(e) => setCompany(e.target.value)}>
          {companies.map((c) => (
            <option key={c}>{c}</option>
          ))}
        </select>
      </div>

      {!data ? (
        <p className="spinner-note">Loading…</p>
      ) : data.empty ? (
        <Alert type="warning">No topic-impact data found for {company}. Waiting on the next scheduled pipeline run.</Alert>
      ) : (
        <>
          <h3>{data.company} Topic Impact Heatmap</h3>
          <PlotlyChart fig={data.fig} height={450} />
          <Alert type="info">
            💡 Each cell is the average of (topic probability × {data.company}'s 5-day forward return) for speeches
            in that month. Green = that theme historically preceded {data.company} gains; red = historically
            preceded {data.company} declines.
          </Alert>
        </>
      )}
    </div>
  )
}
