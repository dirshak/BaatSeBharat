import { useEffect, useState } from 'react'
import { Routes, Route, useNavigate, useLocation } from 'react-router-dom'
import { dataClient } from './dataClient.js'

import Overview from './pages/Overview.jsx'
import Ingestion from './pages/Ingestion.jsx'
import Nlp from './pages/Nlp.jsx'
import MarketImpact from './pages/MarketImpact.jsx'
import Regime from './pages/Regime.jsx'
import CompanyAnalytics from './pages/CompanyAnalytics.jsx'
import Predictions from './pages/Predictions.jsx'
import GlobalInfluenceMap from './pages/GlobalInfluenceMap.jsx'
import GlobalPreview from './pages/GlobalPreview.jsx'

const NAV = [
  { path: '/', label: 'Overview' },
  { path: '/ingestion', label: '01 Ingestion' },
  { path: '/nlp', label: '02 NLP' },
  { path: '/market-impact', label: '03 Impact' },
  { path: '/regime', label: '04 Regime' },
  { path: '/company-analytics', label: '05 Company' },
  { path: '/predictions', label: '06 Predictions' },
  { path: '/global-influence', label: '07 Global' },
  { path: '/preview', label: '08 Preview' },
]

export default function App() {
  const [status, setStatus] = useState(null)
  const navigate = useNavigate()
  const location = useLocation()

  useEffect(() => {
    dataClient.status().then(setStatus).catch(() => setStatus(null))
  }, [])

  return (
    <div className="app-shell">
      <div className="topbar">
        <div className="topbar-brand">
          <img src="/logo.png" alt="" />
          <div>
            <div className="topbar-brand-name">BaatSeBharat</div>
            <div className="topbar-brand-sub">Rhetoric &amp; Markets Intel.</div>
          </div>
        </div>
      </div>

      <nav className="stage-nav">
        {NAV.map((item) => (
          <button
            key={item.path}
            className={location.pathname === item.path ? 'active' : ''}
            onClick={() => navigate(item.path)}
          >
            {item.label}
          </button>
        ))}
      </nav>

      <StatusStrip status={status} />

      {status && status.missingFiles && status.missingFiles.length > 0 ? (
        <div className="main-content">
          <div className="alert alert-error">
            <strong>❌ Missing pipeline output files as of the last scheduled run</strong>
            <ul>
              {status.missingFiles.map((f) => (
                <li key={f}><code>{f}</code></li>
              ))}
            </ul>
          </div>
        </div>
      ) : null}

      <div className="main-content">
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/ingestion" element={<Ingestion />} />
          <Route path="/nlp" element={<Nlp />} />
          <Route path="/market-impact" element={<MarketImpact />} />
          <Route path="/regime" element={<Regime />} />
          <Route path="/company-analytics" element={<CompanyAnalytics />} />
          <Route path="/predictions" element={<Predictions />} />
          <Route path="/global-influence" element={<GlobalInfluenceMap />} />
          <Route path="/preview" element={<GlobalPreview />} />
        </Routes>
      </div>
    </div>
  )
}

function StatusStrip({ status }) {
  if (!status) return null
  const color = status.modelsExist ? 'var(--green)' : 'var(--rust)'
  const text = status.modelsExist ? `Data updated ${status.lastUpdate}` : 'No pipeline data yet'
  return (
    <div className="status-strip">
      <span className="dot" style={{ background: color }}></span>
      {text}
    </div>
  )
}
