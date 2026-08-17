import { useEffect, useState, useCallback } from 'react'
import { Routes, Route, useNavigate, useLocation } from 'react-router-dom'
import { api } from './api.js'

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
  const [running, setRunning] = useState(false)
  const navigate = useNavigate()
  const location = useLocation()

  const loadStatus = useCallback(() => {
    api.status().then(setStatus).catch(() => setStatus(null))
  }, [])

  useEffect(() => {
    loadStatus()
  }, [loadStatus])

  const handleRunPipeline = async () => {
    setRunning(true)
    try {
      const res = await api.runPipeline()
      if (res.success) {
        await loadStatus()
      } else {
        alert('Pipeline execution failed. Check logs/pipeline_error.log')
      }
    } catch (e) {
      alert('Pipeline execution failed: ' + e.message)
    } finally {
      setRunning(false)
    }
  }

  return (
    <div className="app-shell">
      <div className="topbar">
        <div className="topbar-brand">
          {status?.logoBase64 ? (
            <img src={`data:image/png;base64,${status.logoBase64}`} alt="" />
          ) : null}
          <div>
            <div className="topbar-brand-name">BaatSeBharat</div>
            <div className="topbar-brand-sub">Rhetoric &amp; Markets Intel.</div>
          </div>
        </div>
        <button onClick={handleRunPipeline} disabled={running}>
          {running ? 'Running…' : status?.modelsExist ? 'Run Pipeline Again' : 'Run Pipeline'}
        </button>
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
            <strong>❌ CRITICAL: Missing Required Pipeline Files</strong>
            <p>The following files are missing. Click "Run Pipeline" above to generate them.</p>
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
  const parts = []
  if (status.modelsExist) {
    parts.push({ color: 'var(--green)', text: `Updated ${status.lastUpdate}` })
  } else {
    parts.push({ color: 'var(--rust)', text: 'Pipeline not yet run' })
  }
  if (status.predOk && status.llmModeActive) {
    parts.push({ color: 'var(--green)', text: 'LLM mode active' })
  } else if (status.predOk) {
    parts.push({ color: 'var(--saffron)', text: 'AI: rule-based mode' })
  } else {
    parts.push({ color: 'var(--rust)', text: 'Prediction engine offline' })
  }
  return (
    <div className="status-strip">
      {parts.map((p, i) => (
        <span key={i}>
          {i > 0 ? <span className="sep">|</span> : null}
          <span className="dot" style={{ background: p.color }}></span>
          {p.text}
        </span>
      ))}
    </div>
  )
}
