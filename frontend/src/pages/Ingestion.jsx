import { useEffect, useState } from 'react'
import { api } from '../api.js'
import StageHeader from '../components/StageHeader.jsx'
import Tabs from '../components/Tabs.jsx'
import PlotlyChart from '../components/PlotlyChart.jsx'
import Alert from '../components/Alert.jsx'

function SpeechesTab() {
  const [sourceFilter, setSourceFilter] = useState('All')
  const [data, setData] = useState(null)
  const [selectedId, setSelectedId] = useState(null)
  const [text, setText] = useState('')

  useEffect(() => {
    api.ingestionSpeeches(sourceFilter).then((d) => {
      setData(d)
      if (d.speeches.length > 0) setSelectedId(d.speeches[0].id)
    })
  }, [sourceFilter])

  useEffect(() => {
    if (selectedId != null) {
      api.ingestionSpeechText(selectedId).then((d) => setText(d.fullText || '(no text)'))
    }
  }, [selectedId])

  if (!data) return <p className="spinner-note">Loading…</p>
  if (data.speeches.length === 0) return <Alert type="info">Database empty. Run the pipeline first.</Alert>

  const selected = data.speeches.find((s) => s.id === selectedId)

  return (
    <div>
      <div className="field-row">
        <div className="field">
          <label>Filter by Source</label>
          <select value={sourceFilter} onChange={(e) => setSourceFilter(e.target.value)}>
            <option>All</option>
            {data.sources.map((s) => (
              <option key={s}>{s}</option>
            ))}
          </select>
        </div>
        <div className="field" style={{ flex: '2 1 400px' }}>
          <label>Select Speech to Preview</label>
          <select value={selectedId ?? ''} onChange={(e) => setSelectedId(Number(e.target.value))}>
            {data.speeches.map((s) => (
              <option key={s.id} value={s.id}>
                [{s.id}] {s.source} | {s.date || 'N/A'} | {s.title || 'Untitled'}
              </option>
            ))}
          </select>
        </div>
      </div>

      {selected ? (
        <p>
          <strong>Source:</strong> {selected.source} &nbsp;|&nbsp; <strong>Speaker:</strong>{' '}
          {selected.speaker || 'N/A'} &nbsp;|&nbsp; <strong>Date:</strong> {selected.date}
        </p>
      ) : null}
      <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--ink-dim)', margin: '0.6rem 0 0.3rem' }}>
        Transcript Preview
      </label>
      <textarea
        readOnly
        value={text}
        style={{
          width: '100%', height: 250, background: 'var(--surface)', color: 'var(--ink)',
          border: '1px solid var(--line)', borderRadius: 4, padding: '0.6rem', fontFamily: 'IBM Plex Mono, monospace',
          fontSize: '0.82rem',
        }}
      />
    </div>
  )
}

function MarketTab() {
  const [fig, setFig] = useState(undefined)

  useEffect(() => {
    api.ingestionMarket().then((d) => setFig(d.fig))
  }, [])

  if (fig === undefined) return <p className="spinner-note">Loading…</p>
  if (fig === null) return <Alert type="info">No market data. Run the pipeline first.</Alert>
  return <PlotlyChart fig={fig} height={420} />
}

export default function Ingestion() {
  return (
    <div>
      <StageHeader number="01" title="Data Ingestion & Storage" />
      <Tabs
        tabs={[
          { label: 'Speeches (Text)', content: <SpeechesTab /> },
          { label: 'Market (Numerical)', content: <MarketTab /> },
        ]}
      />
    </div>
  )
}
