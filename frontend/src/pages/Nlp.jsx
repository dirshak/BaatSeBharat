import { useEffect, useState } from 'react'
import { api } from '../api.js'
import StageHeader from '../components/StageHeader.jsx'
import PlotlyChart from '../components/PlotlyChart.jsx'
import Alert from '../components/Alert.jsx'

export default function Nlp() {
  const [models, setModels] = useState([])
  const [model, setModel] = useState('Combined (All Sources)')
  const [data, setData] = useState(null)

  useEffect(() => {
    api.nlpModels().then((d) => setModels(d.models))
  }, [])

  useEffect(() => {
    setData(null)
    api.nlpTopics(model).then(setData)
  }, [model])

  return (
    <div>
      <StageHeader number="02" title="NLP & Topic Modeling" />
      <p>
        Topic modeling analyzes the underlying themes in leadership speeches. Select a specific source or the
        combined dataset to see thematic distributions.
      </p>

      <div className="field">
        <label>Select Topic Model</label>
        <select value={model} onChange={(e) => setModel(e.target.value)}>
          {models.map((m) => (
            <option key={m}>{m}</option>
          ))}
        </select>
      </div>

      {!data ? (
        <p className="spinner-note">Loading…</p>
      ) : data.error ? (
        <>
          <Alert type="warning">{data.error}</Alert>
          <Alert type="info">💡 Use the "Run Pipeline" button above to generate results.</Alert>
        </>
      ) : (
        <>
          <h3>Topic distribution: {data.model}</h3>
          <p className="caption">TF-IDF + NMF topic model for {data.model}, deterministically labeled from top keywords.</p>

          <PlotlyChart fig={data.barFig} height={420} />

          {data.heatmapFig ? (
            <>
              <h3>Topic Heatmap (First 30 Speeches — {data.model})</h3>
              <PlotlyChart fig={data.heatmapFig} height={420} />
            </>
          ) : null}

          <div className="grid-2">
            <div>
              <h3>Topics &amp; Top Keywords</h3>
              {data.keywords.length > 0 ? (
                data.keywords.map((k) => (
                  <p key={k.label}>
                    <strong>{k.label}:</strong> {k.keywords.join(', ')}
                  </p>
                ))
              ) : (
                <Alert type="warning">No label file found at <code>{data.labelsFile}</code>. Run the pipeline to generate topic labels.</Alert>
              )}
            </div>
            <div>
              <h3>Model Insight</h3>
              <Alert type="info">
                Model trained on {data.nDocuments} documents with {data.nTopics} latent topics.
              </Alert>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
