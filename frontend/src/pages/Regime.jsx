import { useEffect, useState } from 'react'
import { api } from '../api.js'
import StageHeader from '../components/StageHeader.jsx'
import PlotlyChart from '../components/PlotlyChart.jsx'
import Alert from '../components/Alert.jsx'

export default function Regime() {
  const [tickers, setTickers] = useState([])
  const [ticker, setTicker] = useState(null)
  const [data, setData] = useState(null)

  useEffect(() => {
    api.regimeTickers().then((d) => {
      setTickers(d.tickers)
      if (d.tickers.length > 0) setTicker(d.tickers[0])
    })
  }, [])

  useEffect(() => {
    if (!ticker) return
    setData(null)
    api.regime(ticker).then(setData)
  }, [ticker])

  return (
    <div>
      <StageHeader
        number="04"
        title="Market Regime Intelligence (HMM)"
        subtitle="Quantifying structural market shifts using Hidden Markov Models."
      />

      {!data ? (
        <p className="spinner-note">Loading…</p>
      ) : data.empty ? (
        <Alert type="warning">No regime data found. Run the pipeline first.</Alert>
      ) : (
        <>
          <div className="field">
            <label>Select Ticker for Regime Timeline</label>
            <select value={ticker} onChange={(e) => setTicker(e.target.value)}>
              {tickers.map((t) => (
                <option key={t}>{t}</option>
              ))}
            </select>
          </div>

          <PlotlyChart fig={data.fig} height={550} />

          {!data.hasRegimeData ? (
            <Alert type="info">
              No HMM regime classifications computed yet for <strong>{ticker}</strong>. Regime data currently only
              covers a subset of tickers ({data.coveredTickers.join(', ')}) — run the regime modeling step
              (src/models/market_modeling.py) for the rest.
            </Alert>
          ) : null}
        </>
      )}
    </div>
  )
}
