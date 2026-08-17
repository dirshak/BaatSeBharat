import Plotly from 'plotly.js-dist-min'
import createPlotlyComponent from 'react-plotly.js/factory'

const Plot = createPlotlyComponent(Plotly)

export default function PlotlyChart({ fig, height }) {
  if (!fig) return null
  const layout = { ...fig.layout, autosize: true }
  if (height) layout.height = height
  return (
    <div className="plotly-chart">
      <Plot
        data={fig.data}
        layout={layout}
        config={{ responsive: true, displaylogo: false }}
        useResizeHandler
        style={{ width: '100%' }}
      />
    </div>
  )
}
