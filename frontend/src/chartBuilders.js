/**
 * Client-side Plotly figure builders for the charts that used to be built
 * server-side from computed predictions (backend/routers/predictions.py,
 * geo.py's indicator(), preview.py) -- since those predictions are now
 * computed client-side (predictionMath.js) from static baselines, the
 * figures that depend on them have to be built client-side too. Each
 * builder here is a direct port of the corresponding Python figure
 * construction (px.bar/px.scatter/go.Bar/go.Scattergeo calls), same
 * traces/colors/layout, just built as plain trace objects instead of via
 * Plotly Express.
 */
import { COLORS, SIGNAL_COLORS, CATEGORY_SEQUENCE, DIVERGING_SCALE, baseLayout } from "./plotlyTheme.js";

/** Mirrors all_company_predictions()'s fig_bar (predictions.py:246-254). */
export function buildCompanyBarFig(rows) {
  const sorted = [...rows].sort((a, b) => b.score - a.score);
  return {
    data: [
      {
        type: "bar",
        x: sorted.map((r) => r.company),
        y: sorted.map((r) => r.predictions[5].return_pct),
        marker: { color: sorted.map((r) => SIGNAL_COLORS[r.signal] ?? COLORS.ink_dim) },
        text: sorted.map((r) => `${r.predictions[5].return_pct >= 0 ? "+" : ""}${r.predictions[5].return_pct.toFixed(2)}%`),
        textposition: "outside",
        hovertemplate: "<b>%{x}</b><br>5D: %{y:+.2f}%<extra></extra>",
      },
    ],
    layout: baseLayout({
      height: 370,
      title: { text: "All Companies — 5-Day Return Forecast" },
      xaxis: { tickangle: -35 },
      yaxis: { title: { text: "Forecast Return (%)" } },
      shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, yref: "y", y0: 0, y1: 0, line: { dash: "dot", color: COLORS.line } }],
    }),
  };
}

/** Mirrors all_company_predictions()'s fig_sc (px.scatter, predictions.py:256-260). */
export function buildCompanyScatterFig(rows) {
  const bySignal = { Bullish: [], Neutral: [], Bearish: [] };
  for (const r of rows) (bySignal[r.signal] ??= []).push(r);

  const data = Object.entries(bySignal)
    .filter(([, group]) => group.length > 0)
    .map(([signal, group]) => ({
      type: "scatter",
      mode: "markers",
      name: signal,
      x: group.map((r) => r.score),
      y: group.map((r) => r.confidence),
      text: group.map((r) => r.company),
      marker: { color: SIGNAL_COLORS[signal], size: group.map((r) => Math.max(6, r.confidence / 4)) },
      hovertemplate: "<b>%{text}</b><br>Score: %{x}<br>Confidence: %{y}<extra></extra>",
    }));

  return {
    data,
    layout: baseLayout({
      height: 330,
      title: { text: "Signal Score vs Prediction Confidence" },
      xaxis: { title: { text: "Score" } },
      yaxis: { title: { text: "Confidence" } },
    }),
  };
}

/** Mirrors sector_predictions()'s fig_sec (predictions.py:279-297). */
export function buildSectorBarFig(rows) {
  const sorted = [...rows].sort((a, b) => b.score - a.score);
  const horizonCols = [
    { key: 1, label: "1-Day" },
    { key: 5, label: "1-Week" },
    { key: 10, label: "10-Day" },
  ];

  const data = horizonCols.map(({ key, label }, i) => ({
    type: "bar",
    name: label,
    x: sorted.map((r) => r.sector),
    y: sorted.map((r) => r.predictions[key].return_pct),
    text: sorted.map((r) => `${r.predictions[key].return_pct >= 0 ? "+" : ""}${r.predictions[key].return_pct.toFixed(2)}%`),
    textposition: "outside",
    marker: { color: CATEGORY_SEQUENCE[i % CATEGORY_SEQUENCE.length] },
  }));

  return {
    data,
    layout: baseLayout({
      height: 370,
      title: { text: "Sector Return Forecasts — 1D / 1W / 10D" },
      barmode: "group",
      yaxis: { title: { text: "Forecast Return (%)" } },
      legend: { orientation: "h", yanchor: "bottom", y: 1.02, xanchor: "right", x: 1 },
      shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, yref: "y", y0: 0, y1: 0, line: { dash: "dot", color: COLORS.line } }],
    }),
  };
}

/**
 * Mirrors sector_prediction_map() (predictions.py:328-408) -- skips
 * "Broad Market" (would overlap every other sector's markers) and any
 * sector with zero mapped constituents, same as the backend did.
 */
export function buildSectorMapFig(sectorRows, sectorBaselines, companyLocations) {
  const byName = Object.fromEntries(sectorRows.map((r) => [r.sector, r]));
  const markers = [];
  const skippedSectors = [];

  for (const baseline of sectorBaselines) {
    const row = byName[baseline.sector];
    if (!row) continue;
    if (baseline.sector === "Broad Market" || baseline.constituentCompanies.length === 0) {
      skippedSectors.push(baseline.sector);
      continue;
    }
    const ret5 = row.predictions[5].return_pct;
    const score = row.score ?? ret5;
    for (const company of baseline.constituentCompanies) {
      const loc = companyLocations[company];
      if (!loc) continue;
      markers.push({
        sector: baseline.sector, company, city: loc.city, lat: loc.lat, lon: loc.lon,
        score, return5d: ret5, signal: row.signal, confidence: row.confidence,
      });
    }
  }

  if (markers.length === 0) return { fig: null, skippedSectors };

  const maxAbs = Math.max(...markers.map((m) => Math.abs(m.score)), 1e-6);

  const fig = {
    data: [
      {
        type: "scattergeo",
        mode: "markers",
        lat: markers.map((m) => m.lat),
        lon: markers.map((m) => m.lon),
        marker: {
          size: markers.map((m) => m.confidence / 8 + 6),
          color: markers.map((m) => m.score),
          colorscale: DIVERGING_SCALE,
          cmin: -maxAbs,
          cmax: maxAbs,
          colorbar: { title: { text: "Predicted<br>Effect", font: { color: COLORS.ink_dim } }, tickfont: { color: COLORS.ink_dim } },
          line: { width: 1, color: "white" },
          opacity: 0.9,
        },
        customdata: markers.map((m) => [m.sector, m.company, m.city, m.return5d, m.signal]),
        hovertemplate:
          "<b>%{customdata[1]}</b> (%{customdata[0]})<br>%{customdata[2]}<br>" +
          "5D Forecast: %{customdata[3]:+.2f}%<br>Signal: %{customdata[4]}<extra></extra>",
      },
    ],
    layout: {
      title: { text: "Sector Prediction Map — Bullish (green) → Bearish (red)", font: { family: "Fraunces, serif", color: COLORS.ink, size: 18 } },
      height: 460,
      paper_bgcolor: COLORS.bg,
      plot_bgcolor: COLORS.bg,
      font: { family: "IBM Plex Sans, sans-serif", color: COLORS.ink },
      margin: { l: 0, r: 0, t: 50, b: 0 },
      geo: {
        projection: { type: "natural earth", scale: 3.5 },
        showcoastlines: true,
        showcountries: true,
        countrycolor: COLORS.line,
        coastlinecolor: COLORS.line,
        bgcolor: COLORS.bg,
        landcolor: COLORS.surface,
        lakecolor: COLORS.bg,
        center: { lat: 20, lon: 78 },
      },
    },
  };

  return { fig, skippedSectors };
}

/**
 * Builds an animated choropleth from a raw Country/Year/Value table --
 * mirrors geo.py's indicator() px.choropleth call (geo.py:139-149), but
 * built directly from the exported raw table (see Step-2 audit for why:
 * this lets the country-filter reshape the trace client-side instead of
 * needing a backend to rebuild the figure).
 */
export function buildIndicatorChoroplethFig(rows, indicatorName, selectedCountries) {
  const selected = new Set(selectedCountries);
  const filtered = rows.filter((r) => selected.has(r.Country));
  const years = [...new Set(filtered.map((r) => r.Year))].sort((a, b) => a - b);

  const frameFor = (year) => {
    const yearRows = filtered.filter((r) => r.Year === year);
    return {
      locations: yearRows.map((r) => r.Country),
      z: yearRows.map((r) => r[indicatorName]),
      text: yearRows.map((r) => r.Country),
    };
  };

  const lastYear = years[years.length - 1];
  const initial = frameFor(lastYear);

  return {
    data: [
      {
        type: "choropleth",
        locationmode: "country names",
        locations: initial.locations,
        z: initial.z,
        text: initial.text,
        colorscale: [
          [0.0, "#0F1727"],
          [0.5, "#5A4A2E"],
          [1.0, "#C97A2B"],
        ],
        colorbar: { title: { text: indicatorName } },
      },
    ],
    layout: {
      title: { text: `${indicatorName} (${years[0]}–${lastYear})`, font: { family: "Fraunces, serif", color: COLORS.ink, size: 18 } },
      height: 500,
      font: { family: "IBM Plex Sans, sans-serif", color: COLORS.ink },
      paper_bgcolor: COLORS.bg,
      geo: { projection: { type: "natural earth" }, showcoastlines: true, showcountries: true, bgcolor: COLORS.bg, landcolor: COLORS.surface },
      sliders: [
        {
          active: years.length - 1,
          steps: years.map((y) => ({ label: String(y), method: "skip" })),
        },
      ],
    },
    frames: years.map((y) => ({ name: String(y), data: [frameFor(y)] })),
  };
}

/** Mirrors preview.py's scatter (preview.py:44-54). */
export function buildPredVsActualScatterFig(rows) {
  const withHit = rows.filter((r) => r.hit !== null && r.hit !== undefined);
  const hits = withHit.filter((r) => r.hit);
  const misses = withHit.filter((r) => !r.hit);
  if (withHit.length === 0) return null;

  const allX = withHit.map((r) => r.predicted_return_5d);
  const lo = Math.min(...allX);
  const hi = Math.max(...allX);

  return {
    data: [
      {
        type: "scatter", mode: "markers", name: "Hit",
        x: hits.map((r) => r.predicted_return_5d), y: hits.map((r) => r.actual_return_5d),
        marker: { color: COLORS.green },
        customdata: hits.map((r) => [r.date, r.source, r.company]),
        hovertemplate: "date=%{customdata[0]}<br>source=%{customdata[1]}<br>company=%{customdata[2]}<extra></extra>",
      },
      {
        type: "scatter", mode: "markers", name: "Miss",
        x: misses.map((r) => r.predicted_return_5d), y: misses.map((r) => r.actual_return_5d),
        marker: { color: COLORS.rust },
        customdata: misses.map((r) => [r.date, r.source, r.company]),
        hovertemplate: "date=%{customdata[0]}<br>source=%{customdata[1]}<br>company=%{customdata[2]}<extra></extra>",
      },
    ],
    layout: baseLayout({
      height: 420,
      xaxis: { title: { text: "Predicted 5D Return (%)" } },
      yaxis: { title: { text: "Actual 5D Return (%)" } },
      shapes: [{ type: "line", x0: lo, y0: lo, x1: hi, y1: hi, line: { color: COLORS.ink_dim, dash: "dot" } }],
    }),
  };
}

const mean = (vals) => (vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0);
const absDiff = (rows, predKey, actualKey) => mean(rows.map((r) => Math.abs(r[predKey] - r[actualKey])));
const hitRate = (rows) => {
  const withHit = rows.filter((r) => r.hit !== null && r.hit !== undefined);
  return withHit.length ? withHit.filter((r) => r.hit).length / withHit.length : null;
};

/**
 * Client-side recompute of prediction_history.summarize() (Stage 8),
 * over whatever subset the Source/Company dropdowns have filtered `rows`
 * to -- direct port of prediction_history.py:227-260, so the summary
 * metrics stay exactly in sync with the filtered detail table, matching
 * the original server-side behavior where the filters were applied
 * BEFORE summarize() ran.
 */
export function summarizePreview(rows) {
  if (rows.length === 0) return null;

  const groupBy = (key) => {
    const groups = {};
    for (const r of rows) (groups[r[key]] ??= []).push(r);
    return Object.entries(groups).map(([name, group]) => ({
      [key]: name,
      n: group.length,
      hit_rate: hitRate(group),
      mae_5d: absDiff(group, "predicted_return_5d", "actual_return_5d"),
    }));
  };

  const hr = hitRate(rows);
  return {
    overallHitRatePct: hr !== null ? hr * 100 : null,
    meanAbsError1d: absDiff(rows, "predicted_return_1d", "actual_return_1d"),
    meanAbsError5d: absDiff(rows, "predicted_return_5d", "actual_return_5d"),
    nEvents: rows.length,
    perCompany: groupBy("company"),
    perSource: groupBy("source"),
  };
}
