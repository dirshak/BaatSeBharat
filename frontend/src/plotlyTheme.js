/**
 * Plotly chrome constants, ported from backend/colors.py (itself ported
 * from App.py's original design system) so client-built charts match the
 * ones the backend used to render exactly.
 */
export const COLORS = {
  bg: "#0B1220",
  surface: "#131B2C",
  surface2: "#0F1727",
  ink: "#E8E4D9",
  ink_dim: "#9AA3B5",
  line: "#26324A",
  saffron: "#C97A2B",
  green: "#2F6F4E",
  rust: "#A6503A",
  navy: "#1B2A4A",
};

export const SIGNAL_COLORS = { Bullish: COLORS.green, Neutral: COLORS.ink_dim, Bearish: COLORS.rust };

export const CATEGORY_SEQUENCE = [COLORS.saffron, COLORS.green, COLORS.navy, COLORS.rust, "#5A7A9A", "#7A5A3A"];

export const DIVERGING_SCALE = [
  [0.0, COLORS.rust],
  [0.5, COLORS.surface2],
  [1.0, COLORS.green],
];

export const SEQUENTIAL_SCALE = [
  [0.0, COLORS.surface2],
  [0.5, "#5A4A2E"],
  [1.0, COLORS.saffron],
];

export const PLOTLY_LAYOUT_BASE = {
  paper_bgcolor: COLORS.surface,
  plot_bgcolor: COLORS.surface,
  font: { family: "IBM Plex Sans, sans-serif", color: COLORS.ink, size: 12 },
  title: { font: { family: "Fraunces, serif", color: COLORS.ink, size: 18 } },
  xaxis: {
    gridcolor: COLORS.line, zerolinecolor: COLORS.line, linecolor: COLORS.line,
    tickfont: { family: "IBM Plex Mono, monospace", color: COLORS.ink_dim, size: 11 },
  },
  yaxis: {
    gridcolor: COLORS.line, zerolinecolor: COLORS.line, linecolor: COLORS.line,
    tickfont: { family: "IBM Plex Mono, monospace", color: COLORS.ink_dim, size: 11 },
  },
  legend: { font: { family: "IBM Plex Sans, sans-serif", color: COLORS.ink_dim, size: 11 }, bgcolor: "rgba(0,0,0,0)" },
  margin: { t: 60, l: 10, r: 10, b: 10 },
};

export function baseLayout(extra = {}) {
  return { ...PLOTLY_LAYOUT_BASE, ...extra };
}
