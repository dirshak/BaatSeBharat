"""
Shared Plotly chart chrome / color palette, ported verbatim from App.py's
design-system section so every stage's figures keep the exact same look
after the React migration.
"""

COLORS = {
    "bg":        "#0B1220",
    "surface":   "#131B2C",
    "surface2":  "#0F1727",
    "ink":       "#E8E4D9",
    "ink_dim":   "#9AA3B5",
    "line":      "#26324A",
    "saffron":   "#C97A2B",
    "saffron_dim": "#8A5A24",
    "green":     "#2F6F4E",
    "green_dim": "#1F4A34",
    "rust":      "#A6503A",
    "navy":      "#1B2A4A",
}

SOURCE_COLORS = {
    'Mann Ki Baat': COLORS["saffron"],
    'ECB':          "#5A7A9A",
    'Fed':          COLORS["green"],
}

PLOTLY_TEMPLATE = dict(
    paper_bgcolor=COLORS["surface"],
    plot_bgcolor=COLORS["surface"],
    font=dict(family="IBM Plex Sans, sans-serif", color=COLORS["ink"], size=12),
    title_font=dict(family="Fraunces, serif", color=COLORS["ink"], size=18),
    xaxis=dict(gridcolor=COLORS["line"], zerolinecolor=COLORS["line"], linecolor=COLORS["line"],
                tickfont=dict(family="IBM Plex Mono, monospace", color=COLORS["ink_dim"], size=11)),
    yaxis=dict(gridcolor=COLORS["line"], zerolinecolor=COLORS["line"], linecolor=COLORS["line"],
                tickfont=dict(family="IBM Plex Mono, monospace", color=COLORS["ink_dim"], size=11)),
    legend=dict(font=dict(family="IBM Plex Sans, sans-serif", color=COLORS["ink_dim"], size=11),
                bgcolor="rgba(0,0,0,0)"),
    margin=dict(t=60, l=10, r=10, b=10),
)

CATEGORY_SEQUENCE = [COLORS["saffron"], COLORS["green"], COLORS["navy"], COLORS["rust"], "#5A7A9A", "#7A5A3A"]

SEQUENTIAL_SCALE = [[0.0, COLORS["surface2"]], [0.5, "#5A4A2E"], [1.0, COLORS["saffron"]]]
DIVERGING_SCALE = [[0.0, COLORS["rust"]], [0.5, COLORS["surface2"]], [1.0, COLORS["green"]]]


def apply_chart_theme(fig, height=None):
    """Apply the shared design-system chrome to a Plotly figure in place."""
    fig.update_layout(**PLOTLY_TEMPLATE)
    if height:
        fig.update_layout(height=height)
    return fig
