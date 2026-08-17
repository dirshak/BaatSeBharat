"""
Stage 8 — Global Preview, ported from App.py's
`elif stage == "8. Global Preview":` block.
"""
import json
from fastapi import APIRouter
from typing import Optional

from backend.colors import apply_chart_theme, COLORS
from backend.jsonsafe import records

router = APIRouter()

try:
    from prediction_history import compute_prediction_vs_actual, summarize as summarize_prediction_history
    from prediction_engine import COMPANY_UNIVERSE
    _OK = True
except Exception as _err:
    _OK = False
    _err_msg = str(_err)


@router.get("/preview/companies")
def list_companies():
    if not _OK:
        return {"predHistOk": False, "error": _err_msg}
    return {"predHistOk": True, "companies": list(COMPANY_UNIVERSE.keys())}


@router.get("/preview")
def get_preview(source: Optional[str] = None, company: Optional[str] = None):
    import plotly.express as px

    if not _OK:
        return {"predHistOk": False, "error": _err_msg}

    gp_df = compute_prediction_vs_actual(
        source=None if (source is None or source == "All") else source,
        company=None if (company is None or company == "All") else company,
    )

    if gp_df.empty:
        return {"predHistOk": True, "empty": True}

    gp_summary = summarize_prediction_history(gp_df)
    hr = gp_summary.get("overall_hit_rate")

    gp_plot_df = gp_df.dropna(subset=["hit"]).copy()
    scatter_fig = None
    if not gp_plot_df.empty:
        gp_plot_df["Result"] = gp_plot_df["hit"].map({True: "Hit", False: "Miss"})
        fig_gp = px.scatter(
            gp_plot_df, x="predicted_return_5d", y="actual_return_5d",
            color="Result", hover_data=["date", "source", "company"],
            color_discrete_map={"Hit": COLORS["green"], "Miss": COLORS["rust"]},
        )
        lo = gp_plot_df["predicted_return_5d"].min()
        hi = gp_plot_df["predicted_return_5d"].max()
        fig_gp.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(color=COLORS["ink_dim"], dash="dot"))
        apply_chart_theme(fig_gp, height=420)
        fig_gp.update_layout(xaxis_title="Predicted 5D Return (%)", yaxis_title="Actual 5D Return (%)")
        scatter_fig = json.loads(fig_gp.to_json())

    detail = gp_df.sort_values("date", ascending=False)[[
        "date", "source", "company", "predicted_signal",
        "predicted_return_1d", "predicted_return_5d",
        "actual_return_1d", "actual_return_5d", "hit"
    ]].copy()
    detail["date"] = detail["date"].astype(str)

    per_company = gp_summary.get("per_company")
    per_source = gp_summary.get("per_source")

    return {
        "predHistOk": True,
        "empty": False,
        "summary": {
            "overallHitRatePct": (hr * 100) if hr is not None else None,
            "meanAbsError1d": gp_summary.get("mean_abs_error_1d", 0),
            "meanAbsError5d": gp_summary.get("mean_abs_error_5d", 0),
            "nEvents": gp_summary.get("n_events", 0),
        },
        "scatterFig": scatter_fig,
        "detail": records(detail),
        "perCompany": records(per_company),
        "perSource": records(per_source),
    }
