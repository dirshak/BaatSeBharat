"""
Stage 2 — NLP & Topic Modeling, ported from App.py's
`elif stage == "2. NLP Intelligence":` block.
"""
import os
import json
import numpy as np
import pandas as pd
from fastapi import APIRouter

from backend.cache import ttl_cache
from backend.colors import apply_chart_theme, COLORS, SEQUENTIAL_SCALE

router = APIRouter()

MODEL_OPTIONS = {
    "Combined (All Sources)": "topic_distributions_combined.npy",
    "Federal Reserve (Fed)": "topic_distributions_fed.npy",
    "European Central Bank (ECB)": "topic_distributions_ecb.npy",
    "Mann Ki Baat (MKB)": "topic_distributions_mann_ki_baat.npy",
}


def _labels_file_for(selected_model_name: str) -> str:
    return os.path.join(
        "./data/processed",
        f"topic_labels_{selected_model_name.lower().replace(' (all sources)', '').replace('federal reserve (fed)', 'fed').replace('european central bank (ecb)', 'ecb').replace('mann ki baat (mkb)', 'mann_ki_baat').replace(' ', '_')}.json"
    )


@ttl_cache(ttl_seconds=1800)
def _load_topics(current_topic_file):
    if os.path.exists(current_topic_file):
        return np.load(current_topic_file)
    return None


@router.get("/nlp/models")
def list_models():
    return {"models": list(MODEL_OPTIONS.keys())}


@router.get("/nlp/topics")
def get_topics(model: str = "Combined (All Sources)"):
    import plotly.express as px

    if model not in MODEL_OPTIONS:
        return {"error": f"Unknown model '{model}'"}

    current_topic_file = os.path.join("./data/processed", MODEL_OPTIONS[model])
    labels_file = _labels_file_for(model)

    labels_data = {}
    if os.path.exists(labels_file):
        with open(labels_file, 'r', encoding='utf-8') as f:
            labels_data = json.load(f)

    def topic_label(i):
        info = labels_data.get(str(i))
        return info['label'] if info else f"Topic {i+1}"

    topics = _load_topics(current_topic_file)
    if topics is None:
        return {"error": f"No results found for {model}.", "labelsFileMissing": not os.path.exists(labels_file)}

    topic_names = [topic_label(i) for i in range(topics.shape[1])]

    fig = px.bar(
        x=topic_names,
        y=topics[0],
        labels={'x': 'Topic', 'y': 'Probability'},
        title=f"Dominant Rhetoric Components ({model})",
    )
    fig.update_traces(marker_color=COLORS["saffron"])
    fig.update_layout(xaxis_tickangle=-30)
    apply_chart_theme(fig, height=420)
    bar_fig = json.loads(fig.to_json())

    heatmap_fig = None
    if topics.shape[0] > 1:
        n_show = min(30, topics.shape[0])
        heat_df = pd.DataFrame(topics[:n_show], columns=topic_names)
        fig_heat = px.imshow(
            heat_df.T,
            aspect="auto",
            color_continuous_scale=SEQUENTIAL_SCALE,
            title="Topic Probability Heatmap",
            labels={'y': 'Topic'},
        )
        apply_chart_theme(fig_heat, height=420)
        heatmap_fig = json.loads(fig_heat.to_json())

    keywords = []
    if labels_data:
        for i in range(topics.shape[1]):
            info = labels_data.get(str(i))
            if info:
                keywords.append({"label": info['label'], "keywords": info['keywords'][:5]})

    return {
        "model": model,
        "barFig": bar_fig,
        "heatmapFig": heatmap_fig,
        "keywords": keywords,
        "labelsFileMissing": not bool(labels_data),
        "labelsFile": labels_file,
        "nDocuments": int(topics.shape[0]),
        "nTopics": int(topics.shape[1]),
    }
