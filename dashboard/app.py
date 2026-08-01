"""
dashboard/app.py
AI Agent Workflow Quality Monitor — Streamlit dashboard.
Reads from data/all_traces.jsonl (merged CrewAI + OpenDeepResearch traces).
Uses normalize_trace() so both schemas are handled correctly.
"""

import json
import sys
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Allow imports from repo root when running via `streamlit run dashboard/app.py`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evals.scorer import normalize_trace, score_trace
from evals.drift_detector import compute_ewma, calibrate_threshold, load_traces

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRACE_PATH = "data/all_traces.jsonl"

st.set_page_config(
    page_title="AI Agent Workflow Quality Monitor",
    page_icon="🧠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Load + normalize
# ---------------------------------------------------------------------------
@st.cache_data
def load_and_process(path: str):
    raw_traces = load_traces(path)
    normed     = [normalize_trace(t) for t in raw_traces]
    scores     = [score_trace(t)     for t in raw_traces]
    ewma       = compute_ewma(scores)
    threshold  = calibrate_threshold(scores)
    return normed, scores, ewma, threshold

normed, scores, ewma_series, threshold = load_and_process(TRACE_PATH)

latest_ewma    = ewma_series[-1]
drift_detected = latest_ewma < threshold

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🧠 AI Agent Workflow Quality Monitor")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Traces Analysed", len(normed))
col2.metric("Latest EWMA Score", f"{latest_ewma:.3f}",
            delta=f"Threshold: {threshold:.4f}")
col3.metric(
    "Drift Status",
    "⚠️ DRIFT DETECTED" if drift_detected else "✅ Stable",
    delta_color="inverse"
)
col4.metric(
    "Overall Success Rate",
    f"{sum(n['success'] for n in normed) / len(normed):.1%}"
)

st.divider()

# ---------------------------------------------------------------------------
# Per-agent breakdown
# ---------------------------------------------------------------------------
st.subheader("Agent Comparison")

crew  = [n for n in normed if n["agent_system"] == "crewai"]
odr   = [n for n in normed if n["agent_system"] == "open_deep_research"]

a1, a2 = st.columns(2)
with a1:
    st.markdown("**CrewAI**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Traces",       len(crew))
    c2.metric("Success Rate", f"{sum(n['success'] for n in crew)/len(crew):.1%}" if crew else "—")
    c3.metric("Avg Latency",  f"{sum(n['total_latency_ms'] for n in crew)/len(crew)/1000:.1f}s" if crew else "—")

with a2:
    st.markdown("**OpenDeepResearch**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Traces",       len(odr))
    c2.metric("Success Rate", f"{sum(n['success'] for n in odr)/len(odr):.1%}" if odr else "—")
    c3.metric("Avg Latency",  f"{sum(n['total_latency_ms'] for n in odr)/len(odr)/1000:.1f}s" if odr else "—")

st.divider()

# ---------------------------------------------------------------------------
# EWMA line chart
# ---------------------------------------------------------------------------
st.subheader("EWMA Score Over Time")

df_chart = pd.DataFrame({
    "Trace Index": range(len(scores)),
    "Raw Score":   scores,
    "EWMA Score":  ewma_series,
    "Agent":       [n["agent_system"] for n in normed],
})

fig = px.line(
    df_chart, x="Trace Index", y=["Raw Score", "EWMA Score"],
    title=f"Agent Performance Drift  (EWMA α=0.3)",
    color_discrete_map={"Raw Score": "#636EFA", "EWMA Score": "#00CC96"},
)
fig.add_hline(
    y=threshold, line_dash="dash", line_color="red",
    annotation_text=f"Drift Threshold ({threshold:.3f})",
    annotation_position="bottom right",
)
fig.update_layout(
    xaxis_title="Trace Index",
    yaxis_title="Score",
    yaxis=dict(range=[0, 1.05]),
    legend_title="Series",
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Failure category breakdown  ← FIXED: uses normalize_trace fields
# ---------------------------------------------------------------------------
st.subheader("Failure Category Breakdown")


def get_outcome_label(norm: dict) -> str:
    """
    Map a normalised trace to a human-readable outcome label.

    Priority:
      1. success=True                     → "Success"
      2. error_types list has entries     → first entry (e.g. "timeout", "hallucination")
      3. has_error_span=True              → "Error Span (unlabelled)"
      4. faulty_batch=True                → "Faulty Batch"
      5. fallback                         → "Unknown Failure"
    """
    if norm["success"]:
        return "Success"
    error_types = norm.get("error_types") or []
    if error_types:
        return str(error_types[0]).replace("_", " ").title()
    if norm.get("has_error_span"):
        return "Error Span (unlabelled)"
    if norm.get("faulty_batch"):
        return "Faulty Batch"
    return "Unknown Failure"


labels      = [get_outcome_label(n) for n in normed]
label_df    = pd.Series(labels).value_counts().reset_index()
label_df.columns = ["Outcome", "Count"]

pie_col, table_col = st.columns([2, 1])

with pie_col:
    fig2 = px.pie(
        label_df, names="Outcome", values="Count",
        title="Trace Outcome Distribution",
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig2.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig2, use_container_width=True)

with table_col:
    st.markdown("**Outcome counts**")
    st.dataframe(label_df, use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# Per-agent failure breakdown
# ---------------------------------------------------------------------------
st.subheader("Failure Breakdown by Agent")

df_agents = pd.DataFrame({
    "Agent":   [n["agent_system"] for n in normed],
    "Outcome": labels,
})

fig3 = px.histogram(
    df_agents, x="Agent", color="Outcome",
    barmode="stack",
    title="Outcome Distribution per Agent System",
    color_discrete_sequence=px.colors.qualitative.Safe,
)
st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Raw data explorer (collapsed by default)
# ---------------------------------------------------------------------------
with st.expander("🔍 Raw normalised trace data"):
    df_raw = pd.DataFrame(normed)
    df_raw["score"] = scores
    df_raw["ewma"]  = ewma_series
    st.dataframe(df_raw, use_container_width=True)