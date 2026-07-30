import streamlit as st
import json
import pandas as pd
import plotly.express as px
from evals.drift_detector import run_drift_check

st.set_page_config(page_title="Agent Drift Monitor", layout="wide")
st.title("🧠 AI Agent Workflow Quality Monitor")

TRACE_PATH = "data/crew_traces.jsonl"

result = run_drift_check(TRACE_PATH)

# --- Top metrics ---
col1, col2, col3 = st.columns(3)
col1.metric("Traces Analysed", result["n_traces"])
col2.metric("Latest EWMA Score", f"{result['latest_ewma']:.3f}",
            delta=f"Threshold: {result['threshold']}")
col3.metric("Drift Status",
            "⚠️ DRIFT" if result["drift_detected"] else "✅ Stable",
            delta_color="inverse")

# --- EWMA Line Chart ---
st.subheader("EWMA Score Over Time")
df = pd.DataFrame({
    "Trace Index": range(len(result["ewma_series"])),
    "Raw Score": result["scores"],
    "EWMA Score": result["ewma_series"],
})
fig = px.line(df, x="Trace Index", y=["Raw Score", "EWMA Score"],
              title="Agent Performance Drift (EWMA α=0.3)")
fig.add_hline(y=result["threshold"], line_dash="dash",
              line_color="red", annotation_text="Drift Threshold")
st.plotly_chart(fig, use_container_width=True)

# --- Failure Breakdown ---
st.subheader("Failure Category Breakdown")
with open(TRACE_PATH) as f:
    traces = [json.loads(l) for l in f if l.strip()]

failure_labels = [t.get("failure_type", "success" if t.get("success") else "unknown")
                  for t in traces]
label_counts = pd.Series(failure_labels).value_counts().reset_index()
label_counts.columns = ["Category", "Count"]
fig2 = px.pie(label_counts, names="Category", values="Count",
              title="Trace Outcome Distribution")
st.plotly_chart(fig2, use_container_width=True)