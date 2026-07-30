# STEPS.md — Route 1 Build Plan
> Graph AI Workflow Bottleneck · EWMA Drift Detection + CI Gate + Streamlit Dashboard

**Prerequisite:** Repo restructured per `PROJECT_CONTEXT.md` before starting any step.

---

## STEP 0 — Restructure the Repo (Do This First)

Move scattered files into the clean layout defined in `PROJECT_CONTEXT.md`.
No new code is written in this step — just `git mv` and verify imports still resolve.

```bash
git mv <old_path> <new_path>   # repeat for each file
python -m pytest evals/ -q     # smoke test after moves
git commit -m "chore: restructure repo to standard LLMOps layout"
```

---

## STEP 1 — Define the Composite Score

**File:** `evals/scorer.py`

Compute a per-trace score using only the two reliable fields.

```python
def score_trace(trace: dict) -> float:
    """
    Composite score [0.0 – 1.0] using success + latency only.
    Token/cost fields are excluded: known upstream instrumentation bug
    (context propagation breakdown in async CrewAI workers).
    Architecture is built to ingest them once patched.
    """
    success_score = 1.0 if trace.get("success") else 0.0

    raw_latency = trace.get("total_latency_ms", 0)
    # Clip at 120_000 ms (2 min) — treat anything beyond as worst-case
    MAX_LATENCY = 120_000
    latency_score = 1.0 - min(raw_latency / MAX_LATENCY, 1.0)

    return round(0.7 * success_score + 0.3 * latency_score, 4)
```

**Test it:**
```bash
python -c "from evals.scorer import score_trace; print(score_trace({'success': True, 'total_latency_ms': 5000}))"
# Expected: ~0.9875
```

---

## STEP 2 — Implement EWMA Drift Detection

**File:** `evals/drift_detector.py`

```python
import json
from pathlib import Path
from evals.scorer import score_trace

ALPHA = 0.3          # smoothing factor — higher = more reactive
WINDOW_SIZE = 10     # traces per batch
THRESHOLD = 0.60     # EWMA below this = drift alert


def load_traces(jsonl_path: str) -> list[dict]:
    with open(jsonl_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_ewma(scores: list[float], alpha: float = ALPHA) -> list[float]:
    ewma = []
    current = scores[0]
    for s in scores:
        current = alpha * s + (1 - alpha) * current
        ewma.append(round(current, 4))
    return ewma


def run_drift_check(jsonl_path: str) -> dict:
    traces = load_traces(jsonl_path)
    scores = [score_trace(t) for t in traces]
    ewma_series = compute_ewma(scores)

    latest_ewma = ewma_series[-1]
    drift_detected = latest_ewma < THRESHOLD

    return {
        "n_traces": len(traces),
        "latest_ewma": latest_ewma,
        "threshold": THRESHOLD,
        "drift_detected": drift_detected,
        "scores": scores,
        "ewma_series": ewma_series,
    }


if __name__ == "__main__":
    result = run_drift_check("data/crew_traces.jsonl")
    print(json.dumps(result, indent=2))
    if result["drift_detected"]:
        print(f"\n⚠️  DRIFT DETECTED — EWMA {result['latest_ewma']} < threshold {result['threshold']}")
        exit(1)
    else:
        print(f"\n✅  No drift — EWMA {result['latest_ewma']}")
        exit(0)
```

---

## STEP 2.5 — Fix the Schema Mismatch (Root Cause of the 0.3 Scores)

**What caused all scores to be `0.3`:**

`all_traces.jsonl` contains two merged schemas with different field locations.
The old scorer called `trace.get("success")` and `trace.get("total_latency_ms")` —
neither exists at the top level of either schema, so both defaulted to `False`/`0`,
producing the mathematically inevitable result: `0.7×0 + 0.3×1.0 = 0.3` for every trace.

**Confirmed schema differences:**

| Field | CrewAI location | OpenDeepResearch location |
|---|---|---|
| `success` | `trace["run_labels"]["success"]` | `trace["run_labels"]["success"]` ✅ same |
| `total_latency_ms` | `trace["meta"]["total_latency_ms"]` | `trace["total_latency_ms"]` (top-level) |
| `total_tokens` | `trace["meta"]["total_tokens"]` | `trace["total_tokens"]` (top-level) |
| `llm_model` | `trace["meta"]["llm_model"]` | `trace["llm_model"]` (top-level) |
| `faulty_batch` | `trace["meta"]["faulty_batch"]` | `trace["faulty_batch"]` (top-level) |
| `synthetic_error_type` | `trace["meta"]["synthetic_error_type"]` (string) | `trace["synthetic_error_types"]` (list) |
| `n_spans` | derived from `len(trace["spans"])` | `trace["n_spans"]` (top-level) |

**The fix: replace `evals/scorer.py` entirely with the version that includes `normalize_trace()`.**

The new `scorer.py` adds a `normalize_trace()` function that detects which schema a trace is using
and flattens both into one canonical dict before scoring. `score_trace()` now calls `normalize_trace()`
internally — no changes needed anywhere else.

Replace your `evals/scorer.py` with the `scorer.py` file delivered alongside this STEPS.md.

**Verify the fix:**
```bash
# Should print two non-0.3 scores — one for each schema
python evals/scorer.py

# Should now produce varied scores, not all 0.3
python evals/drift_detector.py
```

**Threshold is now auto-calibrated from your data** — no hardcoded `0.60` anymore.
`drift_detector.py` computes `threshold = mean(scores) − 1σ` at runtime using your actual
score distribution. This is the standard statistical process control approach.

> **Interview talking point:** "After merging two agent trace datasets, I discovered a schema
> mismatch was silently producing incorrect scores. I wrote a `normalize_trace()` adapter that
> detects the source schema and maps both to a canonical flat format before scoring — the same
> pattern used in production ETL pipelines to handle heterogeneous upstream data sources.
> I also replaced the hardcoded CI threshold with a data-driven `mean − 1σ` calibration."

---

## STEP 3 — GitHub Actions CI Gate

**File:** `.github/workflows/eval_gate.yml`

```yaml
name: Eval Quality Gate

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  eval-gate:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run EWMA Drift Check
        run: python evals/drift_detector.py
        # exits 1 if EWMA drops below threshold → PR blocked
```

**Verify locally before pushing:**
```bash
python evals/drift_detector.py   # should exit 0 on clean traces
```

---

## STEP 4 — Streamlit Dashboard

**File:** `dashboard/app.py`

```python
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
```

**Run locally:**
```bash
streamlit run dashboard/app.py
```

---

## STEP 5 — Update requirements.txt

Make sure these are present (add any missing):
```
streamlit
plotly
pandas
torch
torch-geometric
xgboost
langfuse
opentelemetry-sdk
```

---

## STEP 6 — Update README

Add these two sections to your existing README:

```markdown
## Known Limitations

Token/cost fields (`tokens_in`, `tokens_out`, `cost_usd`) are currently
zeroed across all traces. Root cause: **context propagation breakdown** —
when CrewAI spawns async sub-workers, the OpenTelemetry context baton is
dropped, so inner LLM callbacks never fire.

**Workaround (Route 1):** The composite score and EWMA drift detection
are built on `success` and `latency_ms`, which are accurately captured.
The pipeline architecture is designed to ingest cost/token data
transparently once the upstream instrumentation bug is patched.

## CI Quality Gate

Every PR runs `evals/drift_detector.py` via GitHub Actions.
The PR is blocked if the EWMA score of recent traces drops below 0.60.
```

---

## Completion Checklist

- [ ] STEP 0 — Repo restructured, imports pass
- [ ] STEP 1 — `evals/scorer.py` written + manually tested
- [ ] STEP 2 — `evals/drift_detector.py` runs without errors
- [ ] STEP 2.5 — `diagnose_traces.py` run, field names confirmed, `calibrate_threshold.py` run, `THRESHOLD` updated, scores are no longer all `0.3`
- [ ] STEP 3 — `.github/workflows/eval_gate.yml` pushed, Actions green
- [ ] STEP 4 — `dashboard/app.py` runs locally with charts
- [ ] STEP 5 — `requirements.txt` complete
- [ ] STEP 6 — README updated with Known Limitations + CI Gate sections

**Estimated time: 4–6 focused hours**