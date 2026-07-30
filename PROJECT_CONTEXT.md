# PROJECT_CONTEXT.md
> **Graph AI Workflow Bottleneck — LLMOps Observability + Regression-Gated Eval System**
> Keep this file at the root of the repo. It is the single source of truth for any AI assistant or new collaborator.

---

## What This Project Does

An end-to-end **LLMOps observability pipeline** for multi-agent AI workflows (CrewAI / LangGraph).

1. **Captures** execution traces from live agent runs using Langfuse + OpenTelemetry
2. **Converts** those traces into mathematical graphs (nodes = spans, edges = causal flow)
3. **Classifies** each trace using a **GNN (Graph Neural Network) + XGBoost** model into failure modes:
   - ✅ Success
   - 🔁 Infinite Loop
   - 🌀 Hallucination
   - 🔍 RAG Failure
   - ⏱️ Timeout
4. **Monitors** production quality over time using **EWMA (Exponentially Weighted Moving Average)** drift detection
5. **Enforces** a CI/CD quality gate via **GitHub Actions** — blocks PRs if agent performance degrades

---

## Current Status

| Component | Status | Notes |
|---|---|---|
| Trace collection | ✅ Done | 139 traces in `data/` |
| `latency_ms` logging | ✅ Accurate | Core signal for scoring |
| `success` boolean | ✅ Accurate | Core signal for scoring |
| GNN graph construction | ✅ Done | |
| XGBoost classifier | ✅ Done | Classifies failure modes |
| Token/cost fields | ❌ Zeroed | Known bug — see below |
| EWMA drift detection | 🔲 To build | STEP 2 in STEPS.md |
| GitHub Actions CI gate | 🔲 To build | STEP 3 in STEPS.md |
| Streamlit dashboard | 🔲 To build | STEP 4 in STEPS.md |

---

## Known Bug: Context Propagation Breakdown

**Affected fields:** `tokens_in`, `tokens_out`, `cost_usd`, `tool`, `model`

**Root cause:** CrewAI spawns async sub-workers under the hood. OpenTelemetry context propagation relies on a thread-local "baton" handoff. When the async worker is spawned without explicitly passing that context, the outer span still records wall-clock latency (hence `latency_ms` is correct), but the inner LLM callback never fires — so token counts and model metadata default to zero/null.

**Decision:** Route 1 Workaround — do not block on fixing this. The composite score and EWMA pipeline are built on the two reliable fields (`success`, `latency_ms`). The architecture is designed to absorb token/cost data transparently once the upstream bug is patched.

**Where the bug likely lives:** `telemetry/instrument.py` — missing `context.attach()` call before spawning async tasks.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Framework | CrewAI / LangGraph |
| Telemetry | Langfuse + OpenTelemetry |
| Graph ML | PyTorch Geometric (GNN) |
| Classical ML | XGBoost |
| Drift Detection | EWMA (custom, `evals/drift_detector.py`) |
| CI/CD | GitHub Actions |
| Dashboard | Streamlit + Plotly |
| Data Format | JSON (per-trace) + JSONL (`crew_traces.jsonl`) |

---

## Target Repo Structure

**Current state: files are scattered.** Reorganise to this layout before building Route 1.

```
graph-AI-workflow-bottleneck/
│
├── data/
│   ├── raw/                        # 139 individual trace JSON files
│   └── crew_traces.jsonl           # all traces concatenated line-by-line
│
├── telemetry/
│   └── instrument.py               # OpenTelemetry + Langfuse hooks (has the bug)
│
├── graph/
│   ├── builder.py                  # converts trace JSON → PyG graph object
│   └── features.py                 # node/edge feature extraction
│
├── models/
│   ├── gnn.py                      # GNN architecture (message passing layers)
│   ├── classifier.py               # XGBoost wrapper on top of GNN embeddings
│   └── checkpoints/                # saved model weights (gitignored if large)
│
├── evals/                          # ← NEW: everything Route 1 adds
│   ├── scorer.py                   # per-trace composite score (success + latency)
│   └── drift_detector.py           # EWMA calculation + threshold check + exit code
│
├── dashboard/                      # ← NEW
│   └── app.py                      # Streamlit UI
│
├── .github/
│   └── workflows/
│       └── eval_gate.yml           # ← NEW: CI gate workflow
│
├── notebooks/                      # exploratory work, not production code
│   └── eda.ipynb
│
├── requirements.txt
├── README.md
├── STEPS.md                        # build plan for Route 1
└── PROJECT_CONTEXT.md              # this file
```

### Files That Probably Need Moving
Identify which of your current scattered files maps to this structure and `git mv` them:

| If you currently have... | Move to |
|---|---|
| Any trace loader / parser script at root | `graph/builder.py` |
| Model training notebook or script | `models/` |
| Any scoring or eval script | `evals/` |
| Loose `.py` files at root | Appropriate subfolder above |
| `instrument.py` anywhere | `telemetry/instrument.py` |

---

## Data Schema (What a Trace Looks Like)

```json
{
  "trace_id": "abc123",
  "success": true,
  "total_latency_ms": 12400,
  "failure_type": null,
  "spans": [
    {
      "span_id": "span_001",
      "name": "Researcher._execute_core",
      "latency_ms": 8600,
      "tokens_in": 0,
      "tokens_out": 0,
      "cost_usd": 0.0,
      "tool": null,
      "model": ""
    }
  ]
}
```

**Reliable fields:** `success`, `total_latency_ms`, `spans[].latency_ms`
**Broken fields (zeroed):** `tokens_in`, `tokens_out`, `cost_usd`, `tool`, `model`

---

## Composite Score Formula (Route 1)

```
score = 0.7 × success_score + 0.3 × latency_score

where:
  success_score = 1.0 if success else 0.0
  latency_score = 1.0 - min(total_latency_ms / 120_000, 1.0)
```

Score range: 0.0 (worst) → 1.0 (perfect)
EWMA threshold: **0.60** — drop below this = drift alert = CI gate fails

---

## EWMA Parameters

| Parameter | Value | Rationale |
|---|---|---|
| Alpha (α) | 0.3 | Moderate smoothing — reacts to trends without noise overreaction |
| Window | All traces ordered by time | Rolling, not batched |
| Threshold | 0.60 | Conservative — 60% quality floor before blocking PRs |

To tune: increase α (→ 0.5) if you want faster alerts; decrease threshold (→ 0.50) if you want a looser gate.

---

## Interview Narrative (3-Act Structure)

**Act 1 — What the system does:**
"I built an LLMOps observability pipeline that captures traces from multi-agent AI workflows, converts them into execution graphs, and classifies failure modes — infinite loops, RAG failures, hallucinations, timeouts — using a GNN + XGBoost model."

**Act 2 — The bug I caught:**
"During analysis I noticed all token and cost fields were zeroed across 139 traces. I traced it to a context propagation breakdown: CrewAI's async workers were being spawned without passing the OpenTelemetry context baton, so the inner LLM callbacks never fired. Latency was captured fine because it's measured at the outer span level."

**Act 3 — What I built on top:**
"Rather than blocking on a third-party bug, I made a pragmatic call and built a regression-gated eval extension. I defined a composite quality score from the two reliable signals, applied EWMA drift detection — the same math used in production time-series anomaly detection — and wired it into a GitHub Actions CI gate that blocks PRs when agent performance degrades. The pipeline is architected to absorb token/cost data the moment the upstream bug is patched."

**Bullet for resume:**
> Built LLMOps eval pipeline with EWMA drift detection and GitHub Actions CI gate for AI agent observability; diagnosed OpenTelemetry context propagation bug in async CrewAI workers; stack: GNN · XGBoost · Streamlit · Python

---

## Quick Commands Reference

```bash
# Run scorer on a single trace
python -c "from evals.scorer import score_trace; import json; print(score_trace(json.load(open('data/raw/trace_001.json'))))"

# Run drift check on all traces
python evals/drift_detector.py

# Launch dashboard
streamlit run dashboard/app.py

# Run CI gate locally (same as GitHub Actions)
python evals/drift_detector.py && echo "Gate: PASS" || echo "Gate: FAIL"
```
