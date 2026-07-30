# Agentic Pipeline Benchmarking – Project Progress

## Project Goal

Build a unified execution-trace dataset from multiple agent frameworks and train Graph Neural Networks (GAT, RGCN, HeteroGAT) to identify execution failures, bottlenecks, and inefficient workflows.

---

# Overall Pipeline

```text
Agent Run
    ↓
run_batch.py
    ↓
batch_xxx.jsonl
    ↓
export_traces.py
    ↓
Individual trace.json files
    ↓
crew_traces.jsonl
open_deep_research_traces.jsonl
    ↓
all_traces.jsonl
    ↓
Graph Construction
    ↓
PyTorch Geometric Dataset
    ↓
Baseline ML
    ↓
Graph Neural Networks
    ↓
Evaluation & Comparison
```

---

# Project Structure

## 1. run_batch.py

### Purpose

Executes batches of tasks across supported agent frameworks.

### Supported Frameworks

- CrewAI
- Open Deep Research
- FinRobot

### Output

```text
batch_xxx.jsonl
```

Each record contains:

- Task
- Trace ID
- Success/Failure
- Execution Time
- Retry Count
- Synthetic Error Type

---

## 2. export_traces.py

### Purpose

Retrieves execution telemetry from Langfuse and converts it into a unified project schema.

### Output

```text
trace_id.json
```

Each exported trace contains:

- Spans
- Parent-child relationships
- Latency
- Token usage
- Cost
- Metadata
- Labels

---

## 3. build_dataset.py

### Purpose

Validates exported traces and prepares dataset metadata.

Computes labels such as:

- Slow execution
- Expensive execution

Produces

```text
index.jsonl
```

which serves as a lightweight manifest containing:

- Trace path
- Labels
- Metadata

> Note: `index.jsonl` is **not** directly used for GNN training.

---

# Dataset Preparation

Individual traces were merged into:

```text
crew_traces.jsonl
```

and

```text
open_deep_research_traces.jsonl
```

These datasets were shuffled and merged into a single unified dataset:

```text
all_traces.jsonl
```

Each line represents one complete execution trace.

---

# Current Progress

Completed pipeline:

```text
Run agents
      ↓
run_batch.py
      ↓
batch_xxx.jsonl
      ↓
export_traces.py
      ↓
Individual trace.json files
      ↓
crew_traces.jsonl
open_deep_research_traces.jsonl
      ↓
all_traces.jsonl
```

## Completed Tasks

- Agent execution completed
- Trace export completed
- Unified JSON schema finalized
- Individual trace files generated
- Framework datasets merged
- Final `all_traces.jsonl` created

---

# Current Stage

**Machine Learning Preprocessing**

The project is now transitioning from data collection to graph construction for Graph Neural Network training.

---

# Next Steps

## Step 1 — Load Dataset

Load the unified dataset:

```python
all_traces.jsonl
```

Iterate through every trace:

```python
for trace in traces:
    ...
```

---

## Step 2 — Create Graph Nodes

For every trace:

- Read all spans
- Convert each span into a graph node

```text
1 Span = 1 Node
```

---

## Step 3 — Build Graph Edges

Construct graph connectivity using parent-child relationships.

```text
parent_id
      ↓
edge_index
```

---

## Step 4 — Generate Node Features

Extract useful attributes for every node.

Example features:

- Role
- Latency (ms)
- Input Tokens
- Output Tokens
- Cost (USD)
- Error Flag
- Tool Used
- Model Used

---

## Step 5 — Create Graph Labels

Choose prediction targets.

Initial target:

```text
run_labels.success
```

Possible future targets:

- Slow execution
- Expensive execution
- Error type
- Retry prediction

---

## Step 6 — Build PyTorch Geometric Dataset

Convert every execution trace into a

```python
torch_geometric.data.Data
```

object.

Each graph should contain:

- Nodes
- Edge Index
- Node Features
- Labels

---

## Step 7 — Model Training

### Baseline

- Random Forest

### Graph Neural Networks

- Graph Attention Network (GAT)
- Relational Graph Convolution Network (RGCN)
- Heterogeneous Graph Attention Network (HeteroGAT)

---

## Step 8 — Evaluation

Compare models using metrics such as:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC (if applicable)

Analyze:

- Failure prediction
- Performance bottlenecks
- Graph representations
- Model efficiency

---

# Planned Notebook Structure

```text
1. Load all_traces.jsonl

2. Dataset Exploration

3. Feature Engineering

4. Graph Construction

5. PyTorch Geometric Dataset

6. Train / Validation / Test Split

7. Random Forest Baseline

8. Graph Attention Network (GAT)

9. Relational GCN (RGCN)

10. Heterogeneous GAT (HeteroGAT)

11. Model Evaluation

12. Visualizations

13. Results & Discussion
```

---

# Overall Progress

| Phase | Status |
|--------|--------|
| Agent Execution | ✅ Completed |
| Trace Collection | ✅ Completed |
| Trace Export | ✅ Completed |
| Schema Design | ✅ Completed |
| Dataset Merging | ✅ Completed |
| Graph Construction | ⏳ In Progress |
| Feature Engineering | ⏳ Pending |
| Baseline ML | ⏳ Pending |
| GNN Training | ⏳ Pending |
| Evaluation | ⏳ Pending |

---

## Estimated Completion

**Overall Progress:** **~65% Complete**

The data collection and preparation pipeline has been successfully completed. The remaining work focuses on graph construction, feature engineering, Graph Neural Network training, and comparative evaluation.
