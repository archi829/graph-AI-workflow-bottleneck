# Setup Steps

1. Create and activate a virtual environment. -> `source venv/bin/activate`

2. Install each dependency **individually**.
   - Avoid using `requirements.txt` because dependency resolution hangs.

3. Download and install **Ollama** (`ollama.exe`).

4. Verify the installation:
   ```bash
   ollama --version
   ```

5. Start the Ollama server (run in **Command Prompt** and keep the window open):
   ```cmd
   set OLLAMA_HOST=0.0.0.0:11434
   ollama serve
   ```

6. Pull the model:
   ```bash
   ollama pull llama3.2:3b
   ```

7. Test the setup:
   ```bash
   python run_batch.py --system crewai --n 3
   ```

---

# Dataset Generation Pipeline

Run everything in the following order.

## Step 1: Generate traces

```bash
python run_batch.py --system crewai --n 10 --sleep 0
```

Output:
- Writes:
  ```
  data/raw/agent_system=crewai/batch_<timestamp>.jsonl
  ```
- Flushes pending traces to Langfuse.

---

## Step 2: Export traces from Langfuse

```bash
python export_traces.py --input data/raw/agent_system=crewai/batch_<timestamp>.jsonl
```

Output:
- Reads every `trace_id`
- Fetches the complete trace + spans from Langfuse
- Writes one JSON per trace:

```
data/raw/agent_system=crewai/<trace_id>.json
```

---

## Step 3: Build the final dataset

```bash
python build_dataset.py
```

Output:
- Reads every `*.json` inside `data/raw/`
- Computes slow/expensive labels using global percentiles
- Generates the dataset used for GNN training:

```
data/index.jsonl
```

---

# Generate the Full Dataset

## 70 clean traces

```bash
python run_batch.py --system crewai --n 70 --sleep 0
```

---

## 10 faulty traces — Loop motif

```bash
python run_batch.py --system crewai --n 10 --sleep 0 --faulty --error-type loop
```

---

## 10 faulty traces — Retrieval failure

```bash
python run_batch.py --system crewai --n 10 --sleep 0 --faulty --error-type retrieval_fail --prob 0.4
```

---

## 5 faulty traces — Timeout

```bash
python run_batch.py --system crewai --n 5 --sleep 0 --faulty --error-type timeout
```

---

## 5 faulty traces — Hallucination

```bash
python run_batch.py --system crewai --n 5 --sleep 0 --faulty --error-type hallucination --prob 0.4
```

---

# Run Jobs in Background (Recommended)

First create a logs directory:

```bash
mkdir logs
```

Run long jobs with `nohup` so they continue even after the terminal is closed:

```bash
nohup python run_batch.py --system crewai --n 70 --sleep 0 > logs/batch_clean.log 2>&1 &
```

Monitor progress:

```bash
tail -f logs/batch_clean.log
```