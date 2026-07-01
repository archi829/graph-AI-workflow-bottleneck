"""
FastAPI wrapper -- Person A's `app` container entrypoint (Section 6).

    POST /run?system=crewai&n=30&faulty=false

Keeps Person B's Langfuse export agnostic to which system ran (Section 6):
B's exporter just watches Langfuse, it never has to call into this app.

Run locally:
    uvicorn app:app --reload --port 8000
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query

from agents import REGISTRY
from telemetry.instrument import instrument_crewai

app = FastAPI(title="agentic-pipeline-benchmarking", version="0.1.0")


@app.on_event("startup")
def _startup() -> None:
    # Instrument whatever's registered. Cheap to call even if a given
    # request never touches crewai.
    instrument_crewai()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "systems": sorted(REGISTRY.keys())}


@app.post("/run")
def run(
    system: str = Query(..., description="one of: " + ", ".join(sorted(REGISTRY.keys()))),
    n: int = Query(1, ge=1, le=200),
    faulty: bool = Query(False),
    error_type: str = Query("loop", description="loop|timeout|retrieval_fail|hallucination|context_overflow"),
) -> dict:
    if system not in REGISTRY:
        raise HTTPException(400, f"unknown system '{system}'. Known: {sorted(REGISTRY.keys())}")

    import os

    for flag in ("FORCE_LOOP", "FORCE_TIMEOUT", "FAIL_RETRIEVAL_PROB", "HALLUCINATION_RATE", "CONTEXT_OVERFLOW"):
        os.environ.pop(flag, None)
    if faulty:
        flag_map = {
            "loop": ("FORCE_LOOP", "true"),
            "timeout": ("FORCE_TIMEOUT", "true"),
            "retrieval_fail": ("FAIL_RETRIEVAL_PROB", "0.3"),
            "hallucination": ("HALLUCINATION_RATE", "0.2"),
            "context_overflow": ("CONTEXT_OVERFLOW", "true"),
        }
        env_var, val = flag_map.get(error_type, ("FORCE_LOOP", "true"))
        os.environ[env_var] = val

    agent_cls = REGISTRY[system]
    agent = agent_cls()
    pool = getattr(agent, "TASK_POOL", [f"smoke test task {i}" for i in range(n)])

    batch_id = str(uuid.uuid4())
    out_dir = Path(f"data/raw/agent_system={system}")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i in range(n):
        task = pool[i % len(pool)]
        r = agent.run(task)
        results.append(
            {
                "run_id": str(uuid.uuid4()),
                "task": r.task,
                "success": r.success,
                "error": r.error,
                "synthetic_error_type": r.synthetic_error_type,
                "duration_s": round(r.duration_s, 3),
                "tokens_used": r.tokens_used,
                "retries": r.retries,
                "structured_output": r.structured_output,
            }
        )

    return {
        "batch_id": batch_id,
        "system": system,
        "n": n,
        "faulty": faulty,
        "successes": sum(1 for r in results if r["success"]),
        "failures": sum(1 for r in results if not r["success"]),
        "results": results,
    }
