"""
Standalone FastAPI for the Custom LangGraph Agent.
Run locally: uvicorn run_langgraph_app:app --reload --port 8001
(Using port 8001 so it doesn't collide with her app on 8000)
"""
from __future__ import annotations
import uuid
from pathlib import Path
from typing import Literal
import os
os.environ.setdefault("OTEL_SERVICE_NAME", "open-deep-research")
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

from fastapi import FastAPI, Query

from agents.open_deep_research_agent import OpenDeepResearchAgent
from agents.base import FailureInjectionConfig

app = FastAPI(title="LangGraph Benchmark API", version="0.2.0")

ErrorType = Literal["loop", "timeout", "retrieval_fail", "hallucination", "context_overflow"]


def _langfuse_status() -> str:
    try:
        from langfuse import get_client
        return "connected" if get_client().auth_check() else "auth_failed"
    except ImportError:
        return "not_installed"
    except Exception:
        return "unreachable"


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "system": "open_deep_research",
        # Real signal instead of the old LANGCHAIN_TRACING_V2 env var, which
        # only ever indicated LangSmith tracing (a different product) and
        # said nothing about whether Langfuse was actually reachable.
        "langfuse": _langfuse_status(),
    }


@app.post("/run")
def run(
    n: int = Query(1, ge=1, le=200),
    faulty: bool = Query(False),
    error_type: ErrorType = Query("loop"),
    prob: float | None = Query(
        None,
        ge=0.0,
        le=1.0,
        description="Override probability/rate for retrieval_fail or hallucination. Ignored otherwise.",
    ),
) -> dict:
    default_retrieval_prob = 0.3
    default_hallucination_rate = 0.2

    cfg = FailureInjectionConfig(
        force_loop=(faulty and error_type == "loop"),
        force_timeout=(faulty and error_type == "timeout"),
        fail_retrieval_prob=(
            (prob if prob is not None else default_retrieval_prob)
            if (faulty and error_type == "retrieval_fail")
            else 0.0
        ),
        hallucination_rate=(
            (prob if prob is not None else default_hallucination_rate)
            if (faulty and error_type == "hallucination")
            else 0.0
        ),
        context_overflow=(faulty and error_type == "context_overflow"),
    )

    agent = OpenDeepResearchAgent(failure_config=cfg)
    pool = agent.TASK_POOL
    results = []
    for i in range(n):
        task = pool[i % len(pool)]
        r = agent.run(task)
        results.append({
            "run_id": str(uuid.uuid4()),
            "task": r.task,
            "success": r.success,
            "error": r.error,
            "duration_s": round(r.duration_s, 3),
            "retries": r.retries,
        })

    return {
        "batch_id": str(uuid.uuid4()),
        "system": "open_deep_research",
        "n": n,
        "faulty": faulty,
        "error_type": error_type if faulty else None,
        "successes": sum(1 for r in results if r["success"]),
        "failures": sum(1 for r in results if not r["success"]),
        "results": results,
    }