"""
evals/scorer.py
Per-trace composite score + schema normalizer for all_traces.jsonl.

Handles two distinct schemas that were merged into all_traces.jsonl:
  - CrewAI:             success/latency/tokens live inside nested run_labels + meta
  - OpenDeepResearch:   success lives in run_labels; latency/tokens are top-level
"""


def normalize_trace(trace: dict) -> dict:
    """
    Flattens both trace schemas into a single canonical dict.

    CrewAI layout:
        trace["run_labels"]["success"]
        trace["meta"]["total_latency_ms"]
        trace["meta"]["total_tokens"]
        trace["meta"]["llm_model"]
        trace["meta"]["faulty_batch"]
        trace["meta"]["synthetic_error_type"]   ← singular string

    OpenDeepResearch layout:
        trace["run_labels"]["success"]           ← same
        trace["total_latency_ms"]               ← top-level
        trace["total_tokens"]                   ← top-level
        trace["llm_model"]                      ← top-level
        trace["faulty_batch"]                   ← top-level
        trace["synthetic_error_types"]          ← plural list
        trace["has_error_span"]                 ← ODR only
        trace["n_spans"]                        ← ODR only
    """
    run_labels = trace.get("run_labels", {})
    meta = trace.get("meta", {})  # present in CrewAI only; empty dict for ODR

    # --- success: same location in both schemas ---
    success = bool(run_labels.get("success", False))

    # --- latency: top-level in ODR, nested in CrewAI meta ---
    if "total_latency_ms" in trace:
        total_latency_ms = trace["total_latency_ms"] or 0      # ODR
    else:
        total_latency_ms = meta.get("total_latency_ms", 0)     # CrewAI

    # --- tokens: top-level in ODR, nested in CrewAI meta ---
    if "total_tokens" in trace:
        total_tokens = trace["total_tokens"] or 0               # ODR
    else:
        total_tokens = meta.get("total_tokens", 0)              # CrewAI

    # --- llm_model: top-level in ODR, nested in CrewAI meta ---
    llm_model = trace.get("llm_model") or meta.get("llm_model", "unknown")

    # --- faulty_batch: top-level in ODR, nested in CrewAI meta ---
    faulty_batch = (
        trace.get("faulty_batch")
        if "faulty_batch" in trace
        else meta.get("faulty_batch", False)
    )

    # --- error types: plural list in ODR, singular string in CrewAI meta ---
    if "synthetic_error_types" in trace:
        error_types = trace["synthetic_error_types"] or []      # ODR — already a list
    elif meta.get("synthetic_error_type"):
        error_types = [meta["synthetic_error_type"]]            # CrewAI — wrap in list
    else:
        error_types = []

    # --- span count: explicit in ODR, derive from spans list in CrewAI ---
    n_spans = (
        trace["n_spans"]
        if "n_spans" in trace
        else len(trace.get("spans", []))
    )

    return {
        # identifiers (both schemas)
        "trace_id":         trace.get("trace_id"),
        "agent_system":     trace.get("agent_system"),
        "task":             trace.get("task"),
        "run_id":           trace.get("run_id"),
        # normalised signals
        "success":          success,
        "total_latency_ms": total_latency_ms,
        "total_tokens":     total_tokens,
        "llm_model":        llm_model,
        "faulty_batch":     faulty_batch,
        "error_types":      error_types,
        "n_spans":          n_spans,
        # ODR-only fields (False/None for CrewAI traces)
        "has_error_span":   trace.get("has_error_span", False),
        # convenience flags from run_labels
        "slow":             bool(run_labels.get("slow", False)),
        "expensive":        bool(run_labels.get("expensive", False)),
    }


def score_trace(trace: dict) -> float:
    """
    Composite score [0.0 – 1.0] from a normalised trace.

    Formula:
        score = 0.7 × success_score + 0.3 × latency_score

    Token/cost fields are excluded from scoring:
        Known upstream bug — context propagation breakdown in async
        CrewAI workers causes tokens_in/out and cost to zero out.
        The architecture is designed to ingest them once patched.

    Args:
        trace: raw trace dict (either schema) — normalisation is applied internally.

    Returns:
        float in [0.0, 1.0], rounded to 4 decimal places.
    """
    norm = normalize_trace(trace)

    success_score = 1.0 if norm["success"] else 0.0

    # Cap at 120 s — treat anything longer as worst-case latency
    MAX_LATENCY_MS = 120_000
    latency_score = 1.0 - min(norm["total_latency_ms"] / MAX_LATENCY_MS, 1.0)

    return round(0.7 * success_score + 0.3 * latency_score, 4)


# ---------------------------------------------------------------------------
# Quick sanity check — run directly to verify normalisation is working
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    CREW_SAMPLE = {
        "trace_id": "crew_001",
        "agent_system": "crewai",
        "run_labels": {"success": True, "slow": False, "expensive": False},
        "meta": {"total_latency_ms": 12400, "total_tokens": 850,
                 "llm_model": "gpt-4o", "faulty_batch": False,
                 "synthetic_error_type": None, "retries": 0},
        "spans": [],
    }

    ODR_SAMPLE = {
        "trace_id": "odr_001",
        "agent_system": "open_deep_research",
        "run_labels": {"success": False, "slow": True, "expensive": False},
        "total_latency_ms": 86291,
        "total_tokens": 0,
        "llm_model": "",
        "faulty_batch": False,
        "has_error_span": True,
        "synthetic_error_types": ["timeout"],
        "n_spans": 4,
    }

    for sample in [CREW_SAMPLE, ODR_SAMPLE]:
        norm = normalize_trace(sample)
        score = score_trace(sample)
        print(f"\n[{norm['agent_system']}] trace_id={norm['trace_id']}")
        print(f"  success={norm['success']}  latency={norm['total_latency_ms']}ms  tokens={norm['total_tokens']}")
        print(f"  score → {score}")