"""
evals/drift_detector.py
EWMA drift detection over all_traces.jsonl.

Exits 0  → no drift (CI gate passes)
Exits 1  → drift detected (CI gate blocks PR)
"""

import json
import statistics
import sys
from scorer import normalize_trace, score_trace

# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------
ALPHA = 0.3          # EWMA smoothing — higher = reacts faster to recent traces
TRACE_PATH = "data/all_traces.jsonl"

# THRESHOLD is set dynamically via calibrate_threshold() below.
# Hardcode a fallback only if running without prior calibration.
FALLBACK_THRESHOLD = 0.50


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def load_traces(jsonl_path: str) -> list[dict]:
    with open(jsonl_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_ewma(scores: list[float], alpha: float = ALPHA) -> list[float]:
    """Exponentially Weighted Moving Average over an ordered score list."""
    ewma = []
    current = scores[0]
    for s in scores:
        current = alpha * s + (1 - alpha) * current
        ewma.append(round(current, 4))
    return ewma


def calibrate_threshold(scores: list[float]) -> float:
    """
    Data-driven threshold: mean minus one standard deviation.
    This is the standard statistical process control approach —
    flag runs that fall more than 1 std below the baseline mean.
    """
    mean = statistics.mean(scores)
    stdev = statistics.stdev(scores) if len(scores) > 1 else 0.0
    return round(mean - stdev, 4)


def run_drift_check(jsonl_path: str = TRACE_PATH) -> dict:
    traces = load_traces(jsonl_path)

    # Score every trace (normalisation handled inside score_trace)
    scores = [score_trace(t) for t in traces]
    ewma_series = compute_ewma(scores)

    threshold = calibrate_threshold(scores)
    latest_ewma = ewma_series[-1]
    drift_detected = latest_ewma < threshold

    # Per-agent breakdown for dashboard
    agent_scores: dict[str, list[float]] = {}
    for trace, score in zip(traces, scores):
        norm = normalize_trace(trace)
        agent = norm.get("agent_system", "unknown")
        agent_scores.setdefault(agent, []).append(score)

    agent_summary = {
        agent: {
            "n": len(s),
            "mean_score": round(statistics.mean(s), 4),
            "success_rate": round(
                sum(1 for t in traces
                    if normalize_trace(t).get("agent_system") == agent
                    and normalize_trace(t).get("success")) / len(s), 4
            ),
        }
        for agent, s in agent_scores.items()
    }

    return {
        "n_traces":       len(traces),
        "threshold":      threshold,
        "latest_ewma":    latest_ewma,
        "drift_detected": drift_detected,
        "scores":         scores,
        "ewma_series":    ewma_series,
        "agent_summary":  agent_summary,
    }


# ---------------------------------------------------------------------------
# CLI entry point — called directly by GitHub Actions
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    result = run_drift_check()

    # Print summary (suppressed in CI to keep logs clean; full JSON only on failure)
    print(f"Traces loaded : {result['n_traces']}")
    print(f"Threshold     : {result['threshold']}  (mean - 1σ, calibrated from data)")
    print(f"Latest EWMA   : {result['latest_ewma']}")
    print(f"\nAgent breakdown:")
    for agent, summary in result["agent_summary"].items():
        print(f"  {agent}: n={summary['n']}  mean_score={summary['mean_score']}  success_rate={summary['success_rate']}")

    if result["drift_detected"]:
        print(f"\n⚠️  DRIFT DETECTED — EWMA {result['latest_ewma']} < threshold {result['threshold']}")
        print(json.dumps(result, indent=2))
        sys.exit(1)
    else:
        print(f"\n✅  No drift — EWMA {result['latest_ewma']} ≥ threshold {result['threshold']}")
        sys.exit(0)