"""
Offline mode -- genuinely zero model API, zero network calls, zero cost.

This bypasses CrewAI's Agent/Task/Crew/LLM machinery entirely. It is NOT
"CrewAI with a fake LLM" -- it's a separate deterministic code path that
answers using only the local_knowledge tool (agents/tools.py) and basic
templating. Use it to:
  - exercise run_batch.py / app.py / the trace-writing path with zero setup
  - get a deterministic baseline to diff real-LLM output against
  - develop without an internet connection or any account at all

It deliberately does NOT call web_search, since that mode is meant to be
provably offline (DuckDuckGo still requires network egress).

Output is shaped to match agents/schemas.py::FinalAnswer so downstream code
(run_batch.py, app.py, evaluation.py) doesn't need to special-case it.
"""

from __future__ import annotations

from .schemas import FinalAnswer
from .tools import _KNOWLEDGE_BASE, calculator


def _lookup_all(*topics: str) -> list[str]:
    facts = []
    for topic in topics:
        key = topic.strip().lower()
        for k, v in _KNOWLEDGE_BASE.items():
            if k in key or key in k:
                facts.append(v)
                break
        else:
            facts.append(f"(no local knowledge entry for '{topic}')")
    return facts


def _run_product_compare(task: str) -> FinalAnswer:
    # crude entity extraction: pull known product keys mentioned in the task
    mentioned = [k for k in _KNOWLEDGE_BASE if k in task.lower()]
    facts = _lookup_all(*mentioned) if mentioned else ["No matching products found in local knowledge base."]
    return FinalAnswer(
        summary=f"Offline comparison based on {len(facts)} local knowledge entries (no LLM used).",
        details="\n".join(f"- {f}" for f in facts),
        recommendation=(
            "Run in Ollama or cloud mode for a real reasoned recommendation; "
            "offline mode only retrieves raw specs."
        ),
        sources_used=["local_knowledge (offline mode)"],
    )


def _run_trip_plan(task: str) -> FinalAnswer:
    mentioned = [k for k in _KNOWLEDGE_BASE if k in task.lower()]
    facts = _lookup_all(*mentioned) if mentioned else ["No matching destination found in local knowledge base."]
    # deterministic "budget" calc so calculator gets exercised even offline
    budget_note = calculator.func("3 * 1500")
    return FinalAnswer(
        summary="Offline trip outline based on local knowledge (no LLM used).",
        details="\n".join(f"- {f}" for f in facts)
        + f"\n- Rough 3-day placeholder budget estimate (INR, per person): {budget_note}",
        recommendation="Run in Ollama or cloud mode for a real day-by-day itinerary.",
        sources_used=["local_knowledge (offline mode)", "calculator"],
    )


def _run_bug_explain(task: str) -> FinalAnswer:
    mentioned = [k for k in _KNOWLEDGE_BASE if k in task.lower()]
    facts = _lookup_all(*mentioned) if mentioned else ["No matching bug pattern found in local knowledge base."]
    return FinalAnswer(
        summary="Offline bug explanation based on local knowledge (no LLM used).",
        details="\n".join(f"- {f}" for f in facts),
        recommendation="Run in Ollama or cloud mode for a real, code-specific fix.",
        sources_used=["local_knowledge (offline mode)"],
    )


_DISPATCH = {
    "product_compare": _run_product_compare,
    "trip_plan": _run_trip_plan,
    "bug_explain": _run_bug_explain,
}


def run_offline(task: str, task_type: str) -> FinalAnswer:
    handler = _DISPATCH.get(task_type, _run_product_compare)
    return handler(task)
