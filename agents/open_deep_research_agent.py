"""
Custom Deep Research Agent with Tree-Structured LangGraph + Dynamic Send API.
LangGraph workflow: Planner -> Dynamic Researchers (Send) -> Merger -> Writer

Branch width is DRIVEN BY THE PLANNER LLM, not hardcoded or randomized:
- Narrow/simple tasks produce 1-2 subtopics
- Broad/comparative/multi-part tasks produce 3-5 subtopics
- Each subtopic gets its own dynamically-spawned researcher via LangGraph's Send

                    PLANNER --Send--> researcher[0] - tool[0] --+
                                      researcher[1] - tool[1] --|
                                      researcher[2] - tool[2] --+-- MERGER -- WRITER
                                      ... (1-5 dynamic spawns) --+
                                           ^ (force_loop only, re-runs planner)
                                           +-----------------------+

SPAN-LEVEL TRACING (for GNN bottleneck detection):
- Every node wrapped in `langfuse.start_as_current_observation(as_type="span", ...)`
  (NOT `langfuse.span(...)` -- that method does not exist on the v3 client and
  was the reason every run was crashing instantly inside planner_node.)
- Tools wrapped the same way, name="tool_call:*", producing `role=tool` spans
- Token usage extracted from LLM responses, propagated to parent spans AND to
  the local RunResult (tokens_used), so it's visible in both Langfuse and your
  local batch JSONL.
- Cost calculated from Groq's published pricing

FIXES APPLIED IN THIS VERSION (vs. the previous draft):
1. langfuse.span(...) -> langfuse.start_as_current_observation(as_type="span", ...)
   everywhere (planner, researcher, merger, writer, tool call, timeout/retrieval
   failure injection). This was a hard crash on every single run.
2. context_overflow multiplier reduced 4000 -> 600 repeats, to stay under Groq's
   12,000 TPM limit so the run actually completes instead of an immediate 413.
3. force_loop_n range changed (1,3) -> (2,4). At force_loop_n=1 the old range
   let `loop_count(1) < force_n(1)` be False on the very first pass, so ~1/3 of
   "loop" traces did zero extra loops and were indistinguishable from clean
   traces. Minimum is now 2, guaranteeing at least one real extra cycle.
4. tokens_used is now actually accumulated across every LLM call in a run and
   set on the RunResult -- previously always null in the local JSONL despite
   being calculated and sent to Langfuse.
5. trace_id capture now tries span.trace_id first, falls back to
   langfuse.get_current_trace_id() if that attribute isn't present on your
   installed SDK version, instead of silently going to None.
6. Removed the unused HARD_LOOP_CAP constant (was never referenced anywhere).
7. force_timeout / fail_retrieval injection moved INSIDE the graph. Previously
   both raised in _run_task BEFORE _invoke_graph() ever ran, producing a
   trace with a single flat span and zero graph structure (n_spans=1,
   total_tokens=0) -- trivially separable by span count alone, which
   undermines "graph structure matters" as a GNN training signal. Now the
   failure is injected into exactly one dynamically-spawned researcher
   (chosen after the planner runs, once real subtopics exist), so the
   resulting trace still has a real planner_node span, a real partial
   researcher_node span, and an ERROR-flagged tool_call span before the
   exception propagates -- a genuine partial-graph failure signature instead
   of a degenerate one-node trace.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import random
import operator
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Annotated

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from typing import TypedDict

from .base import AgentSystem, RunResult

logger = logging.getLogger(__name__)

# --- Langfuse (optional import) ---
try:
    from langfuse import get_client
    # from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
    from langfuse import propagate_attributes
    _LANGFUSE_AVAILABLE = True
except ImportError:
    _LANGFUSE_AVAILABLE = False
    logger.warning("langfuse not installed -- runs will NOT be traced. `pip install langfuse`.")

# ---------------------------------------------------------------------------
# Groq pricing
# ---------------------------------------------------------------------------
_GROQ_PRICING: dict[str, dict[str, float]] = {
    "llama-3.3-70b-versatile": {"input": 0.59 / 1_000_000, "output": 0.79 / 1_000_000},
    "llama-3.1-8b-instant":    {"input": 0.05 / 1_000_000, "output": 0.08 / 1_000_000},
    "llama-3.2-3b-preview":    {"input": 0.03 / 1_000_000, "output": 0.06 / 1_000_000},
    "mixtral-8x7b-32768":      {"input": 0.24 / 1_000_000, "output": 0.24 / 1_000_000},
    "gemma2-9b-it":            {"input": 0.08 / 1_000_000, "output": 0.08 / 1_000_000},
}


def _calculate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    pricing = _GROQ_PRICING.get(model, _GROQ_PRICING["llama-3.3-70b-versatile"])
    return round(tokens_in * pricing["input"] + tokens_out * pricing["output"], 6)


def _get_token_usage(response: AIMessage) -> dict:
    um = getattr(response, "usage_metadata", None)
    if um:
        return {
            "input": um.get("input_tokens", 0) or 0,
            "output": um.get("output_tokens", 0) or 0,
            "total": um.get("total_tokens", 0) or 0,
        }
    rm = response.response_metadata or {}
    tu = rm.get("token_usage", {}) or {}
    return {
        "input": tu.get("prompt_tokens", 0) or 0,
        "output": tu.get("completion_tokens", 0) or 0,
        "total": tu.get("total_tokens", 0) or 0,
    }


def _get_model_name(response: AIMessage) -> str:
    rm = response.response_metadata or {}
    return rm.get("model_name", "") or ""


def _get_response_token_info(response: AIMessage) -> dict:
    token_info = _get_token_usage(response)
    model = _get_model_name(response)
    cost = _calculate_cost(model, token_info["input"], token_info["output"])
    return {
        "tokens_in": token_info["input"],
        "tokens_out": token_info["output"],
        "total_tokens": token_info["total"],
        "model": model,
        "cost_usd": cost,
    }


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
_last_llm_call_time: float = 0.0
_LLM_MIN_INTERVAL_S = 6.0
_llm_call_lock = None  # set lazily to a threading.Lock() below


def _get_llm_lock():
    global _llm_call_lock
    if _llm_call_lock is None:
        import threading
        _llm_call_lock = threading.Lock()
    return _llm_call_lock


def rate_limited_llm_invoke(llm: ChatGroq, messages: list, config=None):
    global _last_llm_call_time
    lock = _get_llm_lock()
    with lock:
        elapsed = time.monotonic() - _last_llm_call_time
        if elapsed < _LLM_MIN_INTERVAL_S:
            sleep_s = _LLM_MIN_INTERVAL_S - elapsed
            logger.debug(f"Rate limit: sleeping {sleep_s:.1f}s")
            time.sleep(sleep_s)
        _last_llm_call_time = time.monotonic()
    result = llm.invoke(messages, config=config)
    return result


def _get_langfuse_client():
    if not _LANGFUSE_AVAILABLE:
        return None
    try:
        return get_client()
    except Exception:
        return None


def _make_span(langfuse, name: str, input=None, metadata: Optional[dict] = None):
    """Single helper for creating a span, so there's exactly ONE place in the
    file that calls the Langfuse span-creation API. If this needs to change
    again for a future SDK version, only this function needs editing."""
    if langfuse is None:
        return nullcontext()
    return langfuse.start_as_current_observation(
        as_type="span",
        name=name,
        input=input,
        metadata=metadata or {},
    )


def _env(*names: str, default: str | None = None) -> str | None:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return default


TASK_POOL = [
    "What are the key technical bottlenecks in scaling large language models beyond 100B parameters?",
    "Analyze the competitive landscape of vector databases in 2025 for production RAG systems.",
    "What caused the 2023 Silicon Valley Bank collapse and what systemic risks remain in US banking?",
    "Compare LangGraph vs AutoGen vs CrewAI for building production multi-agent AI systems.",
    "What are the most promising approaches to AI alignment and what are their key limitations?",
    "Analyze the impact of US semiconductor export restrictions on India's AI industry development.",
    "What are the key differences between RLHF, DPO, and RLAIF for aligning language models?",
    "Investigate the current state of AI regulation globally and its impact on open source LLMs.",
]


# ===========================================================================
# State with dynamic researcher_outputs accumulator
# ===========================================================================
class ResearchState(TypedDict):
    task: str

    # Planner output
    subtopics: list[str]
    planned: bool

    # Dynamic accumulator: one entry per Send-spawned researcher.
    # NOTE: this uses operator.add, so it APPENDS across loop cycles rather
    # than resetting. On a force_loop run, loop 2's merge will combine loop
    # 1's + loop 2's findings together (context genuinely grows each cycle).
    # This is left as-is intentionally -- it's a realistic "context balloons
    # under repeated looping" bottleneck signature, not a bug. If you want
    # each loop to be a clean independent repeat instead, this needs an
    # explicit reset mechanism (a sentinel value + custom reducer), which is
    # a bigger change -- ask if you want that version.
    researcher_outputs: Annotated[list[str], operator.add]

    # Merger / writer
    merged_context: str

    # Failure injection (input flags, decided once at run start)
    force_loop: bool
    force_loop_n: int
    synthetic_error_type: str | None
    force_timeout: bool
    fail_retrieval: bool

    # Tracking
    loop_count: int
    history: list[str]
    final_context: str

    # Token accounting (accumulated across every LLM call in this run).
    # Must be Annotated with operator.add, NOT plain int -- multiple
    # Send-spawned researcher_node instances write to this key in the same
    # step, and a plain int field raises INVALID_CONCURRENT_GRAPH_UPDATE
    # since LangGraph doesn't know how to reconcile two parallel writes to
    # one key without a reducer. Every node below now returns its OWN
    # increment (not a running total it read-modified-added itself), and
    # LangGraph sums all increments across the step via this reducer.
    total_tokens_in: Annotated[int, operator.add]
    total_tokens_out: Annotated[int, operator.add]


# ===========================================================================
# Tool: web search (shared, instrumented)
# ===========================================================================
def run_tool_web_search(query: str, tool_name: str = "web_search", langfuse_client=None) -> str:
    """Run a web search wrapped in a Langfuse tool span."""
    with _make_span(
        langfuse_client,
        name=f"tool_call:{tool_name}",
        input=query,
        metadata={"tool": tool_name},
    ) as span:
        t0 = time.monotonic()
        error = False
        exc_msg = None
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=4))
            output = "\n".join(f"- {r['title']}: {r['body']}" for r in results) if results else "No results found."
        except Exception as exc:
            output = f"Search failed: {exc}"
            error = True
            exc_msg = str(exc)

        latency_ms = round((time.monotonic() - t0) * 1000, 2)
        if span is not None:
            span.update(
                output=output,
                level="ERROR" if error else "DEFAULT",
                status_message=exc_msg,
                metadata={"latency_ms": latency_ms, "tool": tool_name, "error_flag": error},
            )
        return output


def get_llm():
    api_key = _env("GROQ_API_KEY", "LLM_API_KEY")
    model = _env("ODR_MODEL", "LLM_MODEL") or "llama-3.3-70b-versatile"
    if not api_key:
        raise EnvironmentError("Requires GROQ_API_KEY in .env")
    return ChatGroq(model=model, api_key=api_key, temperature=0.2)


# ===========================================================================
# NODE: Planner -- splits task into sub-research topics
# ===========================================================================
def planner_node(state: ResearchState, config: Optional[RunnableConfig] = None) -> dict:
    """Planner node: breaks the task into subtopics using LLM judgment.

    The LLM decides how many subtopics are needed based on task complexity:
    - 1-2 for narrow/simple tasks
    - 3-5 for broad/comparative/multi-part tasks

    Outputs subtopics which are then fanned out via Send.
    """
    langfuse = _get_langfuse_client()
    with _make_span(
        langfuse,
        name="planner_node",
        input=f"TASK: {state['task'][:300]}",
        metadata={"node_type": "planner"},
    ) as span:
        llm = get_llm()
        prompt = SystemMessage(content=(
            "You are an expert research planner. Your job is to break the given "
            "research task into specific, concrete sub-topics that can be "
            "researched independently and in parallel.\n\n"
            "Guidelines:\n"
            "- For narrow/simple tasks (single fact, definition, basic question): "
            "output 1-2 subtopics.\n"
            "- For broad/comparative/multi-part tasks (comparisons, multi-faceted "
            "analysis, complex topics): output 3-5 subtopics.\n\n"
            "Output EXACTLY:\n"
            "SUBTOPICS:\n"
            "- <subtopic 1>\n"
            "- <subtopic 2>\n"
            "- <subtopic 3>\n"
            "...\n\n"
            "Each subtopic must be a specific search query or research question, "
            "not a generic category."
        ))
        response = rate_limited_llm_invoke(
            llm,
            [prompt, HumanMessage(content=f"TASK: {state['task']}")],
            config=config,
        )
        info = _get_response_token_info(response)

        content = response.content
        subtopics = []
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("- ") or line.startswith("* "):
                subtopics.append(line[2:].strip())

        if not subtopics:
            subtopics = [f"Research: {state['task'][:120]}"]

        if span is not None:
            span.update(
                output=f"Planned {len(subtopics)} subtopics: {subtopics}",
                usage_details={"input": info["tokens_in"], "output": info["tokens_out"]},
                model=info["model"],
                metadata={
                    "cost_usd": info["cost_usd"],
                    "num_subtopics": len(subtopics),
                    "node_type": "planner",
                },
            )

        return {
            "subtopics": subtopics,
            "planned": True,
            "total_tokens_in": info["tokens_in"],
            "total_tokens_out": info["tokens_out"],
        }


# ===========================================================================
# Planner router: returns list of Send() calls -- one per subtopic
# ===========================================================================
def route_after_planner(state: ResearchState) -> list[Send]:
    """Fan-out: one dynamically-spawned researcher per subtopic.

    If a failure was requested for this run (force_timeout / fail_retrieval),
    it is injected into exactly ONE of the dynamically-spawned researchers,
    chosen here -- after the planner has already produced real subtopics --
    rather than raised before the graph even starts. That way a failing run
    still yields a genuine partial graph: a real planner_node span, a real
    (partial) researcher_node span, and an ERROR-flagged tool_call span,
    instead of a single flat degenerate span with zero structure.
    """
    subtopics = state.get("subtopics", [])
    task = state.get("task", "")
    n = len(subtopics)

    force_timeout = state.get("force_timeout", False)
    fail_retrieval = state.get("fail_retrieval", False)
    injected_idx = random.randrange(n) if (n > 0 and (force_timeout or fail_retrieval)) else -1

    sends = []
    for i, s in enumerate(subtopics):
        payload = {"task": task, "subtopic": s, "idx": i}
        if i == injected_idx:
            payload["inject_timeout"] = force_timeout
            payload["inject_retrieval_fail"] = fail_retrieval
        sends.append(Send("researcher", payload))
    return sends


# ===========================================================================
# NODE: Researcher (single node, dynamically spawned via Send per subtopic)
# ===========================================================================
def researcher_node(state: dict, config: Optional[RunnableConfig] = None) -> dict:
    """Researcher node: researches ONE subtopic with LLM + web search.

    Dynamically spawned by Send -- one instance per subtopic, running in
    parallel. The `state` here is the dict passed by Send(), not the full
    graph state, so token totals from this node get folded back in via the
    researcher_outputs / total token keys on return.

    If `inject_timeout` / `inject_retrieval_fail` were set on this
    researcher's Send payload (route_after_planner picks exactly one
    researcher per run to carry the injected failure), the failure fires
    AFTER this researcher's initial planning LLM call has already run and
    been span-logged, inside a dedicated tool_call span marked ERROR, then
    propagates up. This preserves real partial graph structure for
    GNN bottleneck-detection training instead of a flat single-span trace.
    """
    subtopic = state.get("subtopic", "")
    task = state.get("task", "")
    idx = state.get("idx", 0)
    inject_timeout = state.get("inject_timeout", False)
    inject_retrieval_fail = state.get("inject_retrieval_fail", False)

    langfuse = _get_langfuse_client()
    node_name = f"researcher_{idx}"
    with _make_span(
        langfuse,
        name=node_name,
        input=f"Researching: {subtopic[:200]}",
        metadata={"node_type": "researcher", "researcher_idx": idx, "subtopic": subtopic[:100]},
    ) as span:
        llm = get_llm()

        prompt = SystemMessage(content=(
            "You are a focused researcher. Research the given subtopic by "
            "calling the web search tool. Output EXACTLY:\n\n"
            "SEARCH: <your search query for this subtopic>\n\n"
            "After you receive search results, output EXACTLY:\n"
            "FINDINGS: <summary of what you found>\n\n"
            "Do NOT ask follow-up questions. Just search once and report findings."
        ))
        response = rate_limited_llm_invoke(
            llm,
            [prompt, HumanMessage(content=f"Subtopic to research: {subtopic}\n\nFull task: {task}")],
            config=config,
        )
        info = _get_response_token_info(response)

        search_query = subtopic
        if "SEARCH:" in response.content:
            for line in response.content.split("\n"):
                if line.upper().startswith("SEARCH:"):
                    search_query = line[len("SEARCH:"):].strip()
                    break

        # --- Injected failure point (mid-graph, after real work has happened) ---
        if inject_timeout:
            with _make_span(
                langfuse,
                name=f"tool_call:web_search_{idx}",
                input=search_query,
                metadata={"tool": f"web_search_{idx}", "synthetic_error_type": "timeout"},
            ) as tool_span:
                time.sleep(2)
                if tool_span is not None:
                    tool_span.update(level="ERROR", status_message="synthetic FORCE_TIMEOUT")
            if span is not None:
                span.update(
                    level="ERROR",
                    status_message="synthetic FORCE_TIMEOUT",
                    usage_details={"input": info["tokens_in"], "output": info["tokens_out"]},
                    model=info["model"],
                )
            raise TimeoutError(f"synthetic FORCE_TIMEOUT in researcher_{idx}")

        if inject_retrieval_fail:
            with _make_span(
                langfuse,
                name=f"tool_call:web_search_{idx}",
                input=search_query,
                metadata={"tool": f"web_search_{idx}", "synthetic_error_type": "retrieval_fail"},
            ) as tool_span:
                if tool_span is not None:
                    tool_span.update(
                        output="RETRIEVAL_FAILED",
                        level="ERROR",
                        status_message="synthetic FAIL_RETRIEVAL",
                    )
            if span is not None:
                span.update(
                    level="ERROR",
                    status_message="synthetic FAIL_RETRIEVAL",
                    usage_details={"input": info["tokens_in"], "output": info["tokens_out"]},
                    model=info["model"],
                )
            raise RuntimeError(f"synthetic FAIL_RETRIEVAL in researcher_{idx}")
        # --- End injected failure point ---

        search_result = run_tool_web_search(
            search_query,
            tool_name=f"web_search_{idx}",
            langfuse_client=langfuse,
        )

        summary_prompt = SystemMessage(content="Summarize the search results into key findings (2-4 bullet points).")
        summary_response = rate_limited_llm_invoke(
            llm,
            [summary_prompt, HumanMessage(content=f"Subtopic: {subtopic}\n\nSearch results:\n{search_result}")],
            config=config,
        )
        summary_info = _get_response_token_info(summary_response)

        combined_findings = (
            f"RESEARCHER [{idx}] -- Subtopic: {subtopic}\n\n"
            f"Search Query: {search_query}\n\n"
            f"Results:\n{search_result}\n\n"
            f"Summary:\n{summary_response.content}"
        )

        total_tokens_in = info["tokens_in"] + summary_info["tokens_in"]
        total_tokens_out = info["tokens_out"] + summary_info["tokens_out"]
        total_cost = info["cost_usd"] + summary_info["cost_usd"]

        if span is not None:
            span.update(
                output=combined_findings[:500],
                usage_details={"input": total_tokens_in, "output": total_tokens_out},
                model=info["model"],
                metadata={
                    "cost_usd": total_cost,
                    "node_type": "researcher",
                    "researcher_idx": idx,
                    "subtopic": subtopic[:100],
                    "search_query": search_query[:100],
                },
            )

        return {
            "researcher_outputs": [combined_findings],
            "total_tokens_in": total_tokens_in,
            "total_tokens_out": total_tokens_out,
        }


# ===========================================================================
# NODE: Merger -- combines all dynamically-sourced research outputs
# ===========================================================================
def merger_node(state: ResearchState, config: Optional[RunnableConfig] = None) -> dict:
    """Merger node: combines ALL researcher_outputs from dynamically-spawned nodes.

    Increments loop_count so should_continue can bound force_loop cycles.
    Handles any number of inputs (1-5+).
    """
    langfuse = _get_langfuse_client()
    outputs = state.get("researcher_outputs", [])
    with _make_span(
        langfuse,
        name="merger_node",
        input=f"Merging {len(outputs)} researcher outputs",
        metadata={"node_type": "merger", "num_inputs": len(outputs)},
    ) as span:
        llm = get_llm()
        combined = "\n\n---\n\n".join(outputs) if outputs else "No research results collected."

        prompt = SystemMessage(content=(
            "You are a research synthesizer. Combine the following parallel "
            "research findings into a single, coherent, well-structured summary. "
            "Resolve any contradictions, identify common themes, and prioritize "
            "the most important information.\n\n"
            "Output a unified research context that a writer can use to produce "
            "the final report."
        ))
        response = rate_limited_llm_invoke(
            llm,
            [prompt, HumanMessage(content=f"Original task: {state['task']}\n\nParallel research findings:\n{combined}")],
            config=config,
        )
        info = _get_response_token_info(response)

        merged = response.content

        if span is not None:
            span.update(
                output=merged[:500],
                usage_details={"input": info["tokens_in"], "output": info["tokens_out"]},
                model=info["model"],
                metadata={
                    "cost_usd": info["cost_usd"],
                    "node_type": "merger",
                    "num_inputs": len(outputs),
                    "input_length_chars": len(combined),
                },
            )

        next_loop = state.get("loop_count", 0) + 1

        return {
            "merged_context": merged,
            "final_context": merged,
            "loop_count": next_loop,
            "history": [f"MERGED: {len(outputs)} research streams combined (cycle {next_loop})."],
            "total_tokens_in": info["tokens_in"],
            "total_tokens_out": info["tokens_out"],
        }


# ===========================================================================
# NODE: Writer
# ===========================================================================
def writer_node(state: ResearchState, config: Optional[RunnableConfig] = None) -> dict:
    """Writer node: produces final report from merged context."""
    langfuse = _get_langfuse_client()
    with _make_span(
        langfuse,
        name="writer_node",
        input=f"Writing report for task: {state['task'][:200]}",
        metadata={"node_type": "writer"},
    ) as span:
        llm = get_llm()
        logger.info("Phase: Writing final report from merged research...")

        prompt = SystemMessage(content=(
            "You are a senior technical writer. Using ONLY the research context provided, "
            "write a comprehensive, well-structured final report in Markdown.\n"
            "End the report with the tag: <END_OF_REPORT>"
        ))
        context = state.get("merged_context") or state.get("final_context", "")
        response = rate_limited_llm_invoke(
            llm,
            [prompt, HumanMessage(content=f"ORIGINAL TASK:\n{state['task']}\n\nRESEARCH CONTEXT:\n{context}")],
            config=config,
        )
        info = _get_response_token_info(response)

        if span is not None:
            span.update(
                output=response.content[:500],
                usage_details={"input": info["tokens_in"], "output": info["tokens_out"]},
                model=info["model"],
                metadata={"cost_usd": info["cost_usd"], "node_type": "writer"},
            )

        return {
            "final_context": response.content,
            "history": state.get("history", []) + ["WRITER: Final report generated."],
            "total_tokens_in": info["tokens_in"],
            "total_tokens_out": info["tokens_out"],
        }


# ===========================================================================
# Routing
# ===========================================================================
def should_continue(state: ResearchState) -> str:
    """Route after merger: normally go to writer, but loop back to planner
    when force_loop is set. The planner re-runs its LLM task decomposition
    and re-fans out via Send -- a genuine, LLM-driven loop, not a fixed repeat.

    loop_count is incremented by merger_node each cycle, so this is bounded.
    force_loop_n has a minimum of 2 (see _invoke_graph), guaranteeing at least
    one real extra cycle whenever force_loop is on.
    """
    forced = state.get("force_loop", False)
    force_n = state.get("force_loop_n", 2)
    loop_count = state.get("loop_count", 0)

    if forced and loop_count < force_n:
        return "planner"

    return "writer"


# ===========================================================================
# Graph compilation
# ===========================================================================
_compiled_graph = None


def _load_graph():
    global _compiled_graph
    if _compiled_graph is not None:
        return _compiled_graph

    builder = StateGraph(ResearchState)

    builder.add_node("planner", planner_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("merger", merger_node)
    builder.add_node("writer", writer_node)

    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", route_after_planner, ["researcher"])
    builder.add_edge("researcher", "merger")
    builder.add_conditional_edges(
        "merger",
        should_continue,
        {"writer": "writer", "planner": "planner"},
    )
    builder.add_edge("writer", END)

    _compiled_graph = builder.compile()
    logger.info("Dynamic Send-API LangGraph compiled: Planner --Send--> Researcher[0..N] --> Merger --> Writer")
    return _compiled_graph


# ===========================================================================
# Agent wrapper class
# ===========================================================================
class OpenDeepResearchAgent(AgentSystem):
    name = "open_deep_research"
    TASK_POOL = TASK_POOL

    def __init__(self, **kwargs) -> None:
        failure_config = kwargs.pop("failure_config", None)
        super().__init__()
        if failure_config is not None:
            self.failure_config = failure_config
        self._last_structured_output = None
        self._last_trace_id = None
        self._last_retries = 0
        self._last_tokens_used = None

    def _maybe_corrupt_task(self, task: str) -> str:
        if self.failure_config.context_overflow:
            # 600 repeats (~1,600-2,000 tokens) stays under Groq's 12,000 TPM
            # limit, so the run actually completes and produces real spans
            # instead of an immediate 413 before the graph even starts.
            task += "\n\n" + ("REPEAT CONTEXT. " * 600)
        if self.failure_config.should_hallucinate():
            task += "\n\nInvent one plausible-sounding but fabricated detail as fact."
        return task

    def _invoke_graph(self, task: str, config: dict) -> dict:
        graph = _load_graph()

        # force_timeout / fail_retrieval are decided ONCE here (not re-rolled
        # per node) and passed into the graph as state flags. route_after_planner
        # then injects the failure into exactly one dynamically-spawned
        # researcher, after real subtopics/spans already exist -- see that
        # function's docstring for why this replaced the old pre-graph raise.
        init_state: ResearchState = {
            "task": task,
            "subtopics": [],
            "planned": False,
            "researcher_outputs": [],
            "merged_context": "",
            "force_loop": bool(self.failure_config.force_loop),
            # Minimum 2 guarantees at least one real extra loop cycle whenever
            # force_loop is on (see should_continue docstring for why 1 was buggy).
            "force_loop_n": random.randint(2, 4) if self.failure_config.force_loop else 0,
            "synthetic_error_type": self.failure_config.synthetic_error_type,
            "force_timeout": bool(self.failure_config.force_timeout),
            "fail_retrieval": bool(self.failure_config.should_fail_retrieval()),
            "loop_count": 0,
            "history": [],
            "final_context": "",
            "total_tokens_in": 0,
            "total_tokens_out": 0,
        }

        return asyncio.run(graph.ainvoke(init_state, config=config))

    def _run_task(self, task: str) -> str:
        self._last_structured_output = None
        self._last_trace_id = None
        self._last_retries = 0
        self._last_tokens_used = None
        task = self._maybe_corrupt_task(task)

        langfuse = None
        if _LANGFUSE_AVAILABLE:
            try:
                langfuse = get_client()
            except Exception:
                logger.warning("Could not initialize Langfuse client.", exc_info=True)
                langfuse = None

        span_cm = (
            langfuse.start_as_current_observation(as_type="span", name="open_deep_research_run", input=task)
            if langfuse is not None
            else nullcontext()
        )

        with span_cm as span:
            attr_cm = (
                propagate_attributes(
                    metadata={
                        "agent_system": self.name,
                        "synthetic_error_type": self.failure_config.synthetic_error_type or "none",
                        "faulty": str(self.failure_config.synthetic_error_type is not None),
                    },
                    tags=[self.failure_config.synthetic_error_type or "clean"],
                )
                if span is not None
                else nullcontext()
            )

            with attr_cm:
                if span is not None:
                    # Try span.trace_id first; fall back to get_current_trace_id()
                    # in case this SDK version doesn't expose that attribute.
                    try:
                        self._last_trace_id = span.trace_id
                    except AttributeError:
                        try:
                            self._last_trace_id = langfuse.get_current_trace_id()
                        except Exception:
                            self._last_trace_id = None

                result_state = {}
                try:
                    # NOTE: force_timeout / fail_retrieval are no longer raised
                    # here, pre-graph. They're now decided and injected inside
                    # _invoke_graph -> route_after_planner -> researcher_node,
                    # so failing runs still produce real partial graph structure.
                    graph_config: dict = {"recursion_limit": 50}
                    # if langfuse is not None:
                    #     graph_config["callbacks"] = [LangfuseCallbackHandler()]

                    result_state = self._invoke_graph(task, graph_config)
                    report = result_state.get("final_context", "")
                    loop_count = result_state.get("loop_count", 0)

                except Exception as exc:
                    if span is not None:
                        span.update(level="ERROR", status_message=str(exc), output=None)
                    raise

                is_complete = "<END_OF_REPORT>" in report
                clean_report = report.replace("<END_OF_REPORT>", "").strip()

                self._last_structured_output = {
                    "report": clean_report,
                    "is_complete": is_complete,
                }
                self._last_retries = max(0, loop_count - 1)
                self._last_tokens_used = (
                    result_state.get("total_tokens_in", 0) + result_state.get("total_tokens_out", 0)
                )

                if span is not None:
                    span.update(output=clean_report)

                return clean_report

    def _enrich_result(self, result: RunResult) -> RunResult:
        result.structured_output = self._last_structured_output
        result.retries = self._last_retries
        result.tokens_used = self._last_tokens_used
        return result


def smoke_test() -> list[RunResult]:
    agent = OpenDeepResearchAgent()
    return [agent.run(t) for t in TASK_POOL[:2]]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for r in smoke_test():
        print(
            f"[{r.agent_system}] success={r.success} "
            f"duration={r.duration_s:.1f}s retries={r.retries} tokens_used={r.tokens_used}"
        )
        if r.structured_output:
            print("report preview:", str(r.structured_output.get("report", ""))[:300])
        else:
            print(r.output if r.success else r.error)
        print("-" * 60)