"""
Custom ReAct Deep Research Agent.
LangGraph loop: Researcher (ReAct loop) -> Writer
Groq handles reasoning only. Tool execution is pure Python.

Langfuse tracing notes (fixed version):
- Uses the real `langfuse.langchain.CallbackHandler` (langfuse>=3), not the
  LANGCHAIN_TRACING_V2 env var (that's a LangSmith flag and does nothing here).
- The trace is opened BEFORE synthetic failure injection runs, so `timeout`
  and `retrieval_fail` errors still produce a (failed) trace in Langfuse
  instead of vanishing silently.
- `force_loop` is threaded into the graph state itself, so a forced loop
  shows up as genuine repeated researcher<->tool_router edges inside ONE
  trace, instead of three separate throwaway graph runs.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import nullcontext
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from typing import Optional, TypedDict


from .base import AgentSystem, RunResult

logger = logging.getLogger(__name__)

# --- Langfuse (optional import: agent still runs, just untraced, if missing) ---
try:
    from langfuse import get_client
    from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
    from langfuse import propagate_attributes
    _LANGFUSE_AVAILABLE = True
except ImportError:
    _LANGFUSE_AVAILABLE = False
    logger.warning("langfuse not installed — runs will NOT be traced. `pip install langfuse`.")


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

# How many extra researcher<->tool_router round-trips to force when
# force_loop is set, before FINAL_CONTEXT is allowed to end the graph.
FORCE_LOOP_MIN_ITERATIONS = 5
# Hard safety cap regardless of force_loop, to prevent runaway graphs.
HARD_LOOP_CAP = 8


def run_web_search(query: str) -> str:
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=4))
        return "\n".join(f"- {r['title']}: {r['body']}" for r in results) if results else "No results found."
    except Exception as exc:
        return f"Search failed: {exc}"


def get_llm():
    api_key = _env("GROQ_API_KEY", "LLM_API_KEY")
    model = _env("ODR_MODEL", "LLM_MODEL") or "llama-3.3-70b-versatile"
    if not api_key:
        raise EnvironmentError("Requires GROQ_API_KEY in .env")
    return ChatGroq(model=model, api_key=api_key, temperature=0.2)


class ResearchState(TypedDict):
    task: str
    loop_count: int
    history: list[str]
    final_context: str
    force_loop: bool


def researcher_node(state: ResearchState, config: Optional[RunnableConfig] = None) -> dict:
    llm = get_llm()
    history_str = "\n".join(state["history"]) if state["history"] else "None yet."

    prompt = SystemMessage(content=(
        "You are an expert research planner gathering information in a loop.\n\n"
        "INSTRUCTIONS:\n"
        "If you need more information, output EXACTLY:\n"
        "SEARCH: <your search query here>\n\n"
        "If you have enough information, output EXACTLY:\n"
        "FINAL_CONTEXT: <summarize all key facts you found>"
    ))

    user_msg = HumanMessage(content=f"TASK: {state['task']}\n\nPREVIOUS STEPS:\n{history_str}")
    # Forward `config` so the Langfuse callback (if attached at graph.ainvoke
    # time) nests this LLM call under the same trace as every other node,
    # across every loop iteration.
    response = llm.invoke([prompt, user_msg], config=config)

    return {
        "history": state["history"] + [response.content],
        "loop_count": state["loop_count"] + 1,
    }


def tool_router_node(state: ResearchState, config: Optional[RunnableConfig] = None) -> dict:
    last_action = state["history"][-1] if state["history"] else ""

    if last_action.upper().startswith("SEARCH:"):
        query = last_action[len("SEARCH:"):].strip()
        logger.info(f"Loop {state['loop_count']}: Searching for '{query}'")
        search_result = run_web_search(query)
        new_entry = f"SEARCH RESULT for '{query}':\n{search_result}"
        return {
            "history": state["history"] + [new_entry],
            "final_context": state["final_context"],
        }

    elif last_action.upper().startswith("FINAL_CONTEXT:"):
        logger.info(f"Loop {state['loop_count']}: Agent finished gathering info.")
        context = last_action[len("FINAL_CONTEXT:"):].strip()
        return {
            "history": state["history"],
            "final_context": context,
        }

    else:
        # Lenient — just nudge back instead of hard failing
        logger.warning(f"Loop {state['loop_count']}: Bad format, nudging retry.")
        new_entry = "ERROR: Output must start with 'SEARCH:' or 'FINAL_CONTEXT:'. Try again."
        return {
            "history": state["history"] + [new_entry],
            "final_context": state["final_context"],
        }


def should_continue(state: ResearchState) -> str:
    """Route: if final_context is populated go to writer, else loop back.

    When force_loop is set, we deliberately ignore an early FINAL_CONTEXT
    for the first FORCE_LOOP_MIN_ITERATIONS rounds so the graph produces a
    genuine repeating researcher<->tool_router structure in the trace,
    instead of a single normal-looking short run.
    """
    forced = state.get("force_loop", False)

    if forced and state["loop_count"] < FORCE_LOOP_MIN_ITERATIONS:
        return "researcher"

    if state.get("final_context") and len(state["final_context"]) > 50:
        return "writer"

    if state["loop_count"] >= HARD_LOOP_CAP:
        logger.warning("Max loops reached, forcing writer.")
        return "writer"

    return "researcher"


def writer_node(state: ResearchState, config: Optional[RunnableConfig] = None) -> dict:
    llm = get_llm()
    logger.info("Phase 2: Writing final report...")

    prompt = SystemMessage(content=(
        "You are a senior technical writer. Using ONLY the research context provided, "
        "write a comprehensive, well-structured final report in Markdown.\n"
        "End the report with the tag: <END_OF_REPORT>"
    ))

    response = llm.invoke(
        [prompt, HumanMessage(content=f"ORIGINAL TASK:\n{state['task']}\n\nRESEARCH CONTEXT:\n{state['final_context']}")],
        config=config,
    )

    return {"final_context": response.content}


_compiled_graph = None


def _load_graph():
    global _compiled_graph
    if _compiled_graph is not None:
        return _compiled_graph

    builder = StateGraph(ResearchState)

    builder.add_node("researcher", researcher_node)
    builder.add_node("tool_router", tool_router_node)
    builder.add_node("writer", writer_node)

    builder.add_edge(START, "researcher")
    builder.add_edge("researcher", "tool_router")
    builder.add_conditional_edges(
        "tool_router",
        should_continue,
        {"researcher": "researcher", "writer": "writer"},
    )
    builder.add_edge("writer", END)

    _compiled_graph = builder.compile()
    logger.info("Custom ReAct graph compiled.")
    return _compiled_graph


class OpenDeepResearchAgent(AgentSystem):
    name = "open_deep_research"
    TASK_POOL = TASK_POOL

    def __init__(self, **kwargs) -> None:
        failure_config = kwargs.pop("failure_config", None)
        super().__init__()  # base class takes no args, don't pass any
        if failure_config is not None:
            self.failure_config = failure_config
        self._last_structured_output = None
        self._last_trace_id = None
        self._last_retries = 0

    def _maybe_inject_timeout(self) -> None:
        if self.failure_config.force_timeout:
            time.sleep(2)
            raise TimeoutError("synthetic FORCE_TIMEOUT")

    def _maybe_corrupt_task(self, task: str) -> str:
        if self.failure_config.context_overflow:
            task += "\n\n" + ("REPEAT CONTEXT. " * 4000)
        if self.failure_config.should_hallucinate():
            task += "\n\nInvent one plausible-sounding but fabricated detail as fact."
        return task

    def _maybe_fail_retrieval(self) -> None:
        if self.failure_config.should_fail_retrieval():
            raise RuntimeError("synthetic FAIL_RETRIEVAL")

    def _invoke_graph(self, task: str, config: dict) -> dict:
        """Runs the graph once and returns the raw final state dict
        (so the caller can read both final_context and loop_count)."""
        graph = _load_graph()

        init_state: ResearchState = {
            "task": task,
            "loop_count": 0,
            "history": [],
            "final_context": "",
            "force_loop": bool(self.failure_config.force_loop),
        }

        return asyncio.run(graph.ainvoke(init_state, config=config))

    def _run_task(self, task: str) -> str:
        self._last_structured_output = None
        self._last_trace_id = None
        self._last_retries = 0

        task = self._maybe_corrupt_task(task)

        langfuse = None
        if _LANGFUSE_AVAILABLE:
            try:
                langfuse = get_client()
            except Exception:
                logger.warning("Could not initialize Langfuse client; run will be untraced.", exc_info=True)
                langfuse = None

        

        span_cm = (
    langfuse.start_as_current_observation(
        as_type="span",
        name="open_deep_research_run",
        input=task,
    )
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
                    try:
                        self._last_trace_id = span.trace_id
                    except AttributeError:
                        self._last_trace_id = None

                try:
                    self._maybe_inject_timeout()
                    self._maybe_fail_retrieval()

                    graph_config: dict = {"recursion_limit": 50}
                    if langfuse is not None:
                        graph_config["callbacks"] = [LangfuseCallbackHandler()]

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

                if span is not None:
                    span.update(output=clean_report)

                return clean_report

    def _enrich_result(self, result: RunResult) -> RunResult:
        result.structured_output = self._last_structured_output
        result.retries = self._last_retries
        return result


def smoke_test() -> list[RunResult]:
    agent = OpenDeepResearchAgent()
    return [agent.run(t) for t in TASK_POOL[:3]]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for r in smoke_test():
        print(
            f"[{r.agent_system}] success={r.success} "
            f"duration={r.duration_s:.1f}s retries={r.retries}"
        )
        if r.structured_output:
            print("report preview:", str(r.structured_output.get("report", ""))[:300])
        else:
            print(r.output if r.success else r.error)
        print("-" * 60)