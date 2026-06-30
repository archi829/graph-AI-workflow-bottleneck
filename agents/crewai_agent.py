"""
CrewAI wrapper -- the Day 1 system (lowest friction per repo eval, Section 10).

End-to-end pipeline for one task:

    task string
      -> classify (product_compare / trip_plan / bug_explain)
      -> Researcher agent (tools: local_knowledge, web_search, calculator)
         -> Task[output_pydantic=ResearchFindings], guardrail-validated,
            auto-retried by CrewAI up to RESEARCH_GUARDRAIL_RETRIES times
      -> Writer agent (no tools -- synthesis only)
         -> Task[output_pydantic=FinalAnswer], guardrail-validated,
            auto-retried up to WRITER_GUARDRAIL_RETRIES times
      -> crew.kickoff() wrapped in a tenacity retry (CREW_MAX_RETRIES) for
         transient failures (rate limits, network blips) -- distinct from
         the in-loop guardrail retries above, which handle *malformed
         output* rather than *failed calls*.
      -> RunResult enriched with structured_output + run_labels via
         agents/evaluation.py

Covers the three smoke-test task types called out in the project doc:
  - "compare iPhone vs Pixel"      -> product_compare
  - "plan a 3-day trip to Goa"     -> trip_plan
  - "explain this Python bug"      -> bug_explain

Groq wiring follows Section 11 of the project doc:
    from crewai import LLM
    llm = LLM(model="groq/llama-3.1-70b-versatile")

NOTE: current crewai (1.15.1) needs `litellm` installed for non-native
model prefixes like `groq/` to resolve -- see requirements.txt / README.

Langfuse instrumentation is NOT done here -- that's Person B's package
(telemetry/). This module only needs CrewAIInstrumentor().instrument() to
have been called once, process-wide, before any crew.kickoff(). See
telemetry/instrument.py, and app.py for where it's invoked at startup.
"""

from __future__ import annotations

import logging
import os
import time

from crewai import LLM, Agent, Crew, Process, Task
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .base import AgentSystem, RunResult
from .evaluation import compute_run_labels, final_answer_guardrail, research_guardrail
from .schemas import FinalAnswer, ResearchFindings
from .tools import calculator, make_local_knowledge_tool, web_search

logger = logging.getLogger(__name__)

CREW_MAX_RETRIES = int(os.getenv("CREW_MAX_RETRIES", "3"))
RESEARCH_GUARDRAIL_RETRIES = int(os.getenv("RESEARCH_GUARDRAIL_RETRIES", "2"))
WRITER_GUARDRAIL_RETRIES = int(os.getenv("WRITER_GUARDRAIL_RETRIES", "2"))


def _build_llm() -> LLM:
    """Groq drop-in via the OpenAI-compatible endpoint (Section 11)."""
    return LLM(
        model=os.getenv("OPENAI_MODEL", "groq/llama-3.1-70b-versatile"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE", "https://api.groq.com/openai/v1"),
    )


# Transient, retry-worthy failures -- rate limits, connection errors, server
# 5xx -- as opposed to bad input or programming errors, which should fail fast.
_RETRYABLE_EXCEPTIONS = (TimeoutError, ConnectionError, OSError)


def _classify(task: str) -> str:
    """Crude keyword router. Good enough for batch generation where
    run_batch.py is picking tasks from a fixed pool (see TASK_POOL below).
    """
    t = task.lower()
    if "trip" in t or "travel" in t or "itinerary" in t:
        return "trip_plan"
    if "bug" in t or "code" in t or "traceback" in t or "error" in t:
        return "bug_explain"
    return "product_compare"


TASK_POOL = [
    "Compare the iPhone 16 and Google Pixel 9 for a college student on a budget.",
    "Compare the Sony WH-1000XM5 and Bose QC Ultra headphones for frequent flyers.",
    "Plan a 3-day trip to Goa for two people who like beaches and seafood.",
    "Plan a weekend trip to Jaipur focused on architecture and street food.",
    "Explain this Python bug: a recursive function that hits RecursionError "
    "on inputs larger than 900, and suggest a fix.",
    "Explain why a Flask app raises 'Working outside of application context' "
    "and how to fix it.",
]


class CrewAIAgent(AgentSystem):
    name = "crewai"
    TASK_POOL = TASK_POOL  # exposed on the class so run_batch.py can pull tasks generically

    def __init__(self) -> None:
        super().__init__()
        self.llm = _build_llm()
        # Stash from the most recent _run_task call so _enrich_result can
        # pick it up without changing the AgentSystem.run() contract.
        self._last_structured_output: dict | None = None
        self._last_tokens_used: int | None = None
        self._last_retries: int = 0

    # -- failure injection helpers -----------------------------------------
    def _maybe_inject_timeout(self) -> None:
        if self.failure_config.force_timeout:
            # Sleep past a typical tool-timeout threshold, then raise --
            # mirrors Appendix B's "FORCE_TIMEOUT injects artificial sleep
            # > tool timeout threshold" without needing a real tool call.
            time.sleep(2)
            raise TimeoutError("synthetic FORCE_TIMEOUT: tool call exceeded 2s budget")

    def _maybe_corrupt_task(self, description: str) -> str:
        """Best-effort hooks for the flags that make sense for CrewAI.

        retrieval_fail is handled properly via the local_knowledge tool
        itself (agents/tools.py::make_local_knowledge_tool), since that's a
        real tool-call failure rather than a prompt trick. context_overflow
        and hallucination don't have a natural mechanism in a tool-less
        Writer step, so they're approximated at the prompt level here --
        flag this in your synthetic_error_type review if motif fidelity
        matters downstream.
        """
        if self.failure_config.context_overflow:
            description = description + "\n\n" + ("REPEAT CONTEXT. " * 4000)
        if self.failure_config.should_hallucinate():
            description = (
                description
                + "\n\nWhen writing the final answer, invent one plausible-sounding "
                "but fabricated specific detail (a number, a name, or a spec) and "
                "present it as fact without flagging it as uncertain."
            )
        return description

    # -- crew builder -----------------------------------------------------
    def _crew_for(self, task_description: str) -> Crew:
        task_description = self._maybe_corrupt_task(task_description)

        researcher = Agent(
            role="Researcher",
            goal=f"Gather the key facts needed to address: {task_description}",
            backstory=(
                "A meticulous research agent that checks local knowledge first, "
                "falls back to web search for anything not already known, and "
                "uses the calculator for any numeric comparison rather than "
                "estimating by hand."
            ),
            llm=self.llm,
            tools=[
                make_local_knowledge_tool(self.failure_config.fail_retrieval_prob),
                web_search,
                calculator,
            ],
            allow_delegation=False,
            verbose=False,
        )
        writer = Agent(
            role="Writer",
            goal="Turn the researcher's findings into a clear, well-structured final answer.",
            backstory="A clear, concise technical writer who never invents facts not in the research.",
            llm=self.llm,
            allow_delegation=False,
            verbose=False,
        )

        research_task = Task(
            description=(
                f"Research the following request thoroughly, using your tools "
                f"(local_knowledge first, web_search if needed, calculator for any "
                f"arithmetic):\n{task_description}"
            ),
            expected_output=(
                "A ResearchFindings object: key_facts (list of concrete facts/tradeoffs), "
                "open_questions (anything unresolved), confidence (0.0-1.0)."
            ),
            agent=researcher,
            output_pydantic=ResearchFindings,
            guardrail=research_guardrail,
            guardrail_max_retries=RESEARCH_GUARDRAIL_RETRIES,
        )
        write_task = Task(
            description="Using the research findings above, write the final answer for the user.",
            expected_output=(
                "A FinalAnswer object: summary (2-4 sentences), details (full body), "
                "recommendation (one concrete suggestion, if applicable), sources_used."
            ),
            agent=writer,
            context=[research_task],
            output_pydantic=FinalAnswer,
            guardrail=final_answer_guardrail,
            guardrail_max_retries=WRITER_GUARDRAIL_RETRIES,
        )

        tasks = [research_task, write_task]
        if self.failure_config.force_loop:
            # Loop motif: re-run the research step N times before the writer
            # ever sees it, so the trace gets a repeated-node cycle for the
            # GNN to learn as a `loop` motif (Section 7 / Appendix B).
            tasks = [research_task, research_task, research_task, write_task]

        return Crew(
            agents=[researcher, writer],
            tasks=tasks,
            process=Process.sequential,
            verbose=False,
        )

    # -- crew-level retry wrapper -------------------------------------------
    def _kickoff_with_retry(self, crew: Crew):
        """Retries the *entire crew run* on transient failures (rate limits,
        connection errors). This is separate from -- and sits above -- the
        guardrail retries on individual tasks, which handle malformed
        output rather than failed calls. Exponential backoff: 2s, 4s, 8s.
        """
        attempts = {"count": 0}

        @retry(
            reraise=True,
            stop=stop_after_attempt(CREW_MAX_RETRIES),
            wait=wait_exponential(multiplier=2, min=2, max=30),
            retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
        )
        def _go():
            attempts["count"] += 1
            return crew.kickoff()

        result = _go()
        self._last_retries = attempts["count"] - 1
        return result

    # -- AgentSystem interface -------------------------------------------
    def _run_task(self, task: str) -> str:
        self._maybe_inject_timeout()
        self._last_structured_output = None
        self._last_tokens_used = None
        self._last_retries = 0

        crew = self._crew_for(task)
        crew_output = self._kickoff_with_retry(crew)

        # crew_output.pydantic is the *last* task's structured output
        # (the Writer's FinalAnswer), since CrewOutput mirrors the final
        # task in a sequential process.
        final_pydantic = getattr(crew_output, "pydantic", None)
        if final_pydantic is not None:
            self._last_structured_output = final_pydantic.model_dump()

        usage = getattr(crew_output, "token_usage", None)
        if usage is not None:
            self._last_tokens_used = getattr(usage, "total_tokens", None)

        return str(crew_output)

    def _enrich_result(self, result: RunResult) -> RunResult:
        result.structured_output = self._last_structured_output
        result.tokens_used = self._last_tokens_used
        result.retries = self._last_retries
        return result


def smoke_test() -> list[RunResult]:
    """Day-1 smoke test: 3 tasks, one of each type (project doc, Section 15)."""
    agent = CrewAIAgent()
    tasks = [
        "Compare the iPhone 16 and Google Pixel 9 for a college student on a budget.",
        "Plan a 3-day trip to Goa for two people who like beaches and seafood.",
        "Explain this Python bug: a recursive function that hits RecursionError "
        "on inputs larger than 900, and suggest a fix.",
    ]
    return [agent.run(t) for t in tasks]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for r in smoke_test():
        labels = compute_run_labels(
            success=r.success, duration_s=r.duration_s, tokens_used=r.tokens_used
        )
        print(f"[{r.agent_system}] success={r.success} duration={r.duration_s:.1f}s "
              f"retries={r.retries} run_labels={labels}")
        if r.structured_output:
            print("structured_output:", r.structured_output)
        else:
            print(r.output if r.success else r.error)
        print("-" * 60)
