# agentic-pipeline-benchmarking — Person A starter (CrewAI)

## No paid API needed -- three LLM modes

The code auto-detects which mode to run in based on what's in `.env`
(`agents/crewai_agent.py::_resolve_llm_mode`):

| Mode | Cost | Setup | What you get |
|---|---|---|---|
| `offline` (default) | $0, no network | none -- just run it | Deterministic answers from the local knowledge base + calculator. No LLM calls at all. Good for testing the pipeline, not for real output quality. See `agents/offline.py`. |
| `ollama` | $0, runs locally | Install Ollama, `ollama pull llama3.2:3b` | Real LLM reasoning, tool use, structured output -- everything in this README -- with zero per-token cost. |
| `cloud` | Usually free-tier, can be paid | API key (e.g. Groq's free tier) | Same as `ollama` but via a hosted API. |

Leave `.env` empty (or just `cp .env.example .env` without editing) and
you get `offline` mode automatically -- no key, no install, no network call,
ever. Run `python agents/crewai_agent.py` right now and it works.

For real LLM-backed output without paying, **Ollama is the cleanest free
option**: it exposes an OpenAI-compatible endpoint, and CrewAI/litellm both
recognize the `ollama/` model prefix natively (verified against the
installed litellm 1.90.0 -- `ollama/llama3.2:3b` resolves correctly to
provider `ollama`, default endpoint `http://localhost:11434`).

```bash
# 1. Install Ollama (https://ollama.com/download), then:
ollama pull llama3.2:3b      # small, runs on most laptops
# stronger machine? try: ollama pull qwen2.5:7b   or   ollama pull llama3.1:8b

# 2. In .env:
echo "LLM_MODEL=ollama/llama3.2:3b" >> .env
# that's it -- no base_url or api_key needed for the default local setup

# 3. Run it
python agents/crewai_agent.py
```

Langfuse stays fully optional in every mode -- leave those `.env` lines
commented out and the app runs fine, it just won't export traces anywhere.

---


This is the Day 1 system per the project doc: **CrewAI**, picked as the
lowest-friction starting point (and to stay clear of FinRobot, since that
one may already be claimed). Structure follows Appendix A of the project
doc so `open_deep_research` or `finrobot` can be dropped in later as
`agents/open_deep_research_agent.py` / `agents/finrobot_agent.py` without
touching `run_batch.py` or `app.py` — they just register in `agents/__init__.py`.

## What's here

```
agents/
  base.py            # AgentSystem ABC + FailureInjectionConfig + RunResult
  schemas.py         # Pydantic output contracts (ResearchFindings, FinalAnswer)
  tools.py           # calculator, web_search (DuckDuckGo), local_knowledge (fail-injectable)
  evaluation.py       # guardrails (in-loop) + run_labels heuristics (post-hoc)
  crewai_agent.py     # full CrewAI implementation -- see below
  __init__.py         # REGISTRY = {"crewai": CrewAIAgent}
telemetry/
  instrument.py       # one-liner Langfuse/OTel instrumentation hook
run_batch.py          # CLI: --system crewai --n 30 [--faulty --error-type loop]
app.py                # FastAPI POST /run?system=&n=&faulty=&error_type=
Dockerfile
docker-compose.yml    # app container only -- see Langfuse note below
requirements.txt
.env.example
```

## The CrewAI implementation, end-to-end

`agents/crewai_agent.py` is fully built out, not a stub:

**Task decomposition.** Each task is split into a two-agent sequential
pipeline: Researcher gathers facts using tools, Writer synthesizes them into
a final answer using only the research context (no tools, so it can't
invent new "facts" via search). `Task(context=[research_task])` wires the
research output into the writer's prompt automatically.

**Tool usage.** The Researcher has three tools, deliberately different in
nature so the trace graph gets real `role=tool` node diversity:
- `local_knowledge` — small in-repo knowledge base; this is also the hook
  point for `FAIL_RETRIEVAL_PROB` injection (a real tool-call failure, not
  a prompt trick)
- `web_search` — real DuckDuckGo search, no API key needed; degrades to a
  clear "unavailable" string on network failure rather than raising or
  fabricating results
- `calculator` — sandboxed arithmetic eval (character-whitelisted, no
  builtins) for cost/budget comparisons

**Structured outputs.** Both tasks use `output_pydantic` — `ResearchFindings`
and `FinalAnswer` (`agents/schemas.py`). This isn't "ask nicely for JSON in
the prompt": CrewAI validates against the Pydantic model and only accepts
the task as complete once it parses.

**Retries — two distinct layers:**
1. *Guardrail retries* (in-loop): each task has a `guardrail` function
   (`agents/evaluation.py::research_guardrail` / `final_answer_guardrail`)
   that checks the parsed output isn't just schema-valid but *substantively*
   valid (non-empty facts, confidence in range, summary/details long enough).
   On failure CrewAI automatically re-prompts the agent with the rejection
   reason, up to `guardrail_max_retries` (env: `RESEARCH_GUARDRAIL_RETRIES`,
   `WRITER_GUARDRAIL_RETRIES`, default 2).
2. *Crew-level retries* (transient failures): `_kickoff_with_retry` wraps
   the whole `crew.kickoff()` call in a `tenacity` retry with exponential
   backoff (2s/4s/8s), triggered only on `TimeoutError` / `ConnectionError`
   / `OSError` — i.e. infra blips, not bad output (which guardrails already
   handle). Configurable via `CREW_MAX_RETRIES` (default 3). Verified with a
   fake crew that fails twice then succeeds, and one that always fails.

**Evaluation hooks.** `agents/evaluation.py` separates the two places
evaluation actually happens: guardrails run *during* execution and can
block/retry a task; `compute_run_labels` / `evaluate_batch` run *after* a
batch completes and compute `run_labels`-style stats (`success`, `slow`,
`expensive`) matching the shared schema (Section 3) so you can sanity-check
output locally before it ever reaches Person C's labeling pipeline.
`run_batch.py` now prints a scorecard at the end of every batch.

**`RunResult` (agents/base.py)** carries `structured_output` (the Writer's
`FinalAnswer` as a dict), `tokens_used`, and `retries` alongside the
original fields — all enriched via a `_enrich_result` hook so the base
`AgentSystem.run()` contract didn't need to change.

## Verified against the actual installed packages (not just the doc)

I built and tested this against real installs in a sandbox — not from
memory — and want to flag what's drifted or wasn't obvious from docs alone:

1. **`crewai`'s `groq/llama-3.1-70b-versatile` model string needs `litellm`
   installed explicitly.** Current CrewAI (1.15.1) only resolves native
   providers without it; `groq/` isn't native, so it falls through to
   LiteLLM. Added to `requirements.txt`.

2. **Guardrail functions can't have stringified return-type annotations.**
   If the module has `from __future__ import annotations`, function return
   annotations become strings at runtime, and CrewAI's `Task` validator
   (which inspects the guardrail's signature via `inspect.signature`)
   fails with a cryptic `ValueError` because it can't resolve `tuple[bool,
   Any]` as a string. Fix: don't annotate the guardrail's return type at
   all (the docstring documents the contract instead) — see
   `agents/evaluation.py`.

3. **Langfuse self-hosting is not Postgres-only anymore.** The project doc
   (Section 6) describes a 2-service setup. As of Langfuse v3, self-hosting
   needs **six** containers: web, worker, Postgres, ClickHouse, Redis,
   MinIO. For Sprint 1's ~300-400 trace target, I'd suggest **Langfuse
   Cloud's free tier** instead — set `LANGFUSE_HOST=https://cloud.langfuse.com`
   in `.env` and skip the six-container stack. A correct (but commented-out)
   self-hosted block is in `docker-compose.yml` if the team decides
   otherwise.

4. **litellm needs the `ollama/` prefix, not a bare model name + base_url.**
   `LLM(model="llama3.2:3b", base_url="http://localhost:11434/v1")` raises
   `BadRequestError: LLM Provider NOT provided` — litellm routes by prefix,
   not by base_url alone. Use `LLM_MODEL=ollama/llama3.2:3b`. Verified
   directly with `litellm.get_llm_provider()`.

Everything else (Agent/Task/Crew/Process field names, `output_pydantic`,
`guardrail`/`guardrail_max_retries`, `@tool` decorator signature,
`openinference-instrumentation-crewai`, `langfuse`, `tenacity`) matched the
doc and installed versions — checked by direct import/construction, not
assumption.

## Quickstart

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# fill in OPENAI_API_KEY (Groq) and, once you've decided, LANGFUSE_PUBLIC_KEY/SECRET_KEY

# smoke test (3 tasks: product compare, trip plan, bug explain) -- no server needed
python agents/crewai_agent.py

# or via the batch CLI (prints a scorecard at the end)
python run_batch.py --system crewai --n 3

# or via the API
uvicorn app:app --reload --port 8000
curl -X POST "http://localhost:8000/run?system=crewai&n=3"
```

### Running a faulty batch (failure injection, Appendix B)

```bash
python run_batch.py --system crewai --n 10 --faulty --error-type loop
python run_batch.py --system crewai --n 10 --faulty --error-type retrieval_fail --prob 0.3
```

Note on injection fidelity: `loop` (task repetition), `timeout` (real sleep
+ raise), and `retrieval_fail` (the `local_knowledge` tool genuinely returns
a failure string) are mechanically real. `hallucination` and
`context_overflow` are **prompt-level approximations**, since a tool-less
Writer step has no natural mechanism to fail that way — see the docstring
in `agents/crewai_agent.py::_maybe_corrupt_task`. If those two motifs'
density matters a lot for the GNN, they'll be more naturally sourced from
`open_deep_research` or a RAG-style system (project doc Section 10 makes
the same point about `agentic-rag-for-dummies`).

## What's NOT done yet (by design — other people's rows)

- Langfuse deployment + `export_traces.py` → `data/raw/*.json` schema mapping (Person B)
- TRAIL ingestion, labeling, `build_dataset.py` (Person C)
- `open_deep_research` and `finrobot` system wrappers (whoever picks those up next)

## Next step for you

Run the smoke test, confirm `data/raw/agent_system=crewai/batch_*.jsonl` gets
written with `structured_output` populated, and — once Langfuse is reachable
(Cloud or self-hosted) — confirm the 3 smoke-test traces actually show up
with spans visible, per Section 15 Day 1 checklist.
