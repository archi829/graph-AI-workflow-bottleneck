"""
Minimal Langfuse instrumentation so Person A's smoke test ("confirm 3 traces
appear in Langfuse with spans visible", Section 15) actually works end to end
without waiting on Person B's full telemetry/ package.

Person B's export_traces.py is the real owner of the Langfuse stack (Section
4); this module just turns instrumentation on for the current process. Safe
to call multiple times (idempotent no-op if already instrumented).

Usage:
    from telemetry.instrument import instrument_crewai
    instrument_crewai()
"""

from __future__ import annotations

import os

_instrumented = False


def instrument_crewai() -> None:
    global _instrumented
    if _instrumented:
        return

    # Langfuse SDK v3+ reads these from the environment automatically.
    # Set them in .env (see .env.example) -- defaults point at the local
    # docker-compose Langfuse instance from Section 6.
    os.environ.setdefault("LANGFUSE_HOST", "http://localhost:3000")

    from openinference.instrumentation.crewai import CrewAIInstrumentor

    CrewAIInstrumentor().instrument()
    _instrumented = True
