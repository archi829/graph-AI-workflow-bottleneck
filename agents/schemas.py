"""
Structured output contracts for the CrewAI crew's tasks.

Using `output_pydantic` (verified field on crewai.Task in the installed
version) forces the LLM's final task output through Pydantic validation
before the crew considers the task done. This is what makes "structured
outputs" real rather than just "ask nicely for JSON in the prompt": CrewAI
re-prompts the model if the output doesn't parse, since output_pydantic is
backed by its converter/guardrail machinery.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ResearchFindings(BaseModel):
    """Output contract for the Researcher agent's task."""

    key_facts: list[str] = Field(
        ..., min_length=1, description="Bullet list of the key facts or tradeoffs found."
    )
    open_questions: list[str] = Field(
        default_factory=list, description="Anything the researcher couldn't resolve confidently."
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Researcher's self-rated confidence in these findings."
    )


class FinalAnswer(BaseModel):
    """Output contract for the Writer agent's task -- also the trace's final payload."""

    summary: str = Field(..., min_length=1, description="2-4 sentence executive summary.")
    details: str = Field(..., min_length=1, description="Full structured answer body.")
    recommendation: str | None = Field(
        default=None, description="A single concrete recommendation, if the task calls for one."
    )
    sources_used: list[str] = Field(
        default_factory=list, description="Names of tools/sources consulted while answering."
    )
