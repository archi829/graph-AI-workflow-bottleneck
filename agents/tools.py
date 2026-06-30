"""
Tools available to the Researcher agent.

Three tools, deliberately different in nature so the trace graph gets real
`role=tool` node diversity (Section 3 schema, `spans[].tool`):

  - calculator        deterministic, no I/O -- good baseline tool-call span
  - web_search         real external call (DuckDuckGo, no API key required)
  - local_knowledge     in-repo retrieval -- the natural hook point for
                        FAIL_RETRIEVAL_PROB injection (Section 13/Appendix B)

web_search degrades gracefully if `duckduckgo-search` isn't installed or the
network call fails (e.g. offline dev, sandboxed CI) -- it returns a clear
"search unavailable" string rather than raising, so a flaky network doesn't
silently masquerade as a `retrieval_fail` motif. Real retrieval failures are
injected explicitly via local_knowledge / FAIL_RETRIEVAL_PROB instead.
"""

from __future__ import annotations

import random

from crewai.tools import tool

# Tiny local knowledge base so local_knowledge is deterministic and
# fail-injectable without depending on the network. Swap/expand this if a
# given task type needs more grounded facts.
_KNOWLEDGE_BASE = {
    "iphone 16": "iPhone 16: A18 chip, 6.1in display, USB-C, starts at $799.",
    "pixel 9": "Pixel 9: Tensor G4 chip, 6.3in display, USB-C, starts at $799.",
    "wh-1000xm5": "Sony WH-1000XM5: ~30hr battery, industry-leading ANC, $399.",
    "qc ultra": "Bose QC Ultra: ~24hr battery, Immersive Audio mode, $429.",
    "goa": "Goa: beach destination, peak season Nov-Feb, known for seafood and Portuguese-era architecture.",
    "jaipur": "Jaipur: 'Pink City', Amber Fort, Hawa Mahal, known for block-print textiles and street food.",
    "recursionerror": "Python's default recursion limit is 1000 (sys.getrecursionlimit()); deep recursion should usually be rewritten iteratively or with sys.setrecursionlimit() as a stopgap.",
    "application context": "Flask's 'working outside of application context' error means code touching current_app/g/session ran outside an app.app_context() block or a request.",
}


@tool("calculator")
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '120 * 3 + 45'.

    Use this for any cost, budget, or numeric-comparison step (trip
    budgets, price deltas, etc.) instead of doing mental math.
    """
    # Restricted eval: only digits, operators, parens, decimal points, spaces.
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return f"Error: expression contains disallowed characters: {expression!r}"
    try:
        # eval is safe here because the character whitelist above excludes
        # anything but arithmetic -- no names, no attribute access, no calls.
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
        return str(result)
    except Exception as exc:  # noqa: BLE001
        return f"Error evaluating '{expression}': {exc}"


@tool("web_search")
def web_search(query: str) -> str:
    """Search the web for current information not in the local knowledge base.

    Falls back to a clear 'unavailable' message if the search backend isn't
    reachable -- does not silently fabricate results.
    """
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return f"No web results found for: {query}"
        return "\n".join(f"- {r['title']}: {r['body']}" for r in results)
    except ImportError:
        return "web_search unavailable: duckduckgo-search not installed (pip install duckduckgo-search)."
    except Exception as exc:  # noqa: BLE001 -- network errors, rate limits, etc.
        return f"web_search unavailable ({exc}). Proceeding without it."


def make_local_knowledge_tool(fail_probability: float = 0.0):
    """Factory so each CrewAIAgent instance gets a tool wired to its own
    FailureInjectionConfig.fail_retrieval_prob (Appendix B), rather than a
    single module-level tool shared/mutated across instances.
    """

    @tool("local_knowledge")
    def local_knowledge(topic: str) -> str:
        """Look up a known fact about a product, destination, or bug pattern
        from the local knowledge base. Use this before falling back to web_search.
        """
        if fail_probability > 0 and random.random() < fail_probability:
            return "RETRIEVAL_FAILED: no results returned (synthetic FAIL_RETRIEVAL_PROB injection)."

        key = topic.strip().lower()
        for k, v in _KNOWLEDGE_BASE.items():
            if k in key or key in k:
                return v
        return f"No local knowledge entry for '{topic}'. Try web_search instead."

    return local_knowledge
