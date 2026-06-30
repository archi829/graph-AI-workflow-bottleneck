from .base import AgentSystem, FailureInjectionConfig, RunResult
from .crewai_agent import CrewAIAgent

# Add to this registry as more systems come online:
#   open_deep_research_agent.OpenDeepResearchAgent  (Day 2-3)
#   finrobot_agent.FinRobotAgent                     (whoever picks it up)
REGISTRY: dict[str, type[AgentSystem]] = {
    "crewai": CrewAIAgent,
}

__all__ = ["AgentSystem", "FailureInjectionConfig", "RunResult", "REGISTRY"]
