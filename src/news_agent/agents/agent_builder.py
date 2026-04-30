from __future__ import annotations

from google.adk.agents import SequentialAgent

from .coordinator import CoordinatorAgent
from .researcher import ResearchAgent
from .summarizer import SummarizerAgent
from ..services.research import ResearchService
from ..services.summarization import SummarizationService


class AgentGraphBuilder:
    def __init__(
        self,
        research_service: ResearchService,
        summarization_service: SummarizationService,
    ) -> None:
        self.research_service = research_service
        self.summarization_service = summarization_service

    def build(self) -> CoordinatorAgent:
        research_agent = ResearchAgent(
            name="research_agent",
            description="Collects recent articles from curated outlets.",
            service=self.research_service,
        )
        summarizer_agent = SummarizerAgent(
            name="summarizer_agent",
            description="Synthesizes the final triage JSON and brief.",
            service=self.summarization_service,
        )
        pipeline = SequentialAgent(
            name="triage_pipeline",
            description="Runs research then summarization in sequence.",
            sub_agents=[research_agent, summarizer_agent],
        )
        return CoordinatorAgent(
            name="coordinator_agent",
            description="Owns the full ADK workflow and final result.",
            sub_agents=[pipeline],
        )
