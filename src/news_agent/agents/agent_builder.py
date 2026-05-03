from __future__ import annotations

from google.adk.agents import SequentialAgent

from news_agent.agents.analysis import AnalysisAgent
from news_agent.agents.coordinator import CoordinatorAgent
from news_agent.agents.researcher import ResearchAgent
from news_agent.agents.summarizer import SummarizerAgent
from news_agent.services.analysis.analysis_service import AnalysisService
from news_agent.services.research import ResearchService
from news_agent.services.summarization import SummarizationService


class AgentGraphBuilder:
    def __init__(
        self,
        research_service: ResearchService,
        summarization_service: SummarizationService,
        analysis_service: AnalysisService | None = None,
    ) -> None:
        self.research_service = research_service
        self.summarization_service = summarization_service
        self.analysis_service = analysis_service

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
        sub_agents = [research_agent, summarizer_agent]
        if self.analysis_service is not None:
            sub_agents.append(
                AnalysisAgent(
                    name="analysis_agent",
                    description="Adds post-summary evidence-based and red-team analysis.",
                    service=self.analysis_service,
                )
            )
        pipeline = SequentialAgent(
            name="triage_pipeline",
            description="Runs research, summarization, then optional analysis in sequence.",
            sub_agents=sub_agents,
        )
        return CoordinatorAgent(
            name="coordinator_agent",
            description="Owns the full ADK workflow and final result.",
            sub_agents=[pipeline],
        )
