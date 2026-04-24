from __future__ import annotations

import asyncio

from google.adk.agents import SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from .agents.coordinator import CoordinatorAgent
from .agents.researcher import ResearchAgent
from .agents.researcher import ResearchService
from .agents.summarizer import SummarizationService
from .agents.summarizer import SummarizerAgent
from .config import AppConfig
from .model import build_text_generator
from .schemas import TriageBrief
from .usage import UsageGuard

APP_NAME = "agents"


def build_agent_graph(
    config: AppConfig,
    *,
    research_service: ResearchService | None = None,
    summarization_service: SummarizationService | None = None,
) -> CoordinatorAgent:
    usage_guard = UsageGuard(config)
    research_service = research_service or ResearchService(config, usage_guard=usage_guard)
    summarization_service = summarization_service or SummarizationService(
        config=config,
        text_generator=build_text_generator(config.model),
        usage_guard=usage_guard,
    )

    research_agent = ResearchAgent(
        name="research_agent",
        description="Collects recent articles from curated outlets.",
        service=research_service,
    )
    summarizer_agent = SummarizerAgent(
        name="summarizer_agent",
        description="Synthesizes the final triage JSON and brief.",
        service=summarization_service,
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


def run_triage(
    query: str,
    config: AppConfig,
    *,
    research_service: ResearchService | None = None,
    summarization_service: SummarizationService | None = None,
) -> TriageBrief:
    session_service = InMemorySessionService()
    session = asyncio.run(
        session_service.create_session(
            app_name=APP_NAME,
            user_id="local_user",
            state={},
        )
    )
    agent = build_agent_graph(
        config,
        research_service=research_service,
        summarization_service=summarization_service,
    )
    runner = Runner(
        app_name=APP_NAME,
        agent=agent,
        session_service=session_service,
    )

    user_message = types.Content(role="user", parts=[types.Part(text=query)])
    for _ in runner.run(
        user_id="local_user",
        session_id=session.id,
        new_message=user_message,
    ):
        pass

    final_session = asyncio.run(
        session_service.get_session(
            app_name=APP_NAME,
            user_id="local_user",
            session_id=session.id,
        )
    )
    if final_session is None or "triage_brief" not in final_session.state:
        raise RuntimeError("The ADK workflow did not produce a triage brief.")
    return TriageBrief.from_dict(final_session.state["triage_brief"])
