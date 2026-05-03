from __future__ import annotations

import asyncio

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from news_agent.agents.agent_builder import AgentGraphBuilder
from news_agent.bootstrap.container import build_application_container
from news_agent.models.config import AppConfig
from news_agent.models.triage import TriageBrief
from news_agent.services.analysis.analysis_service import AnalysisService
from news_agent.services.debug.debug_output import DebugOutput
from news_agent.services.research import ResearchService
from news_agent.services.summarization import SummarizationService

APP_NAME = "agents"


def run_triage(
    query: str,
    config: AppConfig,
    *,
    debug_output: DebugOutput | None = None,
    research_service: ResearchService | None = None,
    summarization_service: SummarizationService | None = None,
    analysis_service: AnalysisService | None = None,
) -> TriageBrief:
    if (
        research_service is None
        or summarization_service is None
        or (analysis_service is None and config.analysis.enabled)
    ):
        container = build_application_container(
            config=config,
            debug_output=debug_output,
        )
        research_service = research_service or container.research_service
        summarization_service = (
            summarization_service or container.summarization_service
        )
        analysis_service = analysis_service or container.analysis_service

    session_service = InMemorySessionService()
    session = asyncio.run(
        session_service.create_session(
            app_name=APP_NAME,
            user_id="local_user",
            state={},
        )
    )
    agent = AgentGraphBuilder(
        research_service=research_service,
        summarization_service=summarization_service,
        analysis_service=analysis_service,
    ).build()
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
    if final_session is None:
        raise RuntimeError("The ADK workflow did not produce a final session.")

    workflow_error = final_session.state.get("workflow_error")
    if workflow_error:
        stage = final_session.state.get("workflow_error_stage", "unknown")
        raise RuntimeError(f"Workflow failed at stage `{stage}`: {workflow_error}")

    if "triage_brief" not in final_session.state:
        raise RuntimeError("The ADK workflow did not produce a triage brief.")
    return TriageBrief.from_dict(final_session.state["triage_brief"])
