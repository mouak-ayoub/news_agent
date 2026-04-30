from __future__ import annotations

import asyncio

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from .agents.agent_builder import AgentGraphBuilder
from .models.config import AppConfig
from .models.triage import TriageBrief
from .services.article_content_fetcher import ArticleContentFetcher
from .services.metric_extractor import MetricExtractor
from .services.prompt_service import PromptService
from .services.query_planner import QueryPlanner
from .services.question_analyzer import QuestionAnalyzer
from .services.research import ResearchService
from .services.search import build_search_client
from .services.summarization import SummarizationService
from .services.text_generation import build_text_generator

APP_NAME = "agents"


def run_triage(
    query: str,
    config: AppConfig,
    *,
    research_service: ResearchService | None = None,
    summarization_service: SummarizationService | None = None,
) -> TriageBrief:
    prompt_service = PromptService()
    if research_service is None:
        research_text_generator = build_text_generator(
            config.model,
            model_id=config.model.research_model_id or config.model.summary_model_id,
        )
        research_service = ResearchService(
            client=build_search_client(config, prompt_service=prompt_service),
            question_analyzer=QuestionAnalyzer(
                config=config,
                text_generator=research_text_generator,
                prompt_service=prompt_service,
            ),
            query_planner=QueryPlanner(
                config=config,
                text_generator=research_text_generator,
                prompt_service=prompt_service,
            ),
            article_content_fetcher=ArticleContentFetcher(config),
            metric_extractor=MetricExtractor(
                config=config,
                text_generator=research_text_generator,
                prompt_service=prompt_service,
            ),
        )
    if summarization_service is None:
        summarization_service = SummarizationService(
            config=config,
            text_generator=build_text_generator(
                config.model,
                model_id=config.model.summary_model_id,
            ),
            prompt_service=prompt_service,
        )

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
