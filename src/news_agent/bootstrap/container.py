from __future__ import annotations

from dataclasses import dataclass

from news_agent.models.config import AppConfig
from news_agent.services.articles.article_content_fetcher import ArticleContentFetcher
from news_agent.services.articles.article_selector import ArticleSelector
from news_agent.services.articles.candidate_filter import CandidateFilter
from news_agent.services.debug.debug_output import DebugOutput
from news_agent.services.llm.text_generation import build_text_generator
from news_agent.services.prompts.prompt_service import PromptService
from news_agent.services.research import ResearchService
from news_agent.services.research.metric_extractor import MetricExtractor
from news_agent.services.research.query_planner import QueryPlanner
from news_agent.services.research.question_analyzer import QuestionAnalyzer
from news_agent.services.search import build_search_client
from news_agent.services.summarization import SummarizationService


@dataclass(slots=True)
class ApplicationContainer:
    """Runtime services built from application configuration."""

    config: AppConfig
    prompt_service: PromptService
    research_service: ResearchService
    summarization_service: SummarizationService
    debug_output: DebugOutput | None = None


def build_application_container(
    config: AppConfig,
    debug_output: DebugOutput | None = None,
) -> ApplicationContainer:
    """Build the application service graph."""
    prompt_service = PromptService()
    research_service = build_research_service(
        config=config,
        prompt_service=prompt_service,
        debug_output=debug_output,
    )
    summarization_service = build_summarization_service(
        config=config,
        prompt_service=prompt_service,
        debug_output=debug_output,
    )

    return ApplicationContainer(
        config=config,
        prompt_service=prompt_service,
        research_service=research_service,
        summarization_service=summarization_service,
        debug_output=debug_output,
    )


def build_research_service(
    *,
    config: AppConfig,
    prompt_service: PromptService,
    debug_output: DebugOutput | None = None,
) -> ResearchService:
    """Build the research-side service graph."""
    candidate_filter = CandidateFilter(
        config=config,
        prompt_service=prompt_service,
        text_generator=build_text_generator(
            config.model,
            model_id=config.model.model_id_for_step("candidate_filter"),
        ),
        debug_output=debug_output,
    )
    return ResearchService(
        client=build_search_client(
            config,
            prompt_service=prompt_service,
            debug_output=debug_output,
        ),
        question_analyzer=QuestionAnalyzer(
            config=config,
            text_generator=build_text_generator(
                config.model,
                model_id=config.model.model_id_for_step("question_analysis"),
            ),
            prompt_service=prompt_service,
            debug_output=debug_output,
        ),
        query_planner=QueryPlanner(
            config=config,
            text_generator=build_text_generator(
                config.model,
                model_id=config.model.model_id_for_step("query_planning"),
            ),
            prompt_service=prompt_service,
            debug_output=debug_output,
        ),
        article_content_fetcher=ArticleContentFetcher(config),
        article_selector=ArticleSelector(
            config=config,
            prompt_service=prompt_service,
            text_generator=build_text_generator(
                config.model,
                model_id=config.model.model_id_for_step("article_selection"),
            ),
            candidate_filter=candidate_filter,
            debug_output=debug_output,
        ),
        metric_extractor=MetricExtractor(
            config=config,
            text_generator=build_text_generator(
                config.model,
                model_id=config.model.model_id_for_step("metric_extraction"),
            ),
            prompt_service=prompt_service,
            debug_output=debug_output,
        ),
        outlets=config.outlets[: config.search.max_sources],
        max_articles=config.search.max_sources,
    )


def build_summarization_service(
    *,
    config: AppConfig,
    prompt_service: PromptService,
    debug_output: DebugOutput | None = None,
) -> SummarizationService:
    """Build the summarization service."""
    return SummarizationService(
        config=config,
        text_generator=build_text_generator(
            config.model,
            model_id=config.model.model_id_for_step("summarization"),
        ),
        prompt_service=prompt_service,
        debug_output=debug_output,
    )

