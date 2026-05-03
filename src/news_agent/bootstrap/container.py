from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace

from news_agent.models.config import AppConfig
from news_agent.services.analysis.analysis_service import AnalysisService
from news_agent.services.articles.article_content_fetcher import ArticleContentFetcher
from news_agent.services.articles.article_selector import ArticleSelector
from news_agent.services.articles.candidate_filter import CandidateFilter
from news_agent.services.debug.debug_output import DebugOutput
from news_agent.services.llm.text_generation import build_text_generator
from news_agent.services.prompts.prompt_service import PromptService
from news_agent.services.research import ResearchService
from news_agent.services.research import ResearchPipeline
from news_agent.services.research.metric_extractor import MetricExtractor
from news_agent.services.research.query_planner import QueryPlanner
from news_agent.services.research.question_analyzer import QuestionAnalyzer
from news_agent.services.research.steps import AnalyzeQuestionStep
from news_agent.services.research.steps import ApplyAnswerPolicyStep
from news_agent.services.research.steps import BuildResearchBundleStep
from news_agent.services.research.steps import EnrichCandidatesStep
from news_agent.services.research.steps import ExtractMetricsStep
from news_agent.services.research.steps import PlanQueriesStep
from news_agent.services.research.steps import RetrieveCandidatesStep
from news_agent.services.research.steps import SelectArticlesStep
from news_agent.services.search import build_search_client
from news_agent.services.summarization import SummarizationService


@dataclass(slots=True)
class ApplicationContainer:
    """Runtime services built from application configuration."""

    config: AppConfig
    prompt_service: PromptService
    research_service: ResearchService
    summarization_service: SummarizationService
    analysis_service: AnalysisService | None = None
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
    analysis_service = build_analysis_service(
        config=config,
        prompt_service=prompt_service,
        debug_output=debug_output,
    )

    return ApplicationContainer(
        config=config,
        prompt_service=prompt_service,
        research_service=research_service,
        summarization_service=summarization_service,
        analysis_service=analysis_service,
        debug_output=debug_output,
    )


def build_research_service(
    *,
    config: AppConfig,
    prompt_service: PromptService,
    debug_output: DebugOutput | None = None,
) -> ResearchService:
    """Build the research-side service graph."""
    search_client = build_search_client(
        config,
        prompt_service=prompt_service,
        debug_output=debug_output,
    )
    question_analyzer = QuestionAnalyzer(
        config=config,
        text_generator=build_text_generator(
            config.model,
            model_id=config.model.model_id_for_step("question_analysis"),
        ),
        prompt_service=prompt_service,
        debug_output=debug_output,
    )
    query_planner = QueryPlanner(
        config=config,
        text_generator=build_text_generator(
            config.model,
            model_id=config.model.model_id_for_step("query_planning"),
        ),
        prompt_service=prompt_service,
        debug_output=debug_output,
    )
    candidate_filter = CandidateFilter(
        config=config,
        prompt_service=prompt_service,
        text_generator=build_text_generator(
            config.model,
            model_id=config.model.model_id_for_step("candidate_filter"),
        ),
        debug_output=debug_output,
    )
    article_content_fetcher = ArticleContentFetcher(config)
    article_selector = ArticleSelector(
        config=config,
        prompt_service=prompt_service,
        text_generator=build_text_generator(
            config.model,
            model_id=config.model.model_id_for_step("article_selection"),
        ),
        candidate_filter=candidate_filter,
        debug_output=debug_output,
    )
    metric_extractor = MetricExtractor(
        config=config,
        text_generator=build_text_generator(
            config.model,
            model_id=config.model.model_id_for_step("metric_extraction"),
        ),
        prompt_service=prompt_service,
        debug_output=debug_output,
    )
    pipeline = ResearchPipeline(
        steps=[
            AnalyzeQuestionStep(question_analyzer),
            PlanQueriesStep(query_planner),
            RetrieveCandidatesStep(search_client),
            EnrichCandidatesStep(article_content_fetcher),
            SelectArticlesStep(
                article_selector=article_selector,
                outlets=config.outlets[: config.search.max_sources],
                max_articles=config.search.max_sources,
            ),
            BuildResearchBundleStep(),
            ExtractMetricsStep(metric_extractor),
            ApplyAnswerPolicyStep(),
        ]
    )
    return ResearchService(pipeline=pipeline)


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


def build_analysis_service(
    *,
    config: AppConfig,
    prompt_service: PromptService,
    debug_output: DebugOutput | None = None,
) -> AnalysisService | None:
    """Build the optional post-summary analysis service."""
    if not config.analysis.enabled:
        return None

    analysis_model_config = config.model
    if config.analysis.max_output_tokens > 0:
        analysis_model_config = replace(
            config.model,
            max_output_tokens=config.analysis.max_output_tokens,
        )
    return AnalysisService(
        prompt_service=prompt_service,
        text_generator=build_text_generator(
            analysis_model_config,
            model_id=analysis_model_config.model_id_for_step(
                config.analysis.model_step
            ),
        ),
        debug_output=debug_output,
        evidence_prompt=config.analysis.evidence_based_prompt,
        speculative_prompt=config.analysis.speculative_red_team_prompt,
        run_parallel=config.analysis.run_parallel,
    )
