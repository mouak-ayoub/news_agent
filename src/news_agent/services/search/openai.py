from __future__ import annotations

import json
import logging

from ...configuration.settings import OpenAIWebSearchSettings
from ...configuration.settings import resolve_openai_web_search_settings
from ...models.config import AppConfig
from ...models.research import ResearchIntent
from ...models.research import SearchPlan
from ...models.triage import ArticleRecord
from ..debug_output import DebugOutput
from ..prompt_service import PromptService
from ..text_generation import ModelGenerationError
from ..text_generation import ModelOutputError
from ..text_generation import extract_json_block
from .article_deduplicator import ArticleDeduplicator
from .openai_article_normalizer import OpenAIArticleNormalizer
from .openai_gateway import DebuggingOpenAIWebSearchGateway
from .openai_gateway import OpenAIWebSearchGateway
from .openai_gateway import OpenAIWebSearchRequest
from .openai_job_planner import OpenAISearchJobPlanner
from .openai_job_planner import WebSearchJob
from .openai_prompt_builder import OpenAIWebSearchPromptBuilder


logger = logging.getLogger(__name__)


class OpenAIWebSearchClient:
    """OpenAI web-search provider orchestrator."""

    def __init__(
        self,
        config: AppConfig,
        prompt_service: PromptService | None = None,
        debug_output: DebugOutput | None = None,
        settings: OpenAIWebSearchSettings | None = None,
        job_planner: OpenAISearchJobPlanner | None = None,
        prompt_builder: OpenAIWebSearchPromptBuilder | None = None,
        gateway: OpenAIWebSearchGateway | DebuggingOpenAIWebSearchGateway | None = None,
        normalizer: OpenAIArticleNormalizer | None = None,
        deduplicator: ArticleDeduplicator | None = None,
    ) -> None:
        self.config = config
        self.search_config = config.search
        self.settings = settings or resolve_openai_web_search_settings(config)
        self.outlets = config.outlets
        self.prompt_service = prompt_service or PromptService()
        self.debug_output = debug_output
        self.job_planner = job_planner or OpenAISearchJobPlanner()
        self.prompt_builder = prompt_builder or OpenAIWebSearchPromptBuilder(
            self.prompt_service
        )
        if gateway is None:
            inner_gateway = OpenAIWebSearchGateway(api_key_env=self.settings.api_key_env)
            self.gateway = (
                DebuggingOpenAIWebSearchGateway(inner_gateway, debug_output)
                if debug_output
                else inner_gateway
            )
        else:
            self.gateway = gateway
        self.normalizer = normalizer or OpenAIArticleNormalizer()
        self.deduplicator = deduplicator or ArticleDeduplicator()

    def search_candidates(
        self,
        query: str,
        plan: SearchPlan | None = None,
        intent: ResearchIntent | None = None,
    ) -> list[ArticleRecord]:
        """Run deterministic site-filtered jobs and return normalized candidates."""
        outlet_limit = min(self.search_config.max_sources, len(self.outlets))
        target_outlets = self.outlets[:outlet_limit]
        jobs = self.job_planner.build_jobs(
            query=query,
            plan=plan,
            outlets=target_outlets,
            max_calls=self.search_config.max_search_calls_per_run,
        )
        logger.info(
            "openai web search started query=%r outlets=%d jobs=%d",
            query,
            len(target_outlets),
            len(jobs),
        )

        articles: list[ArticleRecord] = []
        for index, job in enumerate(jobs, start=1):
            articles.extend(
                self._run_search_job(
                    query=query,
                    intent=intent,
                    job=job,
                    job_index=index,
                )
            )

        articles = self.deduplicator.dedupe(articles)
        logger.info(
            "openai web search finished candidates=%d",
            len(articles),
        )
        return articles

    def _run_search_job(
        self,
        *,
        query: str,
        intent: ResearchIntent | None,
        job: WebSearchJob,
        job_index: int,
    ) -> list[ArticleRecord]:
        """Execute one concrete search job and parse its article JSON."""
        prompt = self.prompt_builder.build(
            template_name=self.search_config.web_search_prompt,
            query=query,
            job=job,
            days_back=self.search_config.days_back,
            intent=intent,
        )
        try:
            response = self.gateway.search(
                OpenAIWebSearchRequest(
                    call_name=f"openai_web_search_{job_index:02d}",
                    prompt=prompt,
                    search_query=job.search_query,
                    outlet_names=tuple(outlet.name for outlet in job.outlets),
                    model_id=self.settings.model_id,
                    max_output_tokens=self.settings.max_output_tokens,
                    temperature=self.settings.temperature,
                    reasoning_effort=self.settings.reasoning_effort,
                    max_tool_calls=self.settings.max_tool_calls,
                    text_verbosity=self.settings.text_verbosity,
                )
            )
            raw_output = response.raw_text
            data = json.loads(extract_json_block(raw_output))
            articles = self.normalizer.normalize(
                data,
                allowed_outlets=job.outlets,
            )
            logger.info(
                "openai web search job finished index=%d candidates=%d query=%r",
                job_index,
                len(articles),
                job.search_query,
            )
            return articles
        except ModelGenerationError:
            raise
        except (ModelOutputError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ModelOutputError(
                f"OpenAI web search returned unusable article JSON for job {job_index}."
            ) from exc
        except Exception as exc:
            raise ModelGenerationError(
                f"OpenAI web search request failed for job {job_index}."
            ) from exc
