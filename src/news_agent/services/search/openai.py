from __future__ import annotations

import json
import logging

from ...models.config import AppConfig
from ...models.research import ResearchIntent
from ...models.research import SearchPlan
from ...models.triage import ArticleRecord
from ...models.triage import ResearchBundle
from ..debug_output import DebugOutput
from ..prompt_service import PromptService
from ..text_generation import ModelGenerationError
from ..text_generation import ModelOutputError
from ..text_generation import extract_json_block
from .article_deduplicator import ArticleDeduplicator
from .article_selector import ArticleSelector
from .openai_article_normalizer import OpenAIArticleNormalizer
from .openai_gateway import OpenAIWebSearchGateway
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
        job_planner: OpenAISearchJobPlanner | None = None,
        prompt_builder: OpenAIWebSearchPromptBuilder | None = None,
        gateway: OpenAIWebSearchGateway | None = None,
        normalizer: OpenAIArticleNormalizer | None = None,
        deduplicator: ArticleDeduplicator | None = None,
        article_selector: ArticleSelector | None = None,
    ) -> None:
        self.config = config
        self.search_config = config.search
        self.outlets = config.outlets
        self.prompt_service = prompt_service or PromptService()
        self.debug_output = debug_output
        self.job_planner = job_planner or OpenAISearchJobPlanner()
        self.prompt_builder = prompt_builder or OpenAIWebSearchPromptBuilder(
            self.prompt_service
        )
        self.gateway = gateway or OpenAIWebSearchGateway(
            api_key_env=self._api_key_env()
        )
        self.normalizer = normalizer or OpenAIArticleNormalizer()
        self.deduplicator = deduplicator or ArticleDeduplicator()
        self.article_selector = article_selector or ArticleSelector(
            config=config,
            prompt_service=self.prompt_service,
            debug_output=debug_output,
        )

    def search(
        self,
        query: str,
        plan: SearchPlan | None = None,
        intent: ResearchIntent | None = None,
    ) -> ResearchBundle:
        """Run deterministic site-filtered jobs, then select one article per outlet."""
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
        return ResearchBundle(
            query=query,
            articles=self.article_selector.choose_one_per_outlet(
                query=query,
                outlets=target_outlets,
                candidates=articles,
                intent=intent,
            ),
            intent=intent,
            search_plan=plan,
        )

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
        debug_call = (
            self.debug_output.start_model_call(
                f"openai_web_search_{job_index:02d}",
                prompt,
            )
            if self.debug_output
            else None
        )
        try:
            if debug_call:
                debug_call.write_artifact(
                    "search_job.json",
                    json.dumps(
                        {
                            "search_query": job.search_query,
                            "outlets": [outlet.name for outlet in job.outlets],
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                )
            response = self.gateway.search(
                prompt=prompt,
                model_id=self._web_search_model_id(),
                max_output_tokens=self.config.model.max_output_tokens,
                temperature=self.config.model.temperature,
                reasoning_effort=self.search_config.web_search_reasoning_effort,
                max_tool_calls=self.search_config.web_search_max_tool_calls,
                text_verbosity=self.search_config.web_search_text_verbosity,
            )
            if debug_call:
                debug_call.write_artifact("response.json", response.response_dump)

            raw_output = response.raw_text
            if debug_call:
                debug_call.write_output(raw_output)
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
            if debug_call:
                debug_call.write_error(exc)
            raise ModelOutputError(
                f"OpenAI web search returned unusable article JSON for job {job_index}."
            ) from exc
        except Exception as exc:
            if debug_call:
                debug_call.write_error(exc)
            raise ModelGenerationError(
                f"OpenAI web search request failed for job {job_index}."
            ) from exc

    def _web_search_model_id(self) -> str:
        """Return the OpenAI model used only for web-search retrieval."""
        if self.search_config.web_search_model_id:
            return self.search_config.web_search_model_id
        raise ModelGenerationError(
            "OpenAI web search requires `search.web_search_model_id`."
        )

    def _api_key_env(self) -> str:
        """Return the API key env var for the OpenAI web-search provider."""
        if self.search_config.api_key_env:
            return self.search_config.api_key_env
        if self.config.model.backend == "openai" and self.config.model.api_key_env:
            return self.config.model.api_key_env
        raise ModelGenerationError(
            "OpenAI web search requires `search.api_key_env` when the main model backend is not OpenAI."
        )

