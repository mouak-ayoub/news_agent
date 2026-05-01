from __future__ import annotations

import json
import logging
import os
from typing import Any
from urllib.parse import urlparse

from ...models.config import AppConfig
from ...models.config import OutletConfig
from ...models.generation import GenerationResult
from ...models.research import ResearchIntent
from ...models.research import SearchPlan
from ...models.triage import ArticleRecord
from ...models.triage import ResearchBundle
from ..debug_output import DebugOutput
from ..prompt_service import PromptService
from ..text_generation import ModelGenerationError
from ..text_generation import ModelOutputError
from ..text_generation import extract_json_block
from ..text_generation import openai_supports_temperature
from .article_selector import ArticleSelector


logger = logging.getLogger(__name__)


class OpenAIWebSearchClient:
    """OpenAI web-search provider: asks the model for article candidates, then selects one per outlet."""

    def __init__(
        self,
        config: AppConfig,
        prompt_service: PromptService | None = None,
        debug_output: DebugOutput | None = None,
    ) -> None:
        self.config = config
        self.search_config = config.search
        self.outlets = config.outlets
        self.prompt_service = prompt_service or PromptService()
        self.debug_output = debug_output
        self.article_selector = ArticleSelector(
            config=config,
            prompt_service=self.prompt_service,
            debug_output=debug_output,
        )
        self.client: Any
        self.__post_init__()

    def __post_init__(self) -> None:
        """Create the OpenAI client once the configured API key is available."""
        api_key_env = self._api_key_env()
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ModelGenerationError(
                f"Environment variable `{api_key_env}` is not set."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ModelGenerationError(
                "The OpenAI package is not installed. Use provider `google_news_rss` or install the OpenAI dependency."
            ) from exc
        self.client = OpenAI(api_key=api_key)

    def search(
        self,
        query: str,
        plan: SearchPlan | None = None,
        intent: ResearchIntent | None = None,
    ) -> ResearchBundle:
        """Run one outlet-aware web search prompt and normalize returned candidates."""
        outlet_limit = min(self.search_config.max_sources, len(self.outlets))
        target_outlets = self.outlets[:outlet_limit]
        outlets_text = "\n".join(
            f"- {outlet.name} | domain={outlet.domain} | country={outlet.country} | "
            f"type={outlet.medium_type} | orientation={outlet.orientation}"
            for outlet in target_outlets
        )
        prompt = self.prompt_service.build(
            self.search_config.web_search_prompt,
            outlet_limit=outlet_limit,
            days_back=self.search_config.days_back,
            outlets_text=outlets_text,
            planned_queries_json=json.dumps(
                (plan.queries if plan and plan.queries else [query]),
                ensure_ascii=False,
                indent=2,
            ),
            query=query,
        )
        debug_call = (
            self.debug_output.start_model_call("openai_web_search", prompt)
            if self.debug_output
            else None
        )
        try:
            web_search_model_id = self._web_search_model_id()
            request_kwargs: dict[str, Any] = {
                "model": web_search_model_id,
                "tools": [{"type": "web_search"}],
                "input": prompt,
                "max_output_tokens": self.config.model.max_output_tokens,
            }
            if openai_supports_temperature(web_search_model_id):
                request_kwargs["temperature"] = self.config.model.temperature
            response: GenerationResult | Any = self.client.responses.create(**request_kwargs)
            response_dump = _serialize_openai_response(response)
            if debug_call:
                debug_call.write_artifact("response.json", response_dump)

            raw_output = _extract_openai_response_text(response)
            if not raw_output.strip():
                raise ModelOutputError(
                    "OpenAI web search returned an empty final text response."
                )
            if debug_call:
                debug_call.write_output(raw_output)
            data = json.loads(extract_json_block(raw_output))
            articles = self._normalize_articles(data)
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
        except ModelGenerationError:
            raise
        except (ModelOutputError, json.JSONDecodeError, TypeError, ValueError) as exc:
            if debug_call:
                debug_call.write_error(exc)
            raise ModelOutputError("OpenAI web search returned unusable article JSON.") from exc
        except Exception as exc:
            if debug_call:
                debug_call.write_error(exc)
            raise ModelGenerationError("OpenAI web search request failed.") from exc

    def _normalize_articles(self, data: Any) -> list[ArticleRecord]:
        """Convert provider JSON into internal article records."""
        if not isinstance(data, list):
            return []

        articles: list[ArticleRecord] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            outlet = self._match_outlet(item)
            if outlet is None:
                logger.info(
                    "openai web search rejected unconfigured outlet url=%r outlet_name=%r",
                    item.get("url"),
                    item.get("outlet_name"),
                )
                continue
            self._log_candidate_scores(item, outlet)
            try:
                articles.append(
                    ArticleRecord(
                        title=str(item.get("title", "")),
                        url=str(item.get("url", "")),
                        outlet_name=outlet.name,
                        domain=outlet.domain,
                        country=outlet.country,
                        medium_type=outlet.medium_type,
                        orientation=outlet.orientation,
                        published_at=(
                            str(item.get("published_at"))
                            if item.get("published_at") is not None
                            else None
                        ),
                        snippet=str(item.get("snippet", "")),
                        article_text=str(item.get("article_text", "")),
                        search_query=str(item.get("search_query", "")),
                    )
                )
            except Exception:
                continue
        return articles

    def _log_candidate_scores(self, item: dict[str, Any], outlet: OutletConfig) -> None:
        """Log prompt-engineering score fields when a prompt variant returns them."""
        score_keys = (
            "topic_match_score",
            "metric_match_score",
            "recency_score",
            "consistency_votes",
            "selected_branch",
            "selection_reason",
        )
        scores = {key: item.get(key) for key in score_keys if key in item}
        if scores:
            logger.info(
                "openai web search candidate outlet=%r title=%r scores=%s",
                outlet.name,
                item.get("title"),
                json.dumps(scores, ensure_ascii=False),
            )

    def _match_outlet(self, item: dict[str, Any]) -> OutletConfig | None:
        """Accept only articles that match a configured outlet."""
        url_domain = self._clean_domain(urlparse(str(item.get("url", ""))).netloc)
        declared_domain = self._clean_domain(str(item.get("domain", "")))
        outlet_name = str(item.get("outlet_name", "")).strip().lower()

        for outlet in self.outlets:
            canonical_domain = self._clean_domain(outlet.domain)
            if url_domain and (
                url_domain == canonical_domain
                or url_domain.endswith("." + canonical_domain)
            ):
                return outlet
            if not url_domain and declared_domain and (
                declared_domain == canonical_domain
                or declared_domain.endswith("." + canonical_domain)
            ):
                return outlet
            if (
                not url_domain
                and not declared_domain
                and outlet_name
                and outlet_name == outlet.name.strip().lower()
            ):
                return outlet
        return None

    def _clean_domain(self, value: str) -> str:
        """Normalize domains before comparing provider output with config."""
        return value.strip().lower().removeprefix("www.")

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


def _extract_openai_response_text(response: Any) -> str:
    """Read final assistant text from the Responses object without hiding empties."""
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)

    chunks: list[str] = []
    for output_item in _field(response, "output", []) or []:
        for content_item in _field(output_item, "content", []) or []:
            text = _field(content_item, "text", None)
            if text:
                chunks.append(str(text))
    return "\n".join(chunks)


def _serialize_openai_response(response: Any) -> str:
    """Serialize the full provider response so debug can show tool calls and status."""
    if hasattr(response, "model_dump_json"):
        return str(response.model_dump_json(indent=2))
    return json.dumps(response, ensure_ascii=False, indent=2, default=str)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)
