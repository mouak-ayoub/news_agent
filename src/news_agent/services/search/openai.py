from __future__ import annotations

import json
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
from ..prompt_service import PromptService
from ..text_generation import ModelGenerationError
from ..text_generation import ModelOutputError
from ..text_generation import extract_json_block
from .article_selector import ArticleSelector


class OpenAIWebSearchClient:
    """OpenAI web-search provider: asks the model for article candidates, then selects one per outlet."""

    def __init__(
        self,
        config: AppConfig,
        prompt_service: PromptService | None = None,
    ) -> None:
        self.config = config
        self.search_config = config.search
        self.outlets = config.outlets
        self.prompt_service = prompt_service or PromptService()
        self.article_selector = ArticleSelector(
            config=config,
            prompt_service=self.prompt_service,
        )
        self.client: Any
        self.__post_init__()

    def __post_init__(self) -> None:
        """Create the OpenAI client once the configured API key is available."""
        api_key = os.environ.get(self.config.model.api_key_env)
        if not api_key:
            raise ModelGenerationError(
                f"Environment variable `{self.config.model.api_key_env}` is not set."
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
            "web_search_research",
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
        try:
            response: GenerationResult | Any = self.client.responses.create(
                model=self.config.model.research_model_id,
                tools=[{"type": "web_search"}],
                input=prompt,
                max_output_tokens=self.config.model.max_output_tokens,
                temperature=self.config.model.temperature,
            )
            data = json.loads(extract_json_block(response.output_text or "[]"))
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
            raise ModelOutputError("OpenAI web search returned unusable article JSON.") from exc
        except Exception as exc:
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
                continue
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
