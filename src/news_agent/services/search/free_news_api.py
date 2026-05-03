from __future__ import annotations

from datetime import datetime
from datetime import timedelta
import logging
import os
import time
from typing import Any
from urllib.parse import urlparse

import requests

from news_agent.models.config import AppConfig
from news_agent.models.research import ResearchIntent
from news_agent.models.research import SearchPlan
from news_agent.models.triage import ArticleRecord
from news_agent.services.debug.debug_output import DebugOutput
from news_agent.services.prompts.prompt_service import PromptService
from news_agent.services.llm.text_generation import ModelGenerationError
from news_agent.services.llm.text_generation import TextGenerator


logger = logging.getLogger(__name__)


class FreeNewsApiSearchClient:
    """FreeNewsApi provider: retrieve and normalize article candidates only."""

    def __init__(
        self,
        config: AppConfig,
        prompt_service: PromptService | None = None,
        text_generator: TextGenerator | None = None,
        debug_output: DebugOutput | None = None,
    ) -> None:
        self.config = config
        self.search_config = config.search
        self.base_url = (self.search_config.base_url or "https://api.freenewsapi.io").rstrip("/")
        _ = prompt_service, text_generator, debug_output
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.search_config.user_agent,
                "x-api-key": self._api_key(),
            }
        )

    def search_candidates(
        self,
        query: str,
        plan: SearchPlan | None = None,
        intent: ResearchIntent | None = None,
    ) -> list[ArticleRecord]:
        """Search the news API once and preserve each article's real publisher."""
        logger.info("freenewsapi search started query=%r", query)
        listing_items = self._collect_listing_items(query, plan)
        logger.info("freenewsapi listing candidates count=%d", len(listing_items))

        articles = self._fetch_article_details(listing_items)
        logger.info("freenewsapi detail candidates count=%d", len(articles))
        _ = intent
        for article in articles:
            logger.info(
                "freenewsapi candidate publisher=%r url=%s title=%r",
                article.outlet_name,
                article.url,
                article.title,
            )

        logger.info("freenewsapi search finished candidates=%d", len(articles))
        return articles

    def _collect_listing_items(
        self,
        query: str,
        plan: SearchPlan | None,
    ) -> list[dict[str, Any]]:
        """Run planned API searches and deduplicate lightweight result rows by UUID."""
        seen_uuids: set[str] = set()
        items: list[dict[str, Any]] = []
        for planned_query in _planned_queries(query, plan):
            params = self._news_params(planned_query)
            logger.info("freenewsapi global query query=%r", planned_query)
            response = self.session.get(
                f"{self.base_url}/v1/news",
                params=params,
                timeout=self.search_config.request_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            for item in payload.get("data", []):
                if not isinstance(item, dict):
                    continue
                uuid = str(item.get("uuid", "")).strip()
                if not uuid or uuid in seen_uuids:
                    continue
                seen_uuids.add(uuid)
                logger.info(
                    "freenewsapi listing candidate uuid=%s publisher=%r title=%r",
                    uuid,
                    item.get("publisher"),
                    item.get("title"),
                )
                items.append(item)
                if len(items) >= self.search_config.candidate_pool_size:
                    return items
            _respect_rate_limit()
        return items

    def _news_params(self, query: str) -> dict[str, str | int]:
        """Build FreeNewsApi listing params for recent English full-text search."""
        params: dict[str, str | int] = {
            "language": "en",
            "order_by": "archive",
            "page_size": self.search_config.candidate_pool_size,
            "q": query,
        }
        if self.search_config.days_back > 0:
            published_after = datetime.utcnow() - timedelta(
                days=self.search_config.days_back
            )
            params["published_after"] = published_after.strftime("%Y-%m-%dT%H:%M:%SZ")
        return params

    def _fetch_article_details(self, listing_items: list[dict[str, Any]]) -> list[ArticleRecord]:
        """Fetch full article bodies for listed UUIDs."""
        articles: list[ArticleRecord] = []
        for item in listing_items:
            uuid = str(item.get("uuid", "")).strip()
            if not uuid:
                continue
            response = self.session.get(
                f"{self.base_url}/v1/details",
                params={"uuid": uuid},
                timeout=self.search_config.request_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            data = payload.get("data", {})
            if not isinstance(data, dict):
                continue
            article = self._article_from_detail(data)
            if article is not None:
                logger.info(
                    "freenewsapi detail candidate uuid=%s publisher=%r url=%s title=%r",
                    uuid,
                    article.outlet_name,
                    article.url,
                    article.title,
                )
                articles.append(article)
            _respect_rate_limit()
        return articles

    def _article_from_detail(self, data: dict[str, Any]) -> ArticleRecord | None:
        """Normalize one FreeNewsApi details payload into the app article model."""
        title = str(data.get("title", "")).strip()
        body = str(data.get("body", "")).strip()
        if not title and not body:
            return None

        url = str(data.get("original_url") or data.get("url") or "").strip()
        publisher = str(data.get("publisher") or "Unknown publisher").strip()
        countries = data.get("countries", [])
        country = (
            ", ".join(str(country) for country in countries if str(country).strip())
            if isinstance(countries, list)
            else str(countries or "")
        )
        domain = _domain_from_url(url)
        return ArticleRecord(
            title=title,
            url=url,
            outlet_name=publisher,
            domain=domain,
            country=country or "Unknown",
            medium_type="news API / publisher",
            orientation="unknown",
            published_at=str(data.get("published_at") or ""),
            snippet=str(data.get("subtitle") or body[:500]),
            article_text=body,
            search_query="FreeNewsApi details",
        )

    def _api_key(self) -> str:
        env_name = self.search_config.api_key_env or "news_triage_codex_app"
        api_key = os.environ.get(env_name)
        if not api_key:
            raise ModelGenerationError(
                f"Environment variable `{env_name}` is not set for FreeNewsApi."
            )
        return api_key


def _planned_queries(query: str, plan: SearchPlan | None) -> list[str]:
    if plan and plan.queries:
        return plan.queries
    return [query]


def _respect_rate_limit() -> None:
    """FreeNewsApi documents a 2 req/sec limit; stay below it."""
    time.sleep(0.55)


def _domain_from_url(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


