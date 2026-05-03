from __future__ import annotations

import json
import logging
import re
from typing import Any
from urllib.parse import urlparse

from news_agent.models.config import OutletConfig
from news_agent.models.triage import ArticleRecord


logger = logging.getLogger(__name__)


class OpenAIArticleNormalizer:
    """Convert OpenAI model-returned JSON into ArticleRecord objects."""

    def normalize(
        self,
        data: Any,
        *,
        allowed_outlets: tuple[OutletConfig, ...],
    ) -> list[ArticleRecord]:
        if not isinstance(data, list):
            return []

        articles: list[ArticleRecord] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            url = self.clean_article_url(str(item.get("url", "")))
            if not url.startswith(("http://", "https://")):
                logger.info(
                    "openai web search rejected invalid url=%r outlet_name=%r",
                    item.get("url"),
                    item.get("outlet_name"),
                )
                continue
            outlet = self.match_outlet(item, allowed_outlets=allowed_outlets)
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
                        url=url,
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
                        retrieval_metadata=self.retrieval_metadata(item),
                    )
                )
            except Exception:
                continue
        return articles

    def clean_article_url(self, value: str) -> str:
        """Extract a usable URL from raw or Markdown-formatted model output."""
        text = str(value).strip()
        markdown_match = re.search(r"\((https?://[^)]+)\)", text)
        if markdown_match:
            return markdown_match.group(1).strip()

        raw_match = re.search(r"https?://[^\s\])>]+", text)
        if raw_match:
            return raw_match.group(0).strip()
        return text

    def match_outlet(
        self,
        item: dict[str, Any],
        *,
        allowed_outlets: tuple[OutletConfig, ...],
    ) -> OutletConfig | None:
        """Accept only articles that match a configured outlet."""
        url = self.clean_article_url(str(item.get("url", "")))
        url_domain = self._clean_domain(urlparse(url).netloc)
        declared_domain = self._clean_domain(str(item.get("domain", "")))
        outlet_name = str(item.get("outlet_name", "")).strip().lower()

        for outlet in allowed_outlets:
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

    def retrieval_metadata(self, item: dict[str, Any]) -> dict[str, object]:
        """Keep compact prompt-engineering fields for debugging and ranking."""
        keys = (
            "answer_type",
            "requested_answer",
            "topic_match_score",
            "answer_match_score",
            "evidence_match_score",
            "metric_match_score",
            "recency_score",
            "consistency_votes",
            "selected_branch",
            "selection_reason",
        )
        return {
            key: item[key]
            for key in keys
            if key in item and item[key] is not None
        }

    def _log_candidate_scores(
        self,
        item: dict[str, Any],
        outlet: OutletConfig,
    ) -> None:
        score_keys = (
            "answer_type",
            "requested_answer",
            "topic_match_score",
            "answer_match_score",
            "evidence_match_score",
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

    def _clean_domain(self, value: str) -> str:
        """Normalize domains before comparing provider output with config."""
        return value.strip().lower().removeprefix("www.")



