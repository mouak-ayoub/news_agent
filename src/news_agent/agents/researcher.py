from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import json
import os
from typing import Any
from urllib.parse import urlparse

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.genai import types
from openai import OpenAI
from typing_extensions import override

from ..config import AppConfig
from ..model import extract_json_block
from ..model import ModelGenerationError
from ..model import UsageStats
from ..schemas import ArticleRecord
from ..schemas import ResearchBundle
from ..usage import UsageGuard
from ..usage import BudgetExceededError


def _extract_query(user_content: types.Content | None) -> str:
    if not user_content or not user_content.parts:
        return ""
    return " ".join(part.text or "" for part in user_content.parts).strip()


@dataclass(slots=True)
class ResearchService:
    config: AppConfig
    usage_guard: UsageGuard
    client: OpenAI = field(init=False)

    def __post_init__(self) -> None:
        api_key = os.environ.get(self.config.model.api_key_env)
        if not api_key:
            raise ModelGenerationError(
                f"Environment variable `{self.config.model.api_key_env}` is not set."
            )
        self.client = OpenAI(api_key=api_key)

    def research(self, query: str) -> ResearchBundle:
        outlet_limit = min(self.config.search.max_sources, len(self.config.outlets))
        outlets_text = "\n".join(
            f"- {outlet.name} | domain={outlet.domain} | country={outlet.country} | "
            f"type={outlet.medium_type} | orientation={outlet.orientation}"
            for outlet in self.config.outlets
        )

        prompt = f"""
Return JSON only.

Find up to {outlet_limit} recent articles across the curated outlets below.
Return at most one article per outlet.

Requirements:
- use only the curated outlets below
- prefer the last {self.config.search.days_back} days when possible, but if needed you may use a slightly older directly relevant article instead of returning nothing
- partial relevance is allowed when it directly answers an important part of the query
- do not require one article to answer every sub-part perfectly
- for example:
  - for casualty questions, an article about Iran-only casualties is relevant
  - an article about Iran and Qatar is also relevant
  - an article covering all countries is ideal but not required
- for other geopolitical questions, an article addressing one major actor, territory, legal status, or diplomatic position is relevant
- skip only outlets with no directly relevant article
- keep `article_text` to one concise sentence
- only return an article if the URL hostname belongs to the same outlet domain
- return a JSON array and nothing else

For the object return exactly:
- title
- url
- outlet_name
- domain
- country
- medium_type
- orientation
- published_at
- snippet
- article_text
- search_query

Curated outlets:
{outlets_text}

Query:
{query}
""".strip()
        try:
            response = self.client.responses.create(
                model=self.config.model.research_model_id,
                tools=[{"type": "web_search"}],
                input=prompt,
                max_output_tokens=self.config.model.max_output_tokens,
                temperature=self.config.model.temperature,
            )
            usage = UsageStats(
                input_tokens=int(getattr(response.usage, "input_tokens", 0) or 0),
                output_tokens=int(getattr(response.usage, "output_tokens", 0) or 0),
                web_search_calls=1,
            )
            self.usage_guard.record("research", usage)
            data = json.loads(extract_json_block(response.output_text or "[]"))
            articles = self._normalize_articles(data)
            return ResearchBundle(
                query=query,
                articles=self._dedupe_articles(articles)[:outlet_limit],
            )
        except (BudgetExceededError, ModelGenerationError):
            raise
        except Exception:
            return ResearchBundle(query=query, articles=[])

    def _normalize_articles(self, data: Any) -> list[ArticleRecord]:
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

    def _match_outlet(self, item: dict[str, Any]):
        url_domain = self._clean_domain(urlparse(str(item.get("url", ""))).netloc)
        declared_domain = self._clean_domain(str(item.get("domain", "")))
        outlet_name = str(item.get("outlet_name", "")).strip().lower()

        for outlet in self.config.outlets:
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
            if not url_domain and not declared_domain and outlet_name and outlet_name == outlet.name.strip().lower():
                return outlet
        return None

    def _clean_domain(self, value: str) -> str:
        return value.strip().lower().removeprefix("www.")

    def _dedupe_articles(self, articles: list[ArticleRecord]) -> list[ArticleRecord]:
        seen_outlets: set[str] = set()
        deduped: list[ArticleRecord] = []
        for article in articles:
            if article.outlet_name in seen_outlets:
                continue
            seen_outlets.add(article.outlet_name)
            deduped.append(article)
        return deduped


class ResearchAgent(BaseAgent):
    service: Any

    @override
    async def _run_async_impl(self, ctx: InvocationContext):
        query = _extract_query(ctx.user_content)
        bundle = self.service.research(query)
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            actions=EventActions(
                state_delta={
                    "query": query,
                    "research_bundle": bundle.to_dict(),
                }
            ),
            content=types.Content(
                role="model",
                parts=[types.Part(text=f"ResearchAgent gathered {len(bundle.articles)} article(s).")],
            ),
        )
