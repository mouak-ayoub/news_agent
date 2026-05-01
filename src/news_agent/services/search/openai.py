from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
import re
from typing import Any
from urllib.parse import urlparse

from ...models.config import AppConfig
from ...models.config import OutletConfig
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


@dataclass(frozen=True, slots=True)
class WebSearchJob:
    """Provider-local OpenAI search job, not shared pipeline state.
    Move to a shared model only if other providers need the same abstraction."""

    search_query: str
    outlets: tuple[OutletConfig, ...]


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
        """Run deterministic site-filtered search jobs, then normalize returned candidates."""
        outlet_limit = min(self.search_config.max_sources, len(self.outlets))
        target_outlets = self.outlets[:outlet_limit]
        jobs = _build_search_jobs(
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
                    plan=plan,
                    intent=intent,
                    job=job,
                    job_index=index,
                )
            )

        articles = _dedupe_articles(articles)
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
        plan: SearchPlan | None,
        intent: ResearchIntent | None,
        job: WebSearchJob,
        job_index: int,
    ) -> list[ArticleRecord]:
        """Execute one concrete search job and parse its article JSON."""
        prompt = self.prompt_service.build(
            self.search_config.web_search_prompt,
            outlet_limit=len(job.outlets),
            days_back=self.search_config.days_back,
            outlets_text=_outlets_text(job.outlets),
            planned_queries_json=json.dumps(
                [job.search_query],
                ensure_ascii=False,
                indent=2,
            ),
            query=query,
        )
        prompt = _append_search_job_context(
            prompt=prompt,
            job=job,
            requested_answer=(
                intent.requested_metric
                if intent and intent.requested_metric
                else query
            ),
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
            web_search_model_id = self._web_search_model_id()
            request_kwargs: dict[str, Any] = {
                "model": web_search_model_id,
                "tools": [{"type": "web_search"}],
                "input": prompt,
                "max_output_tokens": self.config.model.max_output_tokens,
            }
            if openai_supports_temperature(web_search_model_id):
                request_kwargs["temperature"] = self.config.model.temperature
            response: Any = self.client.responses.create(**request_kwargs)
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
            articles = self._normalize_articles(data, allowed_outlets=job.outlets)
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

    def _normalize_articles(
        self,
        data: Any,
        *,
        allowed_outlets: tuple[OutletConfig, ...],
    ) -> list[ArticleRecord]:
        """Convert provider JSON into internal article records."""
        if not isinstance(data, list):
            return []

        articles: list[ArticleRecord] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            url = _clean_article_url(str(item.get("url", "")))
            if not url.startswith(("http://", "https://")):
                logger.info(
                    "openai web search rejected invalid url=%r outlet_name=%r",
                    item.get("url"),
                    item.get("outlet_name"),
                )
                continue
            outlet = self._match_outlet(item, allowed_outlets=allowed_outlets)
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
                        retrieval_metadata=_retrieval_metadata(item),
                    )
                )
            except Exception:
                continue
        return articles

    def _log_candidate_scores(self, item: dict[str, Any], outlet: OutletConfig) -> None:
        """Log prompt-engineering score fields when a prompt variant returns them."""
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

    def _match_outlet(
        self,
        item: dict[str, Any],
        *,
        allowed_outlets: tuple[OutletConfig, ...],
    ) -> OutletConfig | None:
        """Accept only articles that match a configured outlet."""
        url = _clean_article_url(str(item.get("url", "")))
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


def _build_search_jobs(
    *,
    query: str,
    plan: SearchPlan | None,
    outlets: list[OutletConfig],
    max_calls: int,
) -> list[WebSearchJob]:
    """Build exact site-filtered search queries before calling OpenAI web search."""
    if not outlets:
        return []

    call_limit = max(1, max_calls)
    planned_queries = _planned_queries(query, plan)
    jobs: list[WebSearchJob] = [
        WebSearchJob(
            search_query=_scoped_query(planned_queries[0], outlets),
            outlets=tuple(outlets),
        )
    ]
    if len(jobs) >= call_limit:
        return jobs

    seen_queries: set[str] = set()
    seen_queries.add(jobs[0].search_query)

    outlet_groups = _outlet_groups(outlets, call_limit - len(jobs))
    for planned_query in planned_queries:
        for group in outlet_groups:
            scoped_query = _scoped_query(planned_query, group)
            if scoped_query in seen_queries:
                continue
            jobs.append(
                WebSearchJob(
                    search_query=scoped_query,
                    outlets=tuple(group),
                )
            )
            seen_queries.add(scoped_query)
            if len(jobs) >= call_limit:
                return jobs
    return jobs


def _outlet_groups(outlets: list[OutletConfig], group_limit: int) -> list[list[OutletConfig]]:
    """Split outlets according to the configured search-call budget."""
    group_count = min(max(1, group_limit), len(outlets))
    base_size = len(outlets) // group_count
    remainder = len(outlets) % group_count

    groups: list[list[OutletConfig]] = []
    start = 0
    for index in range(group_count):
        size = base_size + (1 if index < remainder else 0)
        groups.append(outlets[start : start + size])
        start += size
    return groups


def _planned_queries(query: str, plan: SearchPlan | None) -> list[str]:
    """Prefer planner keyword queries and keep the raw question only as a later fallback."""
    values: list[str] = []
    if plan:
        values.extend(plan.queries)
    values.append(query)

    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = " ".join(str(value).split())
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        deduped.append(normalized)
        seen.add(key)

    if not deduped:
        deduped = [query]
    return sorted(deduped, key=_looks_like_question)


def _looks_like_question(value: str) -> bool:
    lowered = value.strip().lower()
    return lowered.endswith("?") or lowered.startswith(
        (
            "what ",
            "why ",
            "how ",
            "when ",
            "where ",
            "who ",
            "which ",
            "is ",
            "are ",
            "do ",
            "does ",
            "did ",
            "can ",
            "could ",
        )
    )


def _scoped_query(query: str, outlets: list[OutletConfig]) -> str:
    """Add explicit outlet domain filters to one keyword query."""
    return f"{query} {_domain_filter(outlets)}".strip()


def _domain_filter(outlets: list[OutletConfig]) -> str:
    return " OR ".join(f"site:{outlet.domain}" for outlet in outlets)


def _outlets_text(outlets: tuple[OutletConfig, ...]) -> str:
    return "\n".join(
        f"- {outlet.name} | domain={outlet.domain} | country={outlet.country} | "
        f"type={outlet.medium_type} | orientation={outlet.orientation}"
        for outlet in outlets
    )


def _append_search_job_context(
    *,
    prompt: str,
    job: WebSearchJob,
    requested_answer: str,
) -> str:
    """Append the concrete retrieval instruction that Python, not the model, chose."""
    return (
        f"{prompt}\n\n"
        "Concrete search job:\n"
        "- Use this exact search query as the primary web-search query.\n"
        f"- search_query: {job.search_query}\n"
        f"- requested_answer: {requested_answer}\n"
        "- Return only candidates from the curated outlets listed in this prompt.\n"
    )


def _dedupe_articles(articles: list[ArticleRecord]) -> list[ArticleRecord]:
    """Keep the first copy of each URL across overlapping search jobs."""
    deduped: list[ArticleRecord] = []
    seen: set[str] = set()
    for article in articles:
        key = article.url.strip().lower() or f"{article.outlet_name}:{article.title}".lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(article)
    return deduped


def _clean_article_url(value: str) -> str:
    """Extract a usable URL from raw or Markdown-formatted model output."""
    text = str(value).strip()
    markdown_match = re.search(r"\((https?://[^)]+)\)", text)
    if markdown_match:
        return markdown_match.group(1).strip()

    raw_match = re.search(r"https?://[^\s\])>]+", text)
    if raw_match:
        return raw_match.group(0).strip()
    return text


def _retrieval_metadata(item: dict[str, Any]) -> dict[str, object]:
    """Keep compact prompt-engineering fields for debugging and downstream ranking."""
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
