from __future__ import annotations

import json

from news_agent.models.config import OutletConfig
from news_agent.models.research import ResearchIntent
from news_agent.services.prompts.prompt_service import PromptService
from .job_planner import WebSearchJob


class OpenAIWebSearchPromptBuilder:
    """Build one prompt for one concrete OpenAI web-search job."""

    def __init__(self, prompt_service: PromptService) -> None:
        self.prompt_service = prompt_service

    def build(
        self,
        *,
        template_name: str,
        query: str,
        job: WebSearchJob,
        days_back: int,
        intent: ResearchIntent | None,
    ) -> str:
        prompt = self.prompt_service.build(
            template_name,
            outlet_limit=len(job.outlets),
            days_back=days_back,
            outlets_text=_outlets_text(job.outlets),
            planned_queries_json=json.dumps(
                [job.search_query],
                ensure_ascii=False,
                indent=2,
            ),
            query=query,
        )
        requested_answer = (
            intent.requested_metric
            if intent and intent.requested_metric
            else query
        )
        return _append_search_job_context(
            prompt=prompt,
            job=job,
            requested_answer=requested_answer,
        )


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
    """Append the concrete retrieval instruction chosen by Python."""
    domain_instruction = _domain_instruction(job)
    return (
        f"{prompt}\n\n"
        "Concrete search job:\n"
        f"{domain_instruction}"
        "- Treat the keyword wording as a starting point, not mandatory wording.\n"
        "- Rewrite broad or conversational wording into short headline-style search variants.\n"
        "- Use multiple internal web searches if useful.\n"
        "- Prefer outlet diversity in the returned candidates.\n"
        "- The returned `search_query` field must contain the actual search wording that found or best matches the article.\n"
        f"- search_query: {job.search_query}\n"
        f"- allowed_domains: {json.dumps(list(job.allowed_domains), ensure_ascii=False)}\n"
        f"- requested_answer: {requested_answer}\n"
        "- Return only candidates from the curated outlets listed in this prompt.\n"
    )


def _domain_instruction(job: WebSearchJob) -> str:
    """Describe domain control without nudging default jobs toward site: syntax."""
    if job.allowed_domains and "site:" not in job.search_query.lower():
        return (
            "- The OpenAI web_search API `allowed_domains` setting already controls which domains may be searched.\n"
            "- Do not add `site:` filters to internal searches for this job; use clean keyword/headline wording only.\n"
            "- Search only within these API-allowed domains.\n"
        )
    if "site:" in job.search_query.lower():
        return (
            "- Keep any existing `site:<domain>` restrictions from this legacy/experimental search job, but improve the surrounding keywords.\n"
        )
    return "- Use the outlet/domain restrictions from this search job.\n"


