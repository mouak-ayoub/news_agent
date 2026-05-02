from __future__ import annotations

import json

from ...models.config import OutletConfig
from ...models.research import ResearchIntent
from ..prompt_service import PromptService
from .openai_job_planner import WebSearchJob


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
    return (
        f"{prompt}\n\n"
        "Concrete search job:\n"
        "- Use this exact search query as the primary web-search query.\n"
        f"- search_query: {job.search_query}\n"
        f"- requested_answer: {requested_answer}\n"
        "- Return only candidates from the curated outlets listed in this prompt.\n"
    )

