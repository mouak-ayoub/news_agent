from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Any

from news_agent.models.config import AppConfig
from news_agent.models.config import OutletConfig
from news_agent.models.research import ResearchIntent
from news_agent.models.research import SearchPlan
from news_agent.models.triage import ArticleRecord
from news_agent.services.debug.debug_output import DebugOutput
from news_agent.services.llm.text_generation import ModelGenerationError
from news_agent.services.llm.text_generation import ModelOutputError
from news_agent.services.llm.text_generation import TextGenerator
from news_agent.services.llm.text_generation import extract_json_block
from news_agent.services.prompts.prompt_service import PromptService
from .domain_utils import normalize_allowed_domain
from .job_planner import WebSearchJob


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AdaptiveObservation:
    candidate_count: int
    outlet_counts: dict[str, int]
    configured_outlets: list[str]
    missing_configured_outlets: list[str]
    dominant_outlet: str
    top_candidates: list[dict[str, str]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_count": self.candidate_count,
            "outlet_counts": self.outlet_counts,
            "configured_outlets": self.configured_outlets,
            "missing_configured_outlets": self.missing_configured_outlets,
            "dominant_outlet": self.dominant_outlet,
            "top_candidates": self.top_candidates,
        }


@dataclass(frozen=True, slots=True)
class RepairDecision:
    action: str
    reason: str
    search_query: str
    allowed_outlets: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "search_query": self.search_query,
            "allowed_outlets": self.allowed_outlets,
        }


class AdaptiveReactRepairPlanner:
    """Plan bounded focused OpenAI web-search repair jobs."""

    def __init__(
        self,
        *,
        config: AppConfig,
        prompt_service: PromptService,
        text_generator: TextGenerator | None,
        debug_output: DebugOutput | None = None,
    ) -> None:
        self.config = config
        self.prompt_service = prompt_service
        self.text_generator = text_generator
        self.debug_output = debug_output

    def build_observation(
        self,
        *,
        articles: list[ArticleRecord],
        outlets: list[OutletConfig],
        previous_actions: list[RepairDecision] | None = None,
        remaining_repair_actions: int | None = None,
    ) -> AdaptiveObservation:
        _ = previous_actions, remaining_repair_actions
        outlet_counts: dict[str, int] = {}
        for article in articles:
            outlet_counts[article.outlet_name] = (
                outlet_counts.get(article.outlet_name, 0) + 1
            )

        configured_outlets = [outlet.name for outlet in outlets]
        missing_configured_outlets = [
            outlet_name
            for outlet_name in configured_outlets
            if outlet_counts.get(outlet_name, 0) == 0
        ]
        dominant_outlet = ""
        if outlet_counts:
            dominant_outlet = max(
                outlet_counts.items(),
                key=lambda item: item[1],
            )[0]

        top_candidates = [
            {
                "outlet_name": article.outlet_name,
                "title": article.title,
                "url": article.url,
                "published_at": article.published_at or "",
                "search_query": article.search_query,
            }
            for article in articles[:8]
        ]

        return AdaptiveObservation(
            candidate_count=len(articles),
            outlet_counts=outlet_counts,
            configured_outlets=configured_outlets,
            missing_configured_outlets=missing_configured_outlets,
            dominant_outlet=dominant_outlet,
            top_candidates=top_candidates,
        )

    def decide(
        self,
        *,
        query: str,
        plan: SearchPlan | None,
        intent: ResearchIntent | None,
        observation: AdaptiveObservation,
        outlets: list[OutletConfig],
        previous_actions: list[RepairDecision] | None = None,
        remaining_repair_actions: int = 0,
    ) -> RepairDecision:
        if self.text_generator is None:
            return _finish("Adaptive repair planner is not configured.")

        try:
            prompt = self.prompt_service.build(
                self.config.search.adaptive_react_repair_prompt,
                query=query,
                outlets_json=json.dumps(
                    [_outlet_payload(outlet) for outlet in outlets],
                    ensure_ascii=False,
                    indent=2,
                ),
                planned_queries_json=json.dumps(
                    plan.queries if plan else [query],
                    ensure_ascii=False,
                    indent=2,
                ),
                intent_json=json.dumps(
                    intent.to_dict() if intent else {},
                    ensure_ascii=False,
                    indent=2,
                ),
                observation_json=json.dumps(
                    observation.to_dict(),
                    ensure_ascii=False,
                    indent=2,
                ),
                previous_actions_json=json.dumps(
                    [
                        previous_action.to_dict()
                        for previous_action in (previous_actions or [])
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                remaining_repair_actions=max(0, int(remaining_repair_actions)),
            )
            debug_call = (
                self.debug_output.start_model_call(
                    "adaptive_react_repair_planner",
                    prompt,
                )
                if self.debug_output
                else None
            )
            result = self.text_generator.generate(prompt)
            if debug_call:
                debug_call.write_output(result.text)
            payload = json.loads(extract_json_block(result.text))
            return _decision_from_payload(payload)
        except (ModelGenerationError, ModelOutputError, json.JSONDecodeError) as exc:
            logger.info("adaptive repair planner skipped after model error: %s", exc)
            if "debug_call" in locals() and debug_call:
                debug_call.write_error(exc)
            return _finish("Adaptive repair planner failed; returning first candidates.")
        except (TypeError, ValueError, KeyError) as exc:
            logger.info("adaptive repair planner skipped after invalid output: %s", exc)
            if "debug_call" in locals() and debug_call:
                debug_call.write_error(exc)
            return _finish("Adaptive repair planner returned invalid output.")

    def build_repair_job(
        self,
        *,
        decision: RepairDecision,
        outlets: list[OutletConfig],
    ) -> WebSearchJob | None:
        if decision.action != "search":
            return None

        outlets_by_name = {outlet.name: outlet for outlet in outlets}
        valid_outlets: list[OutletConfig] = []
        seen: set[str] = set()
        for outlet_name in decision.allowed_outlets:
            outlet = outlets_by_name.get(outlet_name)
            if outlet is None or outlet.name in seen:
                continue
            valid_outlets.append(outlet)
            seen.add(outlet.name)

        if not valid_outlets:
            return None

        allowed_domains = tuple(
            domain
            for domain in (
                normalize_allowed_domain(outlet.domain) for outlet in valid_outlets
            )
            if domain
        )
        if not allowed_domains:
            return None

        return WebSearchJob(
            search_query=decision.search_query,
            outlets=tuple(valid_outlets),
            allowed_domains=allowed_domains,
        )

    def cap_per_outlet(self, articles: list[ArticleRecord]) -> list[ArticleRecord]:
        max_per_outlet = max(
            1,
            int(self.config.search.adaptive_react_max_candidates_per_outlet),
        )
        seen_counts: dict[str, int] = {}
        result: list[ArticleRecord] = []
        for article in articles:
            count = seen_counts.get(article.outlet_name, 0)
            if count >= max_per_outlet:
                continue
            result.append(article)
            seen_counts[article.outlet_name] = count + 1
        return result

    def write_trace(
        self,
        *,
        observations: list[AdaptiveObservation],
        decisions: list[RepairDecision],
        repair_jobs: list[WebSearchJob | None],
        final_articles: list[ArticleRecord],
    ) -> None:
        if not self.debug_output:
            return

        self.debug_output.write_json(
            "adaptive_react_observation.json",
            [observation.to_dict() for observation in observations],
        )
        self.debug_output.write_json(
            "adaptive_react_decision.json",
            [
                {
                    **decision.to_dict(),
                    "valid_repair_outlets": (
                        [outlet.name for outlet in repair_job.outlets]
                        if repair_job
                        else []
                    ),
                    "valid_repair_allowed_domains": (
                        list(repair_job.allowed_domains) if repair_job else []
                    ),
                }
                for decision, repair_job in zip(decisions, repair_jobs)
            ],
        )
        self.debug_output.write_text(
            "adaptive_react_trace.txt",
            _trace_text(
                observations=observations,
                decisions=decisions,
                repair_jobs=repair_jobs,
                final_articles=final_articles,
            ),
        )


def _decision_from_payload(payload: object) -> RepairDecision:
    if not isinstance(payload, dict):
        return _finish("Planner did not return a JSON object.")

    action = str(payload.get("action", "finish")).strip().lower()
    if action not in {"finish", "search"}:
        return _finish("Planner returned an unsupported action.")

    reason = str(payload.get("reason", "")).strip()
    search_query = " ".join(str(payload.get("search_query", "")).split())
    allowed_outlets = _string_list(payload.get("allowed_outlets", []))

    if action == "finish":
        return RepairDecision(
            action="finish",
            reason=reason,
            search_query="",
            allowed_outlets=[],
        )
    if not search_query:
        return _finish("Planner requested search without a search query.")

    return RepairDecision(
        action="search",
        reason=reason,
        search_query=search_query,
        allowed_outlets=allowed_outlets,
    )


def _finish(reason: str) -> RepairDecision:
    return RepairDecision(
        action="finish",
        reason=reason,
        search_query="",
        allowed_outlets=[],
    )


def _outlet_payload(outlet: OutletConfig) -> dict[str, str]:
    return {
        "name": outlet.name,
        "domain": outlet.domain,
        "country": outlet.country,
        "medium_type": outlet.medium_type,
        "orientation": outlet.orientation,
    }


def _string_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _trace_text(
    *,
    observations: list[AdaptiveObservation],
    decisions: list[RepairDecision],
    repair_jobs: list[WebSearchJob | None],
    final_articles: list[ArticleRecord],
) -> str:
    lines = [
        "> Entering adaptive ReAct retrieval chain...",
        "",
        "Action 1: GlobalSearch",
    ]
    if observations:
        lines.extend(_observation_lines(1, observations[0]))

    for index, decision in enumerate(decisions, start=2):
        repair_job = repair_jobs[index - 2] if index - 2 < len(repair_jobs) else None
        lines.extend(
            [
                "",
                f"Reason {index}:",
                decision.reason or "No reason provided.",
                "",
            ]
        )
        if decision.action == "search" and repair_job is not None:
            lines.extend(
                [
                    f"Action {index}: Search",
                    "Action Input:",
                    decision.search_query,
                    "Allowed outlets:",
                ]
            )
            lines.extend(f"- {outlet.name}" for outlet in repair_job.outlets)
        else:
            lines.append(f"Action {index}: Finish")

        observation_index = index - 1
        if observation_index < len(observations):
            lines.extend(["", *_observation_lines(observation_index + 1, observations[observation_index])])

    final_counts: dict[str, int] = {}
    for article in final_articles:
        final_counts[article.outlet_name] = final_counts.get(article.outlet_name, 0) + 1

    lines.extend(
        [
            "",
            "Final:",
            f"Final candidate count: {len(final_articles)}",
            "Final outlet distribution:",
        ]
    )
    lines.extend(f"- {name}: {count}" for name, count in final_counts.items())
    if not final_counts:
        lines.append("- none")
    return "\n".join(lines).strip() + "\n"


def _observation_lines(index: int, observation: AdaptiveObservation) -> list[str]:
    lines = [
        f"Observation {index}:",
        f"Candidate count: {observation.candidate_count}",
        "Outlet distribution:",
    ]
    lines.extend(
        f"- {outlet_name}: {count}"
        for outlet_name, count in observation.outlet_counts.items()
    )
    if not observation.outlet_counts:
        lines.append("- none")

    lines.append("Missing configured outlets:")
    lines.extend(f"- {name}" for name in observation.missing_configured_outlets)
    if not observation.missing_configured_outlets:
        lines.append("- none")

    lines.append(f"Dominant outlet: {observation.dominant_outlet or 'none'}")
    return lines
