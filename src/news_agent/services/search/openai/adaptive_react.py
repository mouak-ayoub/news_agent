from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
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


TOP_CANDIDATE_OBSERVATION_LIMIT = 8
OBSERVED_ARTICLE_TEXT_LIMIT = 2000


@dataclass(frozen=True, slots=True)
class AdaptiveObservation:
    """Raw retrieval state for the LLM repair planner to interpret."""

    candidate_count: int
    outlet_counts: dict[str, int]
    distinct_outlet_count: int
    configured_outlets: list[dict[str, str]]
    top_candidates: list[dict[str, str]]
    previous_actions: list[dict[str, Any]]
    remaining_repair_actions: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_count": self.candidate_count,
            "outlet_counts": self.outlet_counts,
            "distinct_outlet_count": self.distinct_outlet_count,
            "configured_outlets": self.configured_outlets,
            "top_candidates": self.top_candidates,
            "previous_actions": self.previous_actions,
            "remaining_repair_actions": self.remaining_repair_actions,
        }


@dataclass(frozen=True, slots=True)
class RepairDecision:
    action: str
    reason: str
    search_query: str
    allowed_outlets: list[str]
    diagnosis: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "diagnosis": self.diagnosis,
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
        outlet_counts: dict[str, int] = {}
        for article in articles:
            outlet_counts[article.outlet_name] = (
                outlet_counts.get(article.outlet_name, 0) + 1
            )

        top_candidates = [
            {
                "outlet_name": article.outlet_name,
                "title": article.title,
                "url": article.url,
                "published_at": article.published_at or "",
                "snippet": article.snippet,
                "article_text": _observation_text(article.article_text),
                "search_query": article.search_query,
            }
            for article in articles[:TOP_CANDIDATE_OBSERVATION_LIMIT]
        ]

        return AdaptiveObservation(
            candidate_count=len(articles),
            outlet_counts=outlet_counts,
            distinct_outlet_count=len(outlet_counts),
            configured_outlets=[_outlet_payload(outlet) for outlet in outlets],
            top_candidates=top_candidates,
            previous_actions=[
                previous_action.to_dict()
                for previous_action in (previous_actions or [])
            ],
            remaining_repair_actions=max(0, int(remaining_repair_actions or 0)),
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
            decision = _decision_from_payload(payload)
            return _validated_decision(decision, outlets)
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

        valid_outlets = _valid_repair_outlets(
            allowed_outlets=decision.allowed_outlets,
            outlets=outlets,
        )
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

    diagnosis = _diagnosis_from_payload(payload.get("diagnosis", {}))
    action = str(payload.get("action", "finish")).strip().lower()
    if action not in {"finish", "search"}:
        return _finish(
            "Planner returned an unsupported action.",
            diagnosis=diagnosis,
        )

    reason = str(payload.get("reason", "")).strip()
    search_query = " ".join(str(payload.get("search_query", "")).split())
    allowed_outlets = _string_list(payload.get("allowed_outlets", []))

    if action == "finish":
        return RepairDecision(
            diagnosis=diagnosis,
            action="finish",
            reason=reason,
            search_query="",
            allowed_outlets=[],
        )
    if not search_query:
        return _finish(
            "Planner requested search without a search query.",
            diagnosis=diagnosis,
        )

    return RepairDecision(
        diagnosis=diagnosis,
        action="search",
        reason=reason,
        search_query=search_query,
        allowed_outlets=allowed_outlets,
    )


def _validated_decision(
    decision: RepairDecision,
    outlets: list[OutletConfig],
) -> RepairDecision:
    """Enforce configured outlets and fail closed before executing repair search."""
    if decision.action != "search":
        return decision

    valid_outlets = _valid_repair_outlets(
        allowed_outlets=decision.allowed_outlets,
        outlets=outlets,
    )
    if not valid_outlets:
        return _finish(
            "Planner requested search without valid configured outlets.",
            diagnosis=decision.diagnosis,
        )

    valid_names = [outlet.name for outlet in valid_outlets]
    if valid_names == decision.allowed_outlets:
        return decision

    return RepairDecision(
        diagnosis=decision.diagnosis,
        action=decision.action,
        reason=decision.reason,
        search_query=decision.search_query,
        allowed_outlets=valid_names,
    )


def _finish(
    reason: str,
    *,
    diagnosis: dict[str, Any] | None = None,
) -> RepairDecision:
    return RepairDecision(
        diagnosis=diagnosis or {},
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
        "notes": outlet.notes,
    }


def _diagnosis_from_payload(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        "quality": str(value.get("quality", "")).strip(),
        "is_outlet_diverse": (
            value.get("is_outlet_diverse")
            if isinstance(value.get("is_outlet_diverse"), bool)
            else False
        ),
        "is_answer_bearing": (
            value.get("is_answer_bearing")
            if isinstance(value.get("is_answer_bearing"), bool)
            else False
        ),
        "dominance_assessment": str(value.get("dominance_assessment", "")).strip(),
        "strong_candidates": _string_list(value.get("strong_candidates", [])),
        "weak_candidates": _string_list(value.get("weak_candidates", [])),
        "missing_useful_outlets": _string_list(
            value.get("missing_useful_outlets", [])
        ),
        "problem": str(value.get("problem", "")).strip(),
    }


def _string_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _valid_repair_outlets(
    *,
    allowed_outlets: list[str],
    outlets: list[OutletConfig],
) -> list[OutletConfig]:
    outlets_by_name = {outlet.name: outlet for outlet in outlets}
    valid_outlets: list[OutletConfig] = []
    seen: set[str] = set()
    for outlet_name in allowed_outlets:
        outlet = outlets_by_name.get(outlet_name)
        if outlet is None or outlet.name in seen:
            continue
        valid_outlets.append(outlet)
        seen.add(outlet.name)
    return valid_outlets


def _observation_text(value: str) -> str:
    normalized = str(value)
    if len(normalized) <= OBSERVED_ARTICLE_TEXT_LIMIT:
        return normalized
    return normalized[:OBSERVED_ARTICLE_TEXT_LIMIT].rstrip() + "..."


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
        lines.extend(["", *_diagnosis_lines(index, decision.diagnosis)])
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
            lines.extend(
                [
                    "",
                    *_observation_lines(
                        observation_index + 1,
                        observations[observation_index],
                    ),
                ]
            )

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
        f"Distinct outlet count: {observation.distinct_outlet_count}",
        "Outlet distribution:",
    ]
    lines.extend(
        f"- {outlet_name}: {count}"
        for outlet_name, count in observation.outlet_counts.items()
    )
    if not observation.outlet_counts:
        lines.append("- none")

    lines.append("Top candidates:")
    lines.extend(
        f"- {candidate.get('outlet_name', '')}: {candidate.get('title', '')}"
        for candidate in observation.top_candidates
    )
    if not observation.top_candidates:
        lines.append("- none")
    return lines


def _diagnosis_lines(index: int, diagnosis: dict[str, Any]) -> list[str]:
    if not diagnosis:
        return [f"Diagnosis {index}:", "No diagnosis provided."]
    lines = [
        f"Diagnosis {index}:",
        f"Quality: {diagnosis.get('quality', '') or 'unspecified'}",
        f"Outlet diverse: {diagnosis.get('is_outlet_diverse', False)}",
        f"Answer-bearing: {diagnosis.get('is_answer_bearing', False)}",
    ]
    dominance_assessment = str(diagnosis.get("dominance_assessment", "")).strip()
    problem = str(diagnosis.get("problem", "")).strip()
    if dominance_assessment:
        lines.append(f"Dominance: {dominance_assessment}")
    if problem:
        lines.append(f"Problem: {problem}")
    return lines
