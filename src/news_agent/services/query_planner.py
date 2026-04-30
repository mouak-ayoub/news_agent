from __future__ import annotations

import json
import logging

from ..models.config import AppConfig
from ..models.research import ResearchIntent
from ..models.research import SearchPlan
from .prompt_service import PromptService
from .text_generation import ModelGenerationError
from .text_generation import ModelOutputError
from .text_generation import TextGenerator
from .text_generation import extract_json_block


logger = logging.getLogger(__name__)


class QueryPlanner:
    """Builds search phrases from intent instead of relying on the raw query."""

    def __init__(
        self,
        config: AppConfig,
        text_generator: TextGenerator,
        prompt_service: PromptService | None = None,
    ) -> None:
        self.config = config
        self.text_generator = text_generator
        self.prompt_service = prompt_service or PromptService()

    def plan(self, query: str, intent: ResearchIntent) -> SearchPlan:
        """Create metric-focused search queries from the analyzed intent."""
        logger.info("query planning started")
        try:
            result = self.text_generator.generate(
                self.prompt_service.build(
                    "query_planning",
                    query=query,
                    intent_json=json.dumps(intent.to_dict(), ensure_ascii=False, indent=2),
                )
            )
            payload = json.loads(extract_json_block(result.text))
            plan = self._normalize_plan(payload, fallback_query=query)
            logger.info("query planning finished queries=%s", plan.queries)
            return plan
        except ModelGenerationError:
            logger.exception("query planning failed because model generation failed")
            raise
        except (ModelOutputError, json.JSONDecodeError, TypeError, ValueError):
            if not self.config.fallback_to_heuristic:
                logger.exception("query planning failed because model output was unusable")
                raise
            plan = self._fallback_plan(query, intent)
            logger.warning("query planning output unusable; using heuristic plan")
            logger.info("query planning finished mode=heuristic queries=%s", plan.queries)
            return plan

    def _normalize_plan(self, payload: dict, fallback_query: str) -> SearchPlan:
        """Keep valid model queries and retain the original query as a fallback."""
        plan = SearchPlan.from_dict(payload, fallback_query=fallback_query)
        return SearchPlan(queries=_dedupe([fallback_query, *plan.queries])[:5])

    def _fallback_plan(self, query: str, intent: ResearchIntent) -> SearchPlan:
        """Build a simple query plan when model planning is unavailable."""
        terms = " ".join(intent.must_find[:3])
        metric = intent.requested_metric or query
        return SearchPlan(
            queries=_dedupe(
                [
                    query,
                    f"{intent.topic} {metric}",
                    f"{intent.topic} {terms}".strip(),
                    f"{query} latest figures",
                    f"{query} official numbers",
                ]
            )[:5]
        )


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = " ".join(str(value).split())
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result
