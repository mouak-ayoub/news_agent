from __future__ import annotations

import json
import logging

from ..models.config import AppConfig
from ..models.research import ResearchIntent
from ..models.research import SearchPlan
from .debug_output import DebugOutput
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
        debug_output: DebugOutput | None = None,
    ) -> None:
        self.config = config
        self.text_generator = text_generator
        self.prompt_service = prompt_service or PromptService()
        self.debug_output = debug_output

    def plan(self, query: str, intent: ResearchIntent) -> SearchPlan:
        """Create metric-focused search queries from the analyzed intent."""
        logger.info("query planning started")
        prompt = self.prompt_service.build(
            "query_planning",
            query=query,
            intent_json=json.dumps(intent.to_dict(), ensure_ascii=False, indent=2),
        )
        debug_call = (
            self.debug_output.start_model_call("query_planning", prompt)
            if self.debug_output
            else None
        )
        try:
            result = self.text_generator.generate(prompt)
            if debug_call:
                debug_call.write_output(result.text)
            payload = json.loads(extract_json_block(result.text))
            plan = self._normalize_plan(payload, fallback_query=query)
            logger.info("query planning finished queries=%s", plan.queries)
            return plan
        except ModelGenerationError as exc:
            if debug_call:
                debug_call.write_error(exc)
            logger.exception("query planning failed because model generation failed")
            raise
        except (ModelOutputError, json.JSONDecodeError, TypeError, ValueError) as exc:
            if debug_call:
                debug_call.write_error(exc)
            logger.exception("query planning failed because model output was unusable")
            raise

    def _normalize_plan(self, payload: dict, fallback_query: str) -> SearchPlan:
        """Keep valid model queries and retain the original query as a fallback."""
        plan = SearchPlan.from_dict(payload, fallback_query=fallback_query)
        return SearchPlan(queries=_dedupe([fallback_query, *plan.queries])[:5])

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
