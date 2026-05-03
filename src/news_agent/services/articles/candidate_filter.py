from __future__ import annotations

from dataclasses import asdict
import json
import logging

from news_agent.models.config import AppConfig
from news_agent.models.config import OutletConfig
from news_agent.models.research import ResearchIntent
from news_agent.models.triage import ArticleRecord
from news_agent.services.debug.debug_output import DebugOutput
from news_agent.services.prompts.prompt_service import PromptService
from news_agent.services.llm.text_generation import ModelGenerationError
from news_agent.services.llm.text_generation import ModelOutputError
from news_agent.services.llm.text_generation import TextGenerator
from news_agent.services.llm.text_generation import build_text_generator
from news_agent.services.llm.text_generation import extract_json_block


logger = logging.getLogger(__name__)


class CandidateFilter:
    """Rejects candidates that do not appear to contain the requested metric."""

    def __init__(
        self,
        config: AppConfig,
        prompt_service: PromptService | None = None,
        text_generator: TextGenerator | None = None,
        debug_output: DebugOutput | None = None,
    ) -> None:
        self.config = config
        self.prompt_service = prompt_service or PromptService()
        self.text_generator = text_generator or build_text_generator(
            config.model,
            model_id=config.model.model_id_for_step("candidate_filter"),
        )
        self.debug_output = debug_output

    def filter(
        self,
        query: str,
        outlet: OutletConfig,
        candidates: list[ArticleRecord],
        intent: ResearchIntent | None,
    ) -> list[ArticleRecord]:
        """Keep candidates that plausibly answer the requested metric."""
        if not candidates or intent is None:
            return candidates

        try:
            payload = self._filter_with_model(query, outlet, candidates, intent)
            accepted_indexes = _coerce_indexes(payload.get("accepted_indexes"), len(candidates))
            if not accepted_indexes:
                logger.info("candidate filter rejected all outlet=%r", outlet.name)
                return []
            logger.info(
                "candidate filter accepted outlet=%r indexes=%s",
                outlet.name,
                accepted_indexes,
            )
            return [candidates[index] for index in accepted_indexes]
        except ModelGenerationError:
            logger.exception("candidate filtering failed because model generation failed")
            raise
        except (ModelOutputError, json.JSONDecodeError, TypeError, ValueError):
            logger.exception("candidate filtering failed because model output was unusable")
            raise

    def _filter_with_model(
        self,
        query: str,
        outlet: OutletConfig,
        candidates: list[ArticleRecord],
        intent: ResearchIntent,
    ) -> dict:
        """Ask the research model which candidate indexes should stay."""
        candidate_payload = [
            {
                "index": index,
                "title": article.title,
                "published_at": article.published_at,
                "snippet": article.snippet,
                "article_text": article.article_text[:1200],
            }
            for index, article in enumerate(candidates)
        ]
        prompt = self.prompt_service.build(
            "candidate_filter",
            query=query,
            outlet_name=outlet.name,
            intent_json=json.dumps(asdict(intent), ensure_ascii=False, indent=2),
            candidate_lines_json=json.dumps(
                candidate_payload,
                ensure_ascii=False,
                indent=2,
            ),
        )
        debug_call = (
            self.debug_output.start_model_call(f"candidate_filter_{outlet.name}", prompt)
            if self.debug_output
            else None
        )
        try:
            result = self.text_generator.generate(prompt)
            if debug_call:
                debug_call.write_output(result.text)
            return json.loads(extract_json_block(result.text))
        except (ModelGenerationError, ModelOutputError, json.JSONDecodeError, TypeError, ValueError) as exc:
            if debug_call:
                debug_call.write_error(exc)
            raise


def _coerce_indexes(value: object, candidate_count: int) -> list[int]:
    if not isinstance(value, list):
        return []
    indexes: list[int] = []
    for item in value:
        try:
            index = int(item)
        except (TypeError, ValueError):
            continue
        if 0 <= index < candidate_count and index not in indexes:
            indexes.append(index)
    return indexes


