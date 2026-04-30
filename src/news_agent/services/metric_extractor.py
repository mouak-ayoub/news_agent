from __future__ import annotations

from dataclasses import asdict
import json
import logging
import re

from ..models.config import AppConfig
from ..models.research import MetricExtraction
from ..models.research import ResearchIntent
from ..models.triage import ArticleRecord
from ..models.triage import ResearchBundle
from .prompt_service import PromptService
from .text_generation import ModelGenerationError
from .text_generation import ModelOutputError
from .text_generation import TextGenerator
from .text_generation import extract_json_block


NUMBER_PATTERN = re.compile(
    r"\b\d[\d,]*(?:\.\d+)?(?:\s?(?:k|m|b|million|billion|thousand|percent|%))?\b",
    re.IGNORECASE,
)

logger = logging.getLogger(__name__)


class MetricExtractor:
    """Extracts the requested number/count/range from selected articles."""

    def __init__(
        self,
        config: AppConfig,
        text_generator: TextGenerator,
        prompt_service: PromptService | None = None,
    ) -> None:
        self.config = config
        self.text_generator = text_generator
        self.prompt_service = prompt_service or PromptService()

    def enrich_bundle(self, bundle: ResearchBundle) -> ResearchBundle:
        """Attach metric extraction fields to every selected article."""
        if bundle.intent is None:
            return bundle
        logger.info("metric extraction started articles=%d", len(bundle.articles))
        for article in bundle.articles:
            extraction = self.extract(bundle.query, bundle.intent, article)
            article.metric_found = extraction.metric_found
            article.metric_value = extraction.value
            article.metric_type = extraction.metric_type
            article.metric_evidence = extraction.evidence
            article.metric_confidence = extraction.confidence
            article.metric_notes = extraction.notes
        logger.info("metric extraction finished")
        return bundle

    def extract(
        self,
        query: str,
        intent: ResearchIntent,
        article: ArticleRecord,
    ) -> MetricExtraction:
        """Extract the requested metric from one article, not just any number."""
        try:
            result = self.text_generator.generate(
                self.prompt_service.build(
                    "metric_extraction",
                    query=query,
                    intent_json=json.dumps(asdict(intent), ensure_ascii=False, indent=2),
                    article_json=json.dumps(asdict(article), ensure_ascii=False, indent=2),
                )
            )
            payload = json.loads(extract_json_block(result.text))
            return MetricExtraction.from_dict(payload)
        except ModelGenerationError:
            logger.exception("metric extraction failed because model generation failed")
            raise
        except (ModelOutputError, json.JSONDecodeError, TypeError, ValueError):
            if not self.config.fallback_to_heuristic:
                logger.exception("metric extraction failed because model output was unusable")
                raise
            logger.warning("metric extraction output unusable; using heuristic extraction")
            return self._fallback_extract(article)

    def _fallback_extract(self, article: ArticleRecord) -> MetricExtraction:
        """Use plain number extraction when model extraction fails."""
        text = " ".join(
            part for part in (article.title, article.snippet, article.article_text) if part
        )
        numbers = _dedupe(NUMBER_PATTERN.findall(text))
        if not numbers:
            return MetricExtraction(metric_found=False, notes="No explicit number found.")
        return MetricExtraction(
            metric_found=True,
            value=", ".join(numbers[:5]),
            metric_type="number mentioned in selected article",
            evidence=_trim(text),
            confidence="low",
            notes="Heuristic extraction found numbers but did not verify metric meaning.",
        )


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = " ".join(value.split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _trim(text: str, limit: int = 260) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."
