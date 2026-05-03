from __future__ import annotations

from dataclasses import asdict
import json
import logging

from news_agent.models.config import AppConfig
from news_agent.models.research import MetricExtraction
from news_agent.models.research import ResearchIntent
from news_agent.models.triage import ArticleRecord
from news_agent.models.triage import ResearchBundle
from news_agent.services.debug.debug_output import DebugOutput
from news_agent.services.prompts.prompt_service import PromptService
from news_agent.services.llm.text_generation import ModelGenerationError
from news_agent.services.llm.text_generation import ModelOutputError
from news_agent.services.llm.text_generation import TextGenerator
from news_agent.services.llm.text_generation import extract_json_block


logger = logging.getLogger(__name__)


class MetricExtractor:
    """Extracts the requested number/count/range from selected articles."""

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
        prompt = self.prompt_service.build(
            "metric_extraction",
            query=query,
            intent_json=json.dumps(asdict(intent), ensure_ascii=False, indent=2),
            article_json=json.dumps(asdict(article), ensure_ascii=False, indent=2),
        )
        debug_call = (
            self.debug_output.start_model_call(
                f"metric_extraction_{article.outlet_name}",
                prompt,
            )
            if self.debug_output
            else None
        )
        try:
            result = self.text_generator.generate(prompt)
            if debug_call:
                debug_call.write_output(result.text)
            payload = json.loads(extract_json_block(result.text))
            return MetricExtraction.from_dict(payload)
        except ModelGenerationError as exc:
            if debug_call:
                debug_call.write_error(exc)
            logger.exception("metric extraction failed because model generation failed")
            raise
        except (ModelOutputError, json.JSONDecodeError, TypeError, ValueError) as exc:
            if debug_call:
                debug_call.write_error(exc)
            logger.exception("metric extraction failed because model output was unusable")
            raise


