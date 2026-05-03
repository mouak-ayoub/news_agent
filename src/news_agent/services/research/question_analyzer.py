from __future__ import annotations

import json
import logging

from news_agent.models.config import AppConfig
from news_agent.models.research import ResearchIntent
from news_agent.services.debug.debug_output import DebugOutput
from news_agent.services.prompts.prompt_service import PromptService
from news_agent.services.llm.text_generation import ModelGenerationError
from news_agent.services.llm.text_generation import ModelOutputError
from news_agent.services.llm.text_generation import TextGenerator
from news_agent.services.llm.text_generation import extract_json_block


logger = logging.getLogger(__name__)


class QuestionAnalyzer:
    """Turns a raw user question into a reusable research intent."""

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

    def analyze(self, query: str) -> ResearchIntent:
        """Extract the topic, target metric, and obvious wrong-result patterns."""
        logger.info("question analysis started")
        prompt = self.prompt_service.build("question_analysis", query=query)
        debug_call = (
            self.debug_output.start_model_call("question_analysis", prompt)
            if self.debug_output
            else None
        )
        try:
            result = self.text_generator.generate(prompt)
            if debug_call:
                debug_call.write_output(result.text)
            payload = json.loads(extract_json_block(result.text))
            intent = ResearchIntent.from_dict(payload, fallback_query=query)
            logger.info(
                "question analysis finished topic=%r metric=%r",
                intent.topic,
                intent.requested_metric,
            )
            return intent
        except ModelGenerationError as exc:
            if debug_call:
                debug_call.write_error(exc)
            logger.exception("question analysis failed because model generation failed")
            raise
        except (ModelOutputError, json.JSONDecodeError, TypeError, ValueError) as exc:
            if debug_call:
                debug_call.write_error(exc)
            logger.exception("question analysis failed because model output was unusable")
            raise


