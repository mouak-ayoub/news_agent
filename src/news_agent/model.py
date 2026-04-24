from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import json
import os
import re
from typing import Any

from openai import OpenAI

from .config import ModelConfig


class ModelGenerationError(RuntimeError):
    pass


@dataclass(slots=True)
class UsageStats:
    input_tokens: int = 0
    output_tokens: int = 0
    web_search_calls: int = 0


@dataclass(slots=True)
class GenerationResult:
    text: str
    usage: UsageStats


class TextGenerator:
    def generate(self, prompt: str) -> GenerationResult:
        raise NotImplementedError


@dataclass(slots=True)
class OpenAIResponsesTextGenerator(TextGenerator):
    config: ModelConfig
    client: OpenAI = field(init=False)

    def __post_init__(self) -> None:
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise ModelGenerationError(
                f"Environment variable `{self.config.api_key_env}` is not set."
            )
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str) -> GenerationResult:
        try:
            response = self.client.responses.create(
                model=self.config.model_id,
                input=prompt,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens,
            )
        except Exception as exc:
            raise ModelGenerationError("OpenAI response generation failed.") from exc

        usage = UsageStats(
            input_tokens=_safe_int(getattr(response.usage, "input_tokens", 0)),
            output_tokens=_safe_int(getattr(response.usage, "output_tokens", 0)),
            web_search_calls=0,
        )
        return GenerationResult(text=response.output_text or "", usage=usage)


@dataclass(slots=True)
class StaticTextGenerator(TextGenerator):
    response_text: str

    def generate(self, prompt: str) -> GenerationResult:
        return GenerationResult(text=self.response_text, usage=UsageStats())


def extract_json_block(raw_text: str) -> str:
    fenced = re.search(r"```json\s*(\{.*\}|\[.*\])\s*```", raw_text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    bare = re.search(r"(\{.*\}|\[.*\])", raw_text, re.DOTALL)
    if bare:
        return bare.group(1)
    raise ModelGenerationError("Model output did not include JSON.")


def build_text_generator(config: ModelConfig) -> TextGenerator:
    if config.backend != "openai":
        raise ModelGenerationError(f"Unsupported backend: {config.backend}")
    return OpenAIResponsesTextGenerator(config=config)


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
