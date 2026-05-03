from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

import requests

from news_agent.models.config import ModelConfig
from news_agent.models.generation import GenerationResult


logger = logging.getLogger(__name__)


class ModelGenerationError(RuntimeError):
    pass


class ModelOutputError(RuntimeError):
    pass


class TextGenerator:
    def generate(self, prompt: str) -> GenerationResult:
        raise NotImplementedError


class OpenAIResponsesTextGenerator(TextGenerator):
    def __init__(self, config: ModelConfig, model_id: str) -> None:
        self.config = config
        self.model_id = model_id
        self.client: Any
        self.__post_init__()

    def __post_init__(self) -> None:
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise ModelGenerationError(
                f"Environment variable `{self.config.api_key_env}` is not set."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ModelGenerationError(
                "The OpenAI package is not installed. Use backend `heuristic` or install the OpenAI dependency."
            ) from exc
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str) -> GenerationResult:
        request_kwargs: dict[str, Any] = {
            "model": self.model_id,
            "input": prompt,
            "max_output_tokens": self.config.max_output_tokens,
        }
        if openai_supports_temperature(self.model_id):
            request_kwargs["temperature"] = self.config.temperature
        try:
            response = self.client.responses.create(**request_kwargs)
        except Exception as exc:
            raise ModelGenerationError(
                f"OpenAI response generation failed for model `{self.model_id}`: {exc}"
            ) from exc

        return GenerationResult(text=response.output_text or "")


class StaticTextGenerator(TextGenerator):
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text

    def generate(self, prompt: str) -> GenerationResult:
        return GenerationResult(text=self.response_text)


class GeminiTextGenerator(TextGenerator):
    def __init__(self, config: ModelConfig, model_id: str) -> None:
        self.config = config
        self.model_id = model_id
        self.client: Any
        self.types: Any
        self.__post_init__()

    def __post_init__(self) -> None:
        api_key = _resolve_gemini_api_key(self.config)
        if not api_key:
            raise ModelGenerationError(
                f"Environment variable `{self.config.api_key_env or 'GEMINI_API_KEY'}` is not set."
            )
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ModelGenerationError(
                "The Google GenAI package is not installed. Install the Gemini SDK dependency or use another backend."
            ) from exc
        self.types = types
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=_gemini_timeout_ms(self.config.request_timeout_seconds),
            ),
        )

    def generate(self, prompt: str) -> GenerationResult:
        config_kwargs: dict[str, Any] = {
            "temperature": self.config.temperature,
            "response_mime_type": "application/json",
        }
        if self.config.max_output_tokens > 0:
            config_kwargs["max_output_tokens"] = self.config.max_output_tokens

        last_error: Exception | None = None
        retry_attempts = _gemini_retry_attempts(self.config)
        for attempt in range(1, retry_attempts + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=prompt,
                    config=self.types.GenerateContentConfig(**config_kwargs),
                )
                return GenerationResult(text=response.text or "")
            except Exception as exc:
                last_error = exc
                if not _is_transient_gemini_error(exc) or attempt == retry_attempts:
                    break
                retry_delay = _gemini_retry_delay_seconds(self.config, attempt)
                logger.warning(
                    "Gemini transient generation error model=%r attempt=%d/%d retrying_in=%.1fs: %s",
                    self.model_id,
                    attempt,
                    retry_attempts,
                    retry_delay,
                    exc,
                )
                if retry_delay > 0:
                    time.sleep(retry_delay)

        raise ModelGenerationError(
            f"Gemini generation failed for model `{self.model_id}`: {last_error}"
        ) from last_error


class OllamaTextGenerator(TextGenerator):
    def __init__(self, config: ModelConfig, model_id: str) -> None:
        self.config = config
        self.model_id = model_id
        self.session = requests.Session()

    def generate(self, prompt: str) -> GenerationResult:
        payload: dict[str, Any] = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": self.config.temperature,
            },
        }
        if self.config.max_output_tokens > 0:
            payload["options"]["num_predict"] = self.config.max_output_tokens

        url = self.config.base_url.rstrip("/") + "/api/generate"
        try:
            response = self.session.post(
                url,
                json=payload,
                timeout=self.config.request_timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise ModelGenerationError(
                f"Ollama generation failed for model `{self.model_id}`: {exc}"
            ) from exc

        return GenerationResult(text=str(data.get("response", "")))


def extract_json_block(raw_text: str) -> str:
    fenced = re.search(r"```json\s*(\{.*\}|\[.*\])\s*```", raw_text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    bare = re.search(r"(\{.*\}|\[.*\])", raw_text, re.DOTALL)
    if bare:
        return bare.group(1)
    raise ModelOutputError("Model output did not include JSON.")


def build_text_generator(config: ModelConfig, model_id: str | None = None) -> TextGenerator:
    selected_model_id = model_id or config.summary_model_id
    if config.backend == "heuristic":
        return StaticTextGenerator("not json")
    if config.backend == "gemini":
        return GeminiTextGenerator(config=config, model_id=selected_model_id)
    if config.backend == "ollama":
        return OllamaTextGenerator(config=config, model_id=selected_model_id)
    if config.backend != "openai":
        raise ModelGenerationError(f"Unsupported backend: {config.backend}")
    return OpenAIResponsesTextGenerator(config=config, model_id=selected_model_id)


def openai_supports_temperature(model_id: str) -> bool:
    """Return whether this OpenAI model family accepts a temperature parameter."""
    normalized = model_id.strip().lower()
    return not (
        normalized.startswith("gpt-5")
        or normalized.startswith("o1")
        or normalized.startswith("o3")
        or normalized.startswith("o4")
    )


def openai_supports_reasoning_effort(model_id: str) -> bool:
    """Return whether this OpenAI model family accepts reasoning.effort."""
    normalized = model_id.strip().lower()
    return normalized.startswith(("gpt-5", "o1", "o3", "o4"))


def _resolve_gemini_api_key(config: ModelConfig) -> str | None:
    if config.api_key_env:
        api_key = os.environ.get(config.api_key_env)
        if api_key:
            return api_key
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def _gemini_timeout_ms(timeout_seconds: int) -> int:
    """Gemini SDK timeout is milliseconds; API requires at least 10 seconds."""
    timeout_ms = int(timeout_seconds) * 1000
    return max(timeout_ms, 10_000)


def _gemini_retry_attempts(config: ModelConfig) -> int:
    """Return at least one Gemini attempt, including the first call."""
    return max(1, int(config.gemini_retry_attempts))


def _gemini_retry_delay_seconds(config: ModelConfig, attempt: int) -> float:
    """Linear retry delay before the next Gemini attempt."""
    return max(0.0, float(config.gemini_retry_backoff_seconds)) * attempt


def _is_transient_gemini_error(exc: Exception) -> bool:
    """Retry only provider-side temporary failures, not quota or bad requests."""
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and status_code in {500, 502, 503, 504}:
        return True
    text = str(exc)
    return any(marker in text for marker in ("500 INTERNAL", "502", "503", "504"))


