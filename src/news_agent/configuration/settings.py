from __future__ import annotations

from dataclasses import dataclass

from ..models.config import AppConfig
from .validation import ConfigValidationError


@dataclass(frozen=True, slots=True)
class OpenAIWebSearchSettings:
    """Resolved OpenAI web-search runtime settings."""

    api_key_env: str
    model_id: str
    max_output_tokens: int
    temperature: float
    reasoning_effort: str = ""
    max_tool_calls: int = 1
    text_verbosity: str = "low"


def resolve_openai_web_search_settings(config: AppConfig) -> OpenAIWebSearchSettings:
    """Resolve OpenAI web-search settings without reading secret values."""
    api_key_env = config.search.api_key_env.strip()
    if not api_key_env and config.model.backend.strip().lower() == "openai":
        api_key_env = config.model.api_key_env.strip()
    if not api_key_env:
        raise ConfigValidationError(
            "OpenAI web search requires `search.api_key_env`, or `model.api_key_env` when `model.backend` is `openai`."
        )

    model_id = config.search.web_search_model_id.strip()
    if not model_id:
        raise ConfigValidationError(
            "OpenAI web search requires `search.web_search_model_id`."
        )

    return OpenAIWebSearchSettings(
        api_key_env=api_key_env,
        model_id=model_id,
        max_output_tokens=config.model.max_output_tokens,
        temperature=config.model.temperature,
        reasoning_effort=config.search.web_search_reasoning_effort,
        max_tool_calls=config.search.web_search_max_tool_calls,
        text_verbosity=config.search.web_search_text_verbosity,
    )
