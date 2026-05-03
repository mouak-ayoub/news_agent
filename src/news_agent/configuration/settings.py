from __future__ import annotations

from dataclasses import dataclass

from news_agent.models.config import AppConfig
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
    use_allowed_domains: bool = True
    include_sources: bool = True
    tool_choice: str = "required"
    search_context_size: str = "medium"
    use_site_query_filters: bool = False


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
        use_allowed_domains=config.search.web_search_use_allowed_domains,
        include_sources=config.search.web_search_include_sources,
        tool_choice=config.search.web_search_tool_choice,
        search_context_size=config.search.web_search_search_context_size,
        use_site_query_filters=config.search.web_search_use_site_query_filters,
    )

