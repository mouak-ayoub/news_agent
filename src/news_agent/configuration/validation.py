from __future__ import annotations

from news_agent.models.config import AppConfig


class ConfigValidationError(ValueError):
    """Raised when app configuration is structurally invalid."""


class AppConfigValidator:
    """Validate loaded app configuration before runtime wiring."""

    def validate(self, config: AppConfig) -> None:
        self._validate_general(config)
        self._validate_provider(config)

    def _validate_general(self, config: AppConfig) -> None:
        if config.search.max_sources <= 0:
            raise ConfigValidationError("`search.max_sources` must be greater than 0.")
        if config.search.max_search_calls_per_run <= 0:
            raise ConfigValidationError(
                "`search.max_search_calls_per_run` must be greater than 0."
            )
        if config.search.candidate_pool_size <= 0:
            raise ConfigValidationError(
                "`search.candidate_pool_size` must be greater than 0."
            )
        if config.search.days_back <= 0:
            raise ConfigValidationError("`search.days_back` must be greater than 0.")
        if config.model.max_output_tokens < 0:
            raise ConfigValidationError(
                "`model.max_output_tokens` must be greater than or equal to 0."
            )
        if config.analysis.max_output_tokens < 0:
            raise ConfigValidationError(
                "`analysis.max_output_tokens` must be greater than or equal to 0."
            )
        if not config.outlets:
            raise ConfigValidationError("Config must define at least one outlet.")

        self._validate_web_search_controls(config)

        for index, outlet in enumerate(config.outlets, start=1):
            if not outlet.name.strip():
                raise ConfigValidationError(
                    f"`outlets[{index}].name` must not be empty."
                )
            if not outlet.domain.strip():
                raise ConfigValidationError(
                    f"`outlets[{index}].domain` must not be empty."
                )

    def _validate_web_search_controls(self, config: AppConfig) -> None:
        tool_choice = config.search.web_search_tool_choice.strip().lower()
        allowed_tool_choices = {"", "auto", "required", "none"}
        if tool_choice not in allowed_tool_choices:
            raise ConfigValidationError(
                "`search.web_search_tool_choice` must be one of: "
                + ", ".join(repr(value) for value in sorted(allowed_tool_choices))
            )

        search_context_size = (
            config.search.web_search_search_context_size.strip().lower()
        )
        allowed_context_sizes = {"", "low", "medium", "high"}
        if search_context_size not in allowed_context_sizes:
            raise ConfigValidationError(
                "`search.web_search_search_context_size` must be one of: "
                + ", ".join(repr(value) for value in sorted(allowed_context_sizes))
            )

    def _validate_provider(self, config: AppConfig) -> None:
        provider = config.search.provider.strip()
        if provider == "openai_web_search":
            from .settings import resolve_openai_web_search_settings

            resolve_openai_web_search_settings(config)
            return
        if provider == "free_news_api":
            _free_news_api_key_env(config)
            return
        if provider == "google_news_rss":
            return
        raise ConfigValidationError(f"Unsupported search provider: {provider}")


def _free_news_api_key_env(config: AppConfig) -> str:
    """Resolve the configured FreeNewsAPI key env var name without reading it."""
    return config.search.api_key_env.strip() or "news_triage_codex_app"
