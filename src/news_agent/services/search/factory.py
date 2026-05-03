from __future__ import annotations

from news_agent.configuration.settings import resolve_openai_web_search_settings
from news_agent.models.config import AppConfig
from news_agent.services.debug.debug_output import DebugOutput
from news_agent.services.prompts.prompt_service import PromptService
from news_agent.services.llm.text_generation import ModelGenerationError
from .base import SearchClient
from .free_news_api import FreeNewsApiSearchClient
from .openai import OpenAIWebSearchClient
from .openai.gateway import DebuggingOpenAIWebSearchGateway
from .openai.gateway import OpenAIWebSearchGateway
from .rss import GoogleNewsRssSearchClient


def build_search_client(
    config: AppConfig,
    prompt_service: PromptService | None = None,
    debug_output: DebugOutput | None = None,
) -> SearchClient:
    prompt_service = prompt_service or PromptService()
    if config.search.provider == "google_news_rss":
        return GoogleNewsRssSearchClient(
            config=config,
            prompt_service=prompt_service,
            debug_output=debug_output,
        )
    if config.search.provider == "free_news_api":
        return FreeNewsApiSearchClient(
            config=config,
            prompt_service=prompt_service,
            debug_output=debug_output,
        )
    if config.search.provider == "openai_web_search":
        settings = resolve_openai_web_search_settings(config)
        gateway = OpenAIWebSearchGateway(api_key_env=settings.api_key_env)
        if debug_output:
            gateway = DebuggingOpenAIWebSearchGateway(gateway, debug_output)
        return OpenAIWebSearchClient(
            config=config,
            prompt_service=prompt_service,
            settings=settings,
            gateway=gateway,
        )
    raise ModelGenerationError(f"Unsupported search provider: {config.search.provider}")


