from __future__ import annotations

from ...models.config import AppConfig
from ..debug_output import DebugOutput
from ..prompt_service import PromptService
from ..text_generation import ModelGenerationError
from .base import SearchClient
from .free_news_api import FreeNewsApiSearchClient
from .openai import OpenAIWebSearchClient
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
        return OpenAIWebSearchClient(
            config=config,
            prompt_service=prompt_service,
            debug_output=debug_output,
        )
    raise ModelGenerationError(f"Unsupported search provider: {config.search.provider}")
