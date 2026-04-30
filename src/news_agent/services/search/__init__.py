"""Search provider implementations."""

from .base import SearchClient
from .factory import build_search_client
from .free_news_api import FreeNewsApiSearchClient
from .openai import OpenAIWebSearchClient
from .rss import GoogleNewsRssSearchClient

__all__ = [
    "FreeNewsApiSearchClient",
    "GoogleNewsRssSearchClient",
    "OpenAIWebSearchClient",
    "SearchClient",
    "build_search_client",
]
