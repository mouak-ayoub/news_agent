from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path


@dataclass(slots=True)
class ModelConfig:
    backend: str
    api_key_env: str
    research_model_id: str
    summary_model_id: str
    max_output_tokens: int
    temperature: float
    base_url: str = "http://127.0.0.1:11434"
    request_timeout_seconds: int = 240


@dataclass(slots=True)
class SearchConfig:
    provider: str
    days_back: int
    max_sources: int
    max_search_calls_per_run: int
    candidate_pool_size: int = 10
    request_timeout_seconds: int = 12
    user_agent: str = "news-agent/0.1 (+local learning project)"
    allow_google_news_fallback: bool = True
    api_key_env: str = ""
    base_url: str = ""


@dataclass(slots=True)
class OutletConfig:
    name: str
    domain: str
    country: str
    medium_type: str
    orientation: str
    notes: str
    rss_url: str = ""
    rss_urls: list[str] = field(default_factory=list)
    publisher_uuid: str = ""


@dataclass(slots=True)
class AppConfig:
    model: ModelConfig
    search: SearchConfig
    outlets: list[OutletConfig]
    config_path: Path
    fallback_to_heuristic: bool = False
