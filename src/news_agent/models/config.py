from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path


@dataclass(slots=True)
class ModelConfig:
    backend: str = "heuristic"
    api_key_env: str = ""
    summary_model_id: str = ""
    max_output_tokens: int = 0
    temperature: float = 0.0
    question_analysis_model_id: str = ""
    query_planning_model_id: str = ""
    candidate_filter_model_id: str = ""
    article_selection_model_id: str = ""
    metric_extraction_model_id: str = ""
    base_url: str = "http://127.0.0.1:11434"
    request_timeout_seconds: int = 240
    gemini_retry_attempts: int = 3
    gemini_retry_backoff_seconds: float = 2.0

    def model_id_for_step(self, step: str) -> str:
        """Return the configured model for one pipeline step."""
        step_fields = {
            "question_analysis": self.question_analysis_model_id,
            "query_planning": self.query_planning_model_id,
            "candidate_filter": self.candidate_filter_model_id,
            "article_selection": self.article_selection_model_id,
            "metric_extraction": self.metric_extraction_model_id,
            "summarization": self.summary_model_id,
        }
        configured = step_fields.get(step, "")
        if configured:
            return configured
        return self.summary_model_id


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
    web_search_prompt: str = "web_search_research"
    web_search_model_id: str = ""


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
