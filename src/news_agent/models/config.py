from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path


@dataclass(slots=True)
class ModelConfig:
    backend: str = "heuristic"
    api_key_env: str = ""
    summary_model_id: str = ""
    analysis_model_id: str = ""
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
            "analysis": self.analysis_model_id,
        }
        configured = step_fields.get(step, "")
        if configured:
            return configured
        return self.summary_model_id


@dataclass(slots=True)
class AnalysisConfig:
    enabled: bool = False
    run_parallel: bool = True
    model_step: str = "analysis"
    evidence_based_prompt: str = "analysis/evidence_based_analysis"
    speculative_red_team_prompt: str = "analysis/speculative_red_team_analysis"
    max_output_tokens: int = 4000


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
    web_search_reasoning_effort: str = ""
    web_search_max_tool_calls: int = 1
    web_search_text_verbosity: str = "low"
    web_search_use_allowed_domains: bool = True
    web_search_include_sources: bool = True
    web_search_tool_choice: str = "required"
    web_search_search_context_size: str = "medium"
    web_search_use_site_query_filters: bool = False
    adaptive_react_enabled: bool = False
    adaptive_react_repair_prompt: str = "web_search/adaptive_react_repair_planner"
    adaptive_react_repair_max_tool_calls: int = 2
    adaptive_react_max_repair_actions: int = 2
    adaptive_react_max_candidates_per_outlet: int = 2


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
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
