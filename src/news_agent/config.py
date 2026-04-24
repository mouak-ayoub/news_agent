from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(slots=True)
class ModelConfig:
    backend: str
    api_key_env: str
    research_model_id: str
    model_id: str
    max_output_tokens: int
    temperature: float
    fallback_to_heuristic: bool


@dataclass(slots=True)
class SearchConfig:
    provider: str
    days_back: int
    max_sources: int
    max_search_calls_per_run: int


@dataclass(slots=True)
class BudgetConfig:
    max_monthly_spend_usd: float
    max_run_spend_usd: float
    input_cost_per_million: float
    output_cost_per_million: float
    web_search_cost_per_call: float
    ledger_path: str


@dataclass(slots=True)
class OutletConfig:
    name: str
    domain: str
    country: str
    medium_type: str
    orientation: str
    notes: str


@dataclass(slots=True)
class AppConfig:
    model: ModelConfig
    search: SearchConfig
    budget: BudgetConfig
    outlets: list[OutletConfig]
    config_path: Path


def _default_config_path() -> Path:
    env_path = os.environ.get("NEWS_AGENT_CONFIG")
    if env_path:
        return Path(env_path).expanduser().resolve()

    project_root = Path(__file__).resolve().parents[2]
    return project_root / "config" / "news_agent.yaml"


def load_app_config(path: str | Path | None = None) -> AppConfig:
    config_path = Path(path).expanduser().resolve() if path else _default_config_path()
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    return AppConfig(
        model=ModelConfig(**data["model"]),
        search=SearchConfig(**data["search"]),
        budget=BudgetConfig(**data["budget"]),
        outlets=[OutletConfig(**outlet) for outlet in data["outlets"]],
        config_path=config_path,
    )
