from __future__ import annotations

import os
from pathlib import Path

import yaml

from ..models.config import AppConfig
from ..models.config import ModelConfig
from ..models.config import OutletConfig
from ..models.config import SearchConfig


class ConfigLoader:
    def load(self, path: str | Path | None = None) -> AppConfig:
        config_path = (
            self.resolve_config_path(path)
            if path
            else self.default_config_path()
        )
        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        model_data = _normalize_model_config(data["model"])
        fallback_to_heuristic = bool(
            data.get("fallback_to_heuristic", model_data.pop("fallback_to_heuristic", False))
        )

        return AppConfig(
            model=ModelConfig(**model_data),
            search=SearchConfig(**data["search"]),
            outlets=[OutletConfig(**outlet) for outlet in data["outlets"]],
            config_path=config_path,
            fallback_to_heuristic=fallback_to_heuristic,
        )

    def resolve_config_path(self, path: str | Path) -> Path:
        """Resolve explicit config paths and short names like `news_agent_gemini.yaml`."""
        requested_path = Path(path).expanduser()
        if requested_path.exists() or requested_path.parent != Path("."):
            return requested_path.resolve()

        project_config_path = project_root() / "config" / requested_path
        if project_config_path.exists():
            return project_config_path.resolve()

        return requested_path.resolve()

    def default_config_path(self) -> Path:
        env_path = os.environ.get("NEWS_AGENT_CONFIG")
        if env_path:
            return self.resolve_config_path(env_path)

        cwd_config = Path.cwd() / "config" / "news_agent_openai.yaml"
        if cwd_config.exists():
            return cwd_config.resolve()

        source_config = (
            Path(__file__).resolve().parents[3] / "config" / "news_agent_openai.yaml"
        )
        return source_config.resolve()


def load_app_config(path: str | Path | None = None) -> AppConfig:
    return ConfigLoader().load(path)


def resolve_cli_config_arg(config_arg: str | None) -> str | None:
    if config_arg or os.environ.get("NEWS_AGENT_CONFIG"):
        return config_arg

    project_gemini_config = project_root() / "config" / "news_agent_gemini.yaml"
    if (
        (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
        and project_gemini_config.exists()
    ):
        return str(project_gemini_config)

    project_free_config = project_root() / "config" / "news_agent_free.yaml"
    if not os.environ.get("NEWS_AGENT_KEY") and project_free_config.exists():
        return str(project_free_config)

    return None


def report_root_from_config(config_path: Path) -> Path:
    resolved = config_path.resolve()
    return resolved.parents[1] if len(resolved.parents) > 1 else project_root()


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _normalize_model_config(model_data: dict) -> dict:
    normalized = dict(model_data)
    if "summary_model_id" not in normalized and "model_id" in normalized:
        normalized["summary_model_id"] = normalized.pop("model_id")
    return normalized
