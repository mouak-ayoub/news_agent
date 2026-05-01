from __future__ import annotations

from pathlib import Path


class PromptService:
    def __init__(self, prompts_dir: Path | None = None) -> None:
        self.prompts_dir = prompts_dir or _default_prompts_dir()
        self._cache: dict[str, str] = {}

    def build(self, template_name: str, **variables: object) -> str:
        template = self._load(template_name)
        return template.format(**variables).strip()

    def _load(self, template_name: str) -> str:
        if template_name in self._cache:
            return self._cache[template_name]
        path = self.prompts_dir / Path(f"{template_name}.txt")
        template = path.read_text(encoding="utf-8")
        self._cache[template_name] = template
        return template


def _default_prompts_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "config" / "prompts"
