from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import re

from ..models.config import AppConfig


class DebugModelCall:
    """Represents one recorded model call inside a debug run."""

    def __init__(self, call_dir: Path) -> None:
        self.call_dir = call_dir

    def write_output(self, content: str) -> None:
        self._write("output.txt", content)

    def write_error(self, error: BaseException) -> None:
        self._write("error.txt", f"{type(error).__name__}: {error}")

    def write_artifact(self, file_name: str, content: str) -> None:
        """Record an extra file for this call, such as a provider response dump."""
        self._write(file_name, content)

    def _write(self, file_name: str, content: str) -> None:
        path = self.call_dir / file_name
        path.write_text(content, encoding="utf-8")


class DebugOutput:
    """Writes per-run artifacts for prompt and provider-output inspection."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._model_call_count = 0

    def write_run_context(
        self,
        *,
        query: str,
        config: AppConfig,
        html_report_path: Path,
    ) -> None:
        """Record the exact question, resolved config, and report target."""
        payload = {
            "query": query,
            "html_report_path": str(html_report_path),
            "config_path": str(config.config_path),
            "config": asdict(config),
        }
        self.write_json("run_context.json", payload)

    def write_text(self, file_name: str, content: str) -> None:
        path = self.run_dir / file_name
        path.write_text(content, encoding="utf-8")

    def write_json(self, file_name: str, payload: object) -> None:
        path = self.run_dir / file_name
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def start_model_call(self, name: str, input_text: str) -> DebugModelCall:
        """Create a numbered folder and record the exact model input prompt."""
        self._model_call_count += 1
        call_dir = (
            self.run_dir
            / "model_calls"
            / f"{self._model_call_count:03d}_{_slugify(name)}"
        )
        call_dir.mkdir(parents=True, exist_ok=True)
        (call_dir / "input.txt").write_text(input_text, encoding="utf-8")
        return DebugModelCall(call_dir)


def create_debug_output(query: str, project_root: Path) -> DebugOutput:
    """Create one timestamped debug folder for a single run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = project_root / "debug_output" / f"{timestamp}_{_slugify(query)}"
    return DebugOutput(run_dir)


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return normalized[:80] or "run"
