from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .app import run_triage
from .config import load_app_config
from .html_report import default_report_path
from .html_report import write_html_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the news triage ADK workflow.")
    parser.add_argument("query", help="The news question to research and summarize.")
    parser.add_argument("--config", help="Optional path to a YAML config file.")
    parser.add_argument("--html-out", help="Optional path for a local HTML report.")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_app_config(args.config)
        brief = run_triage(args.query, config)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(brief.to_pretty_json())
    print()
    print("Final brief:")
    print(brief.final_brief)

    config_root = _resolve_report_root(config.config_path)
    report_path = (
        Path(args.html_out)
        if args.html_out
        else default_report_path(args.query, config_root)
    )
    output_path = write_html_report(brief, report_path)
    print()
    print(f"HTML report: {output_path}")
    return 0


def _resolve_report_root(config_path: Path) -> Path:
    resolved = config_path.resolve()
    return resolved.parents[1] if len(resolved.parents) > 1 else Path.cwd()
