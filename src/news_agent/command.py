from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

from .workflow import run_triage
from .services.config_loader import load_app_config
from .services.config_loader import report_root_from_config
from .services.config_loader import resolve_cli_config_arg
from .services.reporting import default_report_path
from .services.reporting import write_html_report


logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the news triage ADK workflow.")
    parser.add_argument("query", help="The news question to research and summarize.")
    parser.add_argument("--config", help="Optional path to a YAML config file.")
    parser.add_argument("--html-out", help="Optional path for a local HTML report.")
    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_app_config(resolve_cli_config_arg(args.config))
        brief = run_triage(args.query, config)
    except Exception as exc:
        logger.exception("run failed")
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(brief.to_pretty_json())
    print()
    print("Final brief:")
    print(brief.final_brief)

    config_root = report_root_from_config(config.config_path)
    report_path = (
        Path(args.html_out)
        if args.html_out
        else default_report_path(args.query, config_root)
    )
    output_path = write_html_report(brief, report_path)
    print()
    print(f"HTML report: {output_path}")
    return 0


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
