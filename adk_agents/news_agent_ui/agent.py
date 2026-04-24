from __future__ import annotations

from pathlib import Path
import sys


# ADK Web expects an agents directory with `agent.py` exporting `root_agent`.
# We keep the actual app code in `src/news_agent` and add only this thin adapter
# so the ADK dev UI can load the same coordinator graph used by the CLI.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from news_agent.app import build_agent_graph
from news_agent.config import load_app_config


config = load_app_config()
root_agent = build_agent_graph(config)

