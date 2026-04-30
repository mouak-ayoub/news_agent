import os
from pathlib import Path
import sys

# Edit these values when you want to run the project directly from PyCharm or
# with `.\.venv\Scripts\python.exe main.py` and no command-line arguments.
DEFAULT_QUERY = "What are the latest casualty figures in the Iran-USA conflict?"
# Short config names are resolved from the project `config` folder.
DEFAULT_CONFIG = "news_agent_gemini.yaml"
DEFAULT_HTML_OUT = "reports/gemma4-run.html"


SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from news_agent.command import main


if __name__ == "__main__":
    argv = None
    if len(sys.argv) == 1:
        argv = [DEFAULT_QUERY]
        if DEFAULT_CONFIG:
            argv.extend(["--config", DEFAULT_CONFIG])
        if DEFAULT_HTML_OUT:
            argv.extend(["--html-out", DEFAULT_HTML_OUT])
    raise SystemExit(main(argv))
