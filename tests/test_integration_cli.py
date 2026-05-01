from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class CliIntegrationTests(unittest.TestCase):
    def test_main_cli_runs_end_to_end_with_yaml_config(self) -> None:
        if not os.environ.get("openai_news_api"):
            self.skipTest("openai_news_api is not set for integration test.")

        project_root = Path(__file__).resolve().parents[1]
        config_path = project_root / "config" / "news_agent_openai.yaml"

        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        src_path = str(project_root / "src")
        env["PYTHONPATH"] = src_path if not existing else f"{src_path}{os.pathsep}{existing}"
        query = "What are the latest verified updates on global AI regulation?"

        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "integration_report.html"
            result = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    query,
                    "--html-out",
                    str(report_path),
                ],
                cwd=project_root,
                env={**env, "NEWS_AGENT_CONFIG": str(config_path)},
                capture_output=True,
                text=True,
                timeout=120,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn(f'"query": "{query}"', result.stdout)
            self.assertIn("Final brief:", result.stdout)
            self.assertIn("HTML report:", result.stdout)
            self.assertTrue(report_path.exists())


if __name__ == "__main__":
    unittest.main()
