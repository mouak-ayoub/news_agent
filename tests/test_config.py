from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from news_agent.config import load_app_config


class ConfigTests(unittest.TestCase):
    def test_yaml_config_loads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "news_agent.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    model:
                      backend: openai
                      api_key_env: NEWS_AGENT_KEY
                      research_model_id: gpt-4.1
                      model_id: local/gemma
                      max_output_tokens: 256
                      temperature: 0.2
                      fallback_to_heuristic: true

                    search:
                      provider: openai_web_search
                      days_back: 7
                      max_sources: 5
                      max_search_calls_per_run: 1

                    budget:
                      max_monthly_spend_usd: 10.0
                      max_run_spend_usd: 0.15
                      input_cost_per_million: 0.25
                      output_cost_per_million: 2.0
                      web_search_cost_per_call: 0.01
                      ledger_path: data/test_usage_guard.json

                    outlets:
                      - name: Example
                        domain: example.com
                        country: France
                        medium_type: newspaper
                        orientation: center
                        notes: test
                    """
                ).strip(),
                encoding="utf-8",
            )

            config = load_app_config(config_path)

        self.assertEqual(config.model.model_id, "local/gemma")
        self.assertEqual(config.search.max_sources, 5)
        self.assertEqual(config.outlets[0].domain, "example.com")


if __name__ == "__main__":
    unittest.main()
