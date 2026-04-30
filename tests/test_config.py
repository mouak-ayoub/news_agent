from __future__ import annotations

import os
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from news_agent.services.config_loader import load_app_config
from news_agent.services.config_loader import resolve_cli_config_arg


class ConfigTests(unittest.TestCase):
    def test_yaml_config_loads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "news_agent_openai.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    model:
                      backend: openai
                      api_key_env: NEWS_AGENT_KEY
                      research_model_id: gpt-4.1
                      summary_model_id: local/gemma
                      max_output_tokens: 256
                      temperature: 0.2

                    fallback_to_heuristic: true

                    search:
                      provider: openai_web_search
                      days_back: 7
                      max_sources: 5
                      max_search_calls_per_run: 1

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

        self.assertEqual(config.model.summary_model_id, "local/gemma")
        self.assertTrue(config.fallback_to_heuristic)
        self.assertEqual(config.search.max_sources, 5)
        self.assertEqual(config.outlets[0].domain, "example.com")

    def test_yaml_legacy_model_id_maps_to_summary_model_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "news_agent_openai.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    model:
                      backend: openai
                      api_key_env: NEWS_AGENT_KEY
                      research_model_id: gpt-4.1
                      model_id: legacy-id
                      max_output_tokens: 256
                      temperature: 0.2

                    search:
                      provider: openai_web_search
                      days_back: 7
                      max_sources: 5
                      max_search_calls_per_run: 1

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

        self.assertEqual(config.model.summary_model_id, "legacy-id")

    def test_resolve_cli_config_prefers_gemini_when_key_exists(self) -> None:
        with patch.dict(
            os.environ,
            {
                "GEMINI_API_KEY": "test-key",
                "NEWS_AGENT_CONFIG": "",
                "NEWS_AGENT_KEY": "",
                "GOOGLE_API_KEY": "",
            },
            clear=False,
        ):
            config_path = resolve_cli_config_arg(None)

        self.assertIsNotNone(config_path)
        assert config_path is not None
        self.assertTrue(config_path.endswith("config\\news_agent_gemini.yaml"))


if __name__ == "__main__":
    unittest.main()
