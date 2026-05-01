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
                      question_analysis_model_id: gpt-4.1
                      query_planning_model_id: gpt-4.1
                      candidate_filter_model_id: gpt-4.1
                      article_selection_model_id: gpt-4.1
                      metric_extraction_model_id: gpt-4.1
                      summary_model_id: local/gemma
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

        self.assertEqual(config.model.summary_model_id, "local/gemma")
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
                      question_analysis_model_id: gpt-4.1
                      query_planning_model_id: gpt-4.1
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

    def test_yaml_config_loads_outlets_from_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            outlets_dir = root / "outlets"
            outlets_dir.mkdir()
            (outlets_dir / "test_outlets.yaml").write_text(
                textwrap.dedent(
                    """
                    outlets:
                      - name: Example
                        domain: example.com
                        country: France
                        medium_type: newspaper
                        orientation: center
                        notes: loaded from separate file
                    """
                ).strip(),
                encoding="utf-8",
            )
            config_path = root / "news_agent_openai.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    model:
                      backend: openai
                      api_key_env: NEWS_AGENT_KEY
                      question_analysis_model_id: gpt-4.1
                      query_planning_model_id: gpt-4.1
                      candidate_filter_model_id: gpt-4.1
                      article_selection_model_id: gpt-4.1
                      metric_extraction_model_id: gpt-4.1
                      summary_model_id: gpt-5
                      max_output_tokens: 256
                      temperature: 0.2

                    search:
                      provider: openai_web_search
                      days_back: 7
                      max_sources: 5
                      max_search_calls_per_run: 1

                    outlets_file: outlets/test_outlets.yaml
                    """
                ).strip(),
                encoding="utf-8",
            )

            config = load_app_config(config_path)

        self.assertEqual(config.outlets[0].name, "Example")
        self.assertEqual(config.outlets[0].notes, "loaded from separate file")

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

    def test_openai_config_uses_gemma_steps_and_openai_web_search(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config" / "news_agent_openai.yaml"

        config = load_app_config(config_path)

        self.assertEqual(config.model.backend, "gemini")
        self.assertEqual(config.model.question_analysis_model_id, "gemma-4-31b-it")
        self.assertEqual(config.model.query_planning_model_id, "gemma-4-31b-it")
        self.assertEqual(config.model.candidate_filter_model_id, "gemma-4-31b-it")
        self.assertEqual(config.model.article_selection_model_id, "gemma-4-31b-it")
        self.assertEqual(config.model.metric_extraction_model_id, "gemma-4-31b-it")
        self.assertEqual(config.model.summary_model_id, "gemma-4-31b-it")
        self.assertEqual(config.search.provider, "openai_web_search")
        self.assertEqual(config.search.api_key_env, "openai_news_api")
        self.assertEqual(config.search.web_search_model_id, "gpt-4.1")


if __name__ == "__main__":
    unittest.main()
