from __future__ import annotations

import os
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from news_agent.configuration.settings import resolve_openai_web_search_settings
from news_agent.configuration.validation import AppConfigValidator
from news_agent.configuration.validation import ConfigValidationError
from news_agent.models.config import AppConfig
from news_agent.models.config import ModelConfig
from news_agent.models.config import OutletConfig
from news_agent.models.config import SearchConfig
from news_agent.configuration.loader import load_app_config
from news_agent.configuration.loader import resolve_cli_config_arg


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
                      web_search_model_id: gpt-5.4-mini

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
                      web_search_model_id: gpt-5.4-mini

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
                      web_search_model_id: gpt-5.4-mini

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
        self.assertEqual(config.search.web_search_model_id, "gpt-5.4-mini")
        self.assertEqual(config.search.web_search_reasoning_effort, "low")
        self.assertEqual(config.search.web_search_max_tool_calls, 8)
        self.assertEqual(config.search.web_search_text_verbosity, "low")
        self.assertTrue(config.search.web_search_use_allowed_domains)
        self.assertTrue(config.search.web_search_include_sources)
        self.assertEqual(config.search.web_search_tool_choice, "required")
        self.assertEqual(config.search.web_search_search_context_size, "medium")
        self.assertFalse(config.search.web_search_use_site_query_filters)

    def test_search_config_accepts_allowed_domains_settings(self) -> None:
        config = _app_config()
        config.search.web_search_use_allowed_domains = True
        config.search.web_search_include_sources = True
        config.search.web_search_tool_choice = "required"
        config.search.web_search_search_context_size = "high"
        config.search.web_search_use_site_query_filters = False

        AppConfigValidator().validate(config)

    def test_search_config_rejects_invalid_search_context_size(self) -> None:
        config = _app_config()
        config.search.web_search_search_context_size = "huge"

        with self.assertRaisesRegex(
            ConfigValidationError,
            "web_search_search_context_size",
        ):
            AppConfigValidator().validate(config)

    def test_search_config_rejects_invalid_tool_choice(self) -> None:
        config = _app_config()
        config.search.web_search_tool_choice = "always"

        with self.assertRaisesRegex(ConfigValidationError, "web_search_tool_choice"):
            AppConfigValidator().validate(config)

    def test_openai_settings_use_search_api_key_env_when_present(self) -> None:
        config = _app_config()
        config.search.api_key_env = "SEARCH_KEY"
        config.model.api_key_env = "MODEL_KEY"

        settings = resolve_openai_web_search_settings(config)

        self.assertEqual(settings.api_key_env, "SEARCH_KEY")

    def test_openai_settings_fallback_to_model_api_key_for_openai_backend(self) -> None:
        config = _app_config()
        config.search.api_key_env = ""
        config.model.backend = "openai"
        config.model.api_key_env = "MODEL_KEY"

        settings = resolve_openai_web_search_settings(config)

        self.assertEqual(settings.api_key_env, "MODEL_KEY")

    def test_openai_settings_fail_without_api_key_env(self) -> None:
        config = _app_config()
        config.search.api_key_env = ""
        config.model.backend = "gemini"
        config.model.api_key_env = "MODEL_KEY"

        with self.assertRaisesRegex(ConfigValidationError, "api_key_env"):
            resolve_openai_web_search_settings(config)

    def test_openai_settings_fail_without_model_id(self) -> None:
        config = _app_config()
        config.search.web_search_model_id = ""

        with self.assertRaisesRegex(ConfigValidationError, "web_search_model_id"):
            resolve_openai_web_search_settings(config)

    def test_openai_settings_copy_generation_fields(self) -> None:
        config = _app_config()
        config.model.max_output_tokens = 123
        config.model.temperature = 0.4
        config.search.web_search_reasoning_effort = "high"
        config.search.web_search_max_tool_calls = 3
        config.search.web_search_text_verbosity = "medium"

        settings = resolve_openai_web_search_settings(config)

        self.assertEqual(settings.model_id, "gpt-5.4-mini")
        self.assertEqual(settings.max_output_tokens, 123)
        self.assertEqual(settings.temperature, 0.4)
        self.assertEqual(settings.reasoning_effort, "high")
        self.assertEqual(settings.max_tool_calls, 3)
        self.assertEqual(settings.text_verbosity, "medium")

    def test_validator_fails_on_invalid_max_sources(self) -> None:
        config = _app_config()
        config.search.max_sources = 0

        with self.assertRaisesRegex(ConfigValidationError, "max_sources"):
            AppConfigValidator().validate(config)

    def test_validator_fails_on_invalid_max_search_calls(self) -> None:
        config = _app_config()
        config.search.max_search_calls_per_run = 0

        with self.assertRaisesRegex(ConfigValidationError, "max_search_calls_per_run"):
            AppConfigValidator().validate(config)

    def test_validator_fails_when_no_outlets_exist(self) -> None:
        config = _app_config()
        config.outlets = []

        with self.assertRaisesRegex(ConfigValidationError, "at least one outlet"):
            AppConfigValidator().validate(config)

    def test_validator_uses_openai_settings_resolver(self) -> None:
        config = _app_config()
        config.search.web_search_model_id = ""

        with self.assertRaisesRegex(ConfigValidationError, "web_search_model_id"):
            AppConfigValidator().validate(config)


def _app_config() -> AppConfig:
    return AppConfig(
        model=ModelConfig(
            backend="gemini",
            api_key_env="GEMINI_KEY",
            question_analysis_model_id="gemma-4-31b-it",
            query_planning_model_id="gemma-4-31b-it",
            candidate_filter_model_id="gemma-4-31b-it",
            article_selection_model_id="gemma-4-31b-it",
            metric_extraction_model_id="gemma-4-31b-it",
            summary_model_id="gemma-4-31b-it",
            max_output_tokens=256,
            temperature=0.2,
        ),
        search=SearchConfig(
            provider="openai_web_search",
            days_back=7,
            max_sources=5,
            max_search_calls_per_run=1,
            api_key_env="OPENAI_SEARCH_KEY",
            web_search_model_id="gpt-5.4-mini",
            web_search_reasoning_effort="low",
            web_search_max_tool_calls=1,
            web_search_text_verbosity="low",
        ),
        outlets=[
            OutletConfig(
                name="Example",
                domain="example.com",
                country="France",
                medium_type="newspaper",
                orientation="center",
                notes="test",
            )
        ],
        config_path=Path("config/news_agent_openai.yaml"),
    )


if __name__ == "__main__":
    unittest.main()
