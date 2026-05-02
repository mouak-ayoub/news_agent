from __future__ import annotations

from datetime import datetime
from pathlib import Path
import tempfile
import unittest

from news_agent.workflow import run_triage
from news_agent.models.config import AppConfig
from news_agent.models.config import ModelConfig
from news_agent.models.config import OutletConfig
from news_agent.models.config import SearchConfig
from news_agent.models.triage import ArticleRecord
from news_agent.models.triage import ResearchBundle
from news_agent.services.debug_output import DebugOutput
from news_agent.services.debug_output import create_debug_output
from news_agent.services.summarization import SummarizationService
from news_agent.services.text_generation import ModelGenerationError
from news_agent.services.text_generation import ModelOutputError
from news_agent.services.text_generation import StaticTextGenerator
from news_agent.services.text_generation import _gemini_retry_attempts
from news_agent.services.text_generation import _gemini_retry_delay_seconds
from news_agent.services.text_generation import openai_supports_temperature


class FailingTextGenerator:
    def generate(self, prompt: str):
        raise ModelGenerationError("provider failed")


class FakeResearchService:
    def __init__(self, articles: list[ArticleRecord]) -> None:
        self._articles = articles

    def research(self, query: str) -> ResearchBundle:
        return ResearchBundle(query=query, articles=self._articles)


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AppConfig(
            model=ModelConfig(
                backend="openai",
                api_key_env="NEWS_AGENT_KEY",
                question_analysis_model_id="gpt-4.1",
                query_planning_model_id="gpt-4.1",
                candidate_filter_model_id="gpt-4.1",
                article_selection_model_id="gpt-4.1",
                metric_extraction_model_id="gpt-4.1",
                summary_model_id="gpt-5-mini",
                max_output_tokens=256,
                temperature=0.2,
            ),
            search=SearchConfig(
                provider="openai_web_search",
                days_back=7,
                max_sources=5,
                max_search_calls_per_run=1,
            ),
            outlets=[
                OutletConfig(
                    name="CNN",
                    domain="cnn.com",
                    country="United States",
                    medium_type="TV / digital",
                    orientation="center-left",
                    notes="test",
                )
            ],
            config_path=Path("config/news_agent_openai.yaml"),
        )
        self.articles = [
            ArticleRecord(
                title="US outlet frames the strike as deterrence",
                url="https://example.com/a",
                outlet_name="CNN",
                domain="cnn.com",
                country="United States",
                medium_type="TV / digital",
                orientation="center-left",
                published_at="2026-04-04T10:00:00+00:00",
                snippet="Snippet A",
                article_text="Body A",
                search_query="query",
            ),
            ArticleRecord(
                title="French outlet stresses escalation risk",
                url="https://example.com/b",
                outlet_name="Le Monde",
                domain="lemonde.fr",
                country="France",
                medium_type="newspaper",
                orientation="center-left",
                published_at="2026-04-04T09:00:00+00:00",
                snippet="Snippet B",
                article_text="Body B",
                search_query="query",
            ),
        ]

    def test_pipeline_returns_structured_brief(self) -> None:
        research_service = FakeResearchService(self.articles)
        summarization_service = SummarizationService(
            config=self.config,
            text_generator=StaticTextGenerator(
                """
                {
                  "query": "placeholder",
                  "main_claims": [
                    {
                      "claim": "The escalation is real but framed differently.",
                      "status": "partly confirmed",
                      "evidence_level": "moderate"
                    }
                  ],
                  "entities": {
                    "countries": ["France", "United States"],
                    "people": [],
                    "organizations": [],
                    "locations": []
                  },
                  "source_profiles": [],
                  "framing_analysis": ["US coverage stresses deterrence while French coverage stresses risk."],
                  "historical_context": ["Similar regional crises are often framed through alliance and legality lenses."],
                  "uncertainties": ["The exact trigger remains contested."],
                  "fact_inference_speculation": {
                    "observation": ["Headlines diverge across outlets."],
                    "evidence_backed_inference": ["Different media systems emphasize different stakes."],
                    "speculation": ["Some silence may be strategic."]
                  },
                  "final_brief": "The event appears real, but justification and risk are framed differently."
                }
                """
            ),
        )

        brief = run_triage(
            "Why are they describing the escalation differently?",
            self.config,
            research_service=research_service,
            summarization_service=summarization_service,
        )

        self.assertEqual(brief.query, "Why are they describing the escalation differently?")
        self.assertEqual(brief.main_claims[0].status, "partly confirmed")
        self.assertEqual(len(brief.source_profiles), 2)
        self.assertEqual(len(brief.source_findings), 2)
        self.assertEqual(brief.source_findings[0].outlet_name, "CNN")
        self.assertIn("framed differently", brief.final_brief)

    def test_external_backend_bad_output_stops_instead_of_falling_back(self) -> None:
        summarization_service = SummarizationService(
            config=self.config,
            text_generator=StaticTextGenerator("not json"),
        )

        with self.assertRaises(ModelOutputError):
            summarization_service.summarize(
                "Did the model return bad output?",
                ResearchBundle(
                    query="Did the model return bad output?",
                    articles=self.articles[:1],
                ),
            )

    def test_generation_error_stops_instead_of_falling_back(self) -> None:
        summarization_service = SummarizationService(
            config=self.config,
            text_generator=FailingTextGenerator(),
        )

        with self.assertRaises(ModelGenerationError):
            summarization_service.summarize(
                "Did the provider fail?",
                ResearchBundle(query="Did the provider fail?", articles=self.articles[:1]),
            )

    def test_debug_output_records_model_input_and_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            summarization_service = SummarizationService(
                config=self.config,
                text_generator=StaticTextGenerator(
                    """
                    {
                      "query": "placeholder",
                      "main_claims": [],
                      "entities": {},
                      "source_profiles": [],
                      "source_findings": [],
                      "framing_analysis": [],
                      "historical_context": [],
                      "uncertainties": [],
                      "fact_inference_speculation": {},
                      "final_brief": "Debug test brief."
                    }
                    """
                ),
                debug_output=DebugOutput(Path(tmp_dir)),
            )

            summarization_service.summarize(
                "What changed?",
                ResearchBundle(query="What changed?", articles=self.articles[:1]),
            )

            call_dirs = sorted((Path(tmp_dir) / "model_calls").iterdir())
            self.assertEqual(len(call_dirs), 1)
            self.assertTrue((call_dirs[0] / "input.txt").exists())
            self.assertTrue((call_dirs[0] / "output.txt").exists())
            self.assertIn("What changed?", (call_dirs[0] / "input.txt").read_text())
            self.assertIn("Debug test brief", (call_dirs[0] / "output.txt").read_text())

    def test_debug_output_writes_git_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            debug_output = create_debug_output("What changed?", Path(tmp_dir))

            self.assertEqual(
                debug_output.run_dir.parent.name,
                datetime.now().strftime("%Y-%m-%d"),
            )
            self.assertTrue((debug_output.run_dir / "git_fingerprint.json").exists())

    def test_gpt5_family_omits_temperature(self) -> None:
        self.assertFalse(openai_supports_temperature("gpt-5"))
        self.assertFalse(openai_supports_temperature("gpt-5-mini"))
        self.assertTrue(openai_supports_temperature("gpt-4.1"))

    def test_gemini_retry_config_is_sanitized(self) -> None:
        config = ModelConfig(
            backend="gemini",
            api_key_env="GEMINI_API_KEY",
            summary_model_id="gemma-4-31b-it",
            gemini_retry_attempts=0,
            gemini_retry_backoff_seconds=-2.0,
        )

        self.assertEqual(_gemini_retry_attempts(config), 1)
        self.assertEqual(_gemini_retry_delay_seconds(config, attempt=2), 0.0)


if __name__ == "__main__":
    unittest.main()
