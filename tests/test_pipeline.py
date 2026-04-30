from __future__ import annotations

from pathlib import Path
import unittest

from news_agent.workflow import run_triage
from news_agent.models.config import AppConfig
from news_agent.models.config import ModelConfig
from news_agent.models.config import OutletConfig
from news_agent.models.config import SearchConfig
from news_agent.models.triage import ArticleRecord
from news_agent.models.triage import ResearchBundle
from news_agent.services.summarization import SummarizationService
from news_agent.services.text_generation import ModelGenerationError
from news_agent.services.text_generation import ModelOutputError
from news_agent.services.text_generation import StaticTextGenerator


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
                research_model_id="gpt-4.1",
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
            fallback_to_heuristic=False,
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

    def test_heuristic_fallback_returns_safe_output(self) -> None:
        self.config.fallback_to_heuristic = True
        research_service = FakeResearchService(self.articles[:1])
        summarization_service = SummarizationService(
            config=self.config,
            text_generator=StaticTextGenerator("not json"),
        )

        brief = run_triage(
            "Did the escalation really happen?",
            self.config,
            research_service=research_service,
            summarization_service=summarization_service,
        )

        self.assertEqual(brief.query, "Did the escalation really happen?")
        self.assertTrue(brief.uncertainties)
        self.assertTrue(brief.fact_inference_speculation.speculation)
        self.assertEqual(len(brief.source_findings), 1)

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


if __name__ == "__main__":
    unittest.main()
