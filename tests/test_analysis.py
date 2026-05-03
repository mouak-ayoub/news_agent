from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from news_agent.models.generation import GenerationResult
from news_agent.models.triage import ArticleRecord
from news_agent.models.triage import ResearchBundle
from news_agent.models.triage import TriageBrief
from news_agent.services.analysis.analysis_service import AnalysisService
from news_agent.services.debug.debug_output import DebugOutput
from news_agent.services.llm.text_generation import ModelGenerationError
from news_agent.services.prompts.prompt_service import PromptService


class SequencedTextGenerator:
    def __init__(self, outputs: list[str | BaseException]) -> None:
        self.outputs = list(outputs)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> GenerationResult:
        self.prompts.append(prompt)
        if not self.outputs:
            raise AssertionError("No fake analysis output remains.")
        output = self.outputs.pop(0)
        if isinstance(output, BaseException):
            raise output
        return GenerationResult(text=output)


class AnalysisServiceTests(unittest.TestCase):
    def test_analysis_service_parses_evidence_based_json(self) -> None:
        generator = SequencedTextGenerator([_evidence_json(), _speculative_json()])
        service = _analysis_service(generator)

        bundle = service.analyze(
            query="What changed?",
            bundle=_research_bundle(),
            brief=_brief(),
        )

        self.assertIsNotNone(bundle.evidence_based)
        assert bundle.evidence_based is not None
        self.assertEqual(bundle.evidence_based.title, "Evidence-based analysis")
        self.assertEqual(bundle.evidence_based.facts, ["The article reports one fact."])
        self.assertEqual(bundle.evidence_based.confidence, "medium")

    def test_analysis_service_parses_speculative_json(self) -> None:
        generator = SequencedTextGenerator([_evidence_json(), _speculative_json()])
        service = _analysis_service(generator)

        bundle = service.analyze(
            query="What changed?",
            bundle=_research_bundle(),
            brief=_brief(),
        )

        self.assertIsNotNone(bundle.speculative_red_team)
        assert bundle.speculative_red_team is not None
        self.assertEqual(
            bundle.speculative_red_team.title,
            "Speculative red-team lens",
        )
        self.assertEqual(
            bundle.speculative_red_team.speculative_hypotheses,
            ["Hypothesis, explicitly speculative."],
        )
        self.assertEqual(
            bundle.speculative_red_team.adversarial_reading,
            "A harsher alternative reading.",
        )
        self.assertEqual(
            bundle.speculative_red_team.who_benefits,
            ["An actor may benefit narratively."],
        )
        self.assertEqual(
            bundle.speculative_red_team.mainstream_blind_spots,
            ["A missing question."],
        )
        self.assertEqual(bundle.speculative_red_team.confidence, "speculative")

    def test_analysis_service_survives_one_failed_agent(self) -> None:
        generator = SequencedTextGenerator(
            [
                ModelGenerationError("provider failed"),
                _speculative_json(),
            ]
        )
        service = _analysis_service(generator)

        bundle = service.analyze(
            query="What changed?",
            bundle=_research_bundle(),
            brief=_brief(),
        )

        self.assertIsNone(bundle.evidence_based)
        self.assertIsNotNone(bundle.speculative_red_team)

    def test_analysis_agents_receive_same_evidence_bundle(self) -> None:
        generator = SequencedTextGenerator([_evidence_json(), _speculative_json()])
        service = _analysis_service(generator)

        service.analyze(
            query="What changed?",
            bundle=_research_bundle(),
            brief=_brief(),
        )

        self.assertEqual(len(generator.prompts), 2)
        evidence_payload = _extract_prompt_bundle(generator.prompts[0])
        speculative_payload = _extract_prompt_bundle(generator.prompts[1])
        self.assertEqual(evidence_payload, speculative_payload)
        self.assertEqual(evidence_payload["final_summary"], "Final summary.")
        self.assertIn(
            "Full article body with source evidence.",
            evidence_payload["selected_sources"][0]["article_text"],
        )

    def test_analysis_service_writes_debug_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            generator = SequencedTextGenerator([_evidence_json(), _speculative_json()])
            service = _analysis_service(
                generator,
                debug_output=DebugOutput(Path(tmp_dir)),
            )

            service.analyze(
                query="What changed?",
                bundle=_research_bundle(),
                brief=_brief(),
            )

            root = Path(tmp_dir)
            self.assertTrue((root / "evidence_based_analysis_input.txt").exists())
            self.assertTrue((root / "evidence_based_analysis_output.txt").exists())
            self.assertTrue((root / "speculative_red_team_analysis_input.txt").exists())
            self.assertTrue((root / "speculative_red_team_analysis_output.txt").exists())
            self.assertTrue((root / "analysis_bundle.json").exists())


def _analysis_service(
    generator: SequencedTextGenerator,
    *,
    debug_output: DebugOutput | None = None,
) -> AnalysisService:
    return AnalysisService(
        prompt_service=PromptService(),
        text_generator=generator,
        debug_output=debug_output,
        run_parallel=False,
    )


def _research_bundle() -> ResearchBundle:
    return ResearchBundle(
        query="What changed?",
        articles=[
            ArticleRecord(
                title="Outlet reports the event",
                url="https://example.com/story",
                outlet_name="Example",
                domain="example.com",
                country="United States",
                medium_type="digital",
                orientation="center",
                published_at="2026-05-01",
                snippet="Snippet",
                article_text="Full article body with source evidence.",
                search_query="query",
                metric_found=True,
                metric_value="12",
                metric_type="count",
                metric_evidence="The article says 12.",
                metric_confidence="high",
            )
        ],
    )


def _brief() -> TriageBrief:
    return TriageBrief(
        query="What changed?",
        final_brief="Final summary.",
        framing_analysis=["One outlet frames it as escalation."],
        uncertainties=["The official count may change."],
    )


def _evidence_json() -> str:
    return json.dumps(
        {
            "title": "Evidence-based analysis",
            "overall_assessment": "The evidence supports a cautious reading.",
            "facts": ["The article reports one fact."],
            "evidence_backed_inferences": ["The event likely matters."],
            "uncertainties": ["The count may change."],
            "source_disagreements": [],
            "confidence": "medium",
        }
    )


def _speculative_json() -> str:
    return json.dumps(
        {
            "title": "Speculative red-team lens",
            "core_suspicion": "Speculatively, timing may matter.",
            "adversarial_reading": "A harsher alternative reading.",
            "who_benefits": ["An actor may benefit narratively."],
            "suspicious_patterns": ["The timing is notable."],
            "possible_hidden_actors_or_incentives": ["A state actor could benefit."],
            "speculative_hypotheses": ["Hypothesis, explicitly speculative."],
            "mainstream_blind_spots": ["A missing question."],
            "weaknesses_in_this_reading": ["The evidence is thin."],
            "evidence_needed": ["Primary documents."],
            "confidence": "speculative",
        }
    )


def _extract_prompt_bundle(prompt: str) -> dict:
    marker = "Input evidence bundle:\n"
    return json.loads(prompt.split(marker, 1)[1])


if __name__ == "__main__":
    unittest.main()
