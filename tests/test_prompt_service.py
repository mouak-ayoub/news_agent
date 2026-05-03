from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from news_agent.configuration.loader import load_app_config
from news_agent.services.prompts.prompt_service import PromptService


class PromptServiceTests(unittest.TestCase):
    def test_prompt_service_injects_editorial_standards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            prompts = Path(tmp_dir)
            (prompts / "common").mkdir(parents=True)
            (prompts / "common" / "editorial_standards.txt").write_text(
                "Editorial standards:\n- Do not invent facts.",
                encoding="utf-8",
            )
            (prompts / "example.txt").write_text(
                "{editorial_standards}\n\nRole:\nYou are a tester.\n\nQuery: {query}",
                encoding="utf-8",
            )

            prompt = PromptService(prompts).build(
                "example",
                query="What changed?",
            )

        self.assertIn("Do not invent facts", prompt)
        self.assertIn("You are a tester", prompt)
        self.assertIn("What changed?", prompt)

    def test_prompt_service_allows_template_override_of_editorial_standards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            prompts = Path(tmp_dir)
            (prompts / "common").mkdir(parents=True)
            (prompts / "common" / "editorial_standards.txt").write_text(
                "Editorial standards:\n- Default.",
                encoding="utf-8",
            )
            (prompts / "example.txt").write_text(
                "{editorial_standards}\n\nQuery: {query}",
                encoding="utf-8",
            )

            prompt = PromptService(prompts).build(
                "example",
                query="What changed?",
                editorial_standards="Custom standards.",
            )

        self.assertIn("Custom standards.", prompt)
        self.assertNotIn("Default.", prompt)

    def test_prompt_service_missing_common_prefix_returns_empty_string(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            prompts = Path(tmp_dir)
            (prompts / "example.txt").write_text(
                "Start>{editorial_standards}<End Query: {query}",
                encoding="utf-8",
            )

            prompt = PromptService(prompts).build(
                "example",
                query="What changed?",
            )

        self.assertIn("Start><End", prompt)
        self.assertIn("What changed?", prompt)

    def test_question_analysis_prompt_renders_with_editorial_standards(self) -> None:
        prompt = PromptService().build("question_analysis", query="What changed?")

        self.assertIn("Editorial standards:", prompt)
        self.assertIn("You are a research intake analyst.", prompt)

    def test_query_planning_prompt_renders_with_editorial_standards(self) -> None:
        prompt = PromptService().build(
            "query_planning",
            query="What changed?",
            intent_json="{}",
        )

        self.assertIn("Editorial standards:", prompt)
        self.assertIn("You are a search-query planner.", prompt)

    def test_metric_extraction_prompt_renders_with_editorial_standards(self) -> None:
        prompt = PromptService().build(
            "metric_extraction",
            query="What changed?",
            intent_json="{}",
            article_json="{}",
        )

        self.assertIn("Editorial standards:", prompt)
        self.assertIn("You are a forensic fact extractor.", prompt)

    def test_summarization_prompt_renders_with_editorial_standards(self) -> None:
        prompt = PromptService().build(
            "summarization",
            query="What changed?",
            article_payload_json="[]",
        )

        self.assertIn("Editorial standards:", prompt)
        self.assertIn("briefing writer", prompt)

    def test_evidence_based_prompt_renders(self) -> None:
        prompt = PromptService().build(
            "analysis/evidence_based_analysis",
            evidence_bundle_json="{}",
        )

        self.assertIn("Editorial standards:", prompt)
        self.assertIn("You are an evidence-based news analyst.", prompt)
        self.assertIn("separate facts from evidence-backed inference", prompt)
        self.assertIn('"overall_assessment"', prompt)

    def test_speculative_red_team_prompt_renders(self) -> None:
        prompt = PromptService().build(
            "analysis/speculative_red_team_analysis",
            evidence_bundle_json="{}",
        )

        self.assertIn("Editorial standards:", prompt)
        self.assertIn("You are a hard-nosed speculative red-team news analyst.", prompt)
        self.assertIn("Do not give a weak or polite alternative reading.", prompt)
        self.assertIn("Do not present speculation as fact.", prompt)
        self.assertIn("Do not blame protected ethnic, religious, or national groups", prompt)
        self.assertIn('"adversarial_reading"', prompt)
        self.assertIn('"who_benefits"', prompt)
        self.assertIn('"mainstream_blind_spots"', prompt)
        self.assertIn('"speculative_hypotheses"', prompt)

    def test_web_search_prompt_renders_with_editorial_standards(self) -> None:
        prompt = PromptService().build(
            "web_search/web_search_research_new",
            outlet_limit=1,
            days_back=7,
            outlets_text="- Example | domain=example.com",
            planned_queries_json='["query"]',
            query="What changed?",
        )

        self.assertIn("Editorial standards:", prompt)
        self.assertIn("You are a news retrieval specialist.", prompt)

    def test_repair_prompt_renders_with_editorial_standards(self) -> None:
        prompt = PromptService().build(
            "web_search/adaptive_react_repair_planner",
            query="What changed?",
            outlets_json='[{"name": "Example"}]',
            planned_queries_json='["What changed?"]',
            previous_actions_json="[]",
            remaining_repair_actions=2,
            intent_json="{}",
            observation_json="{}",
        )

        self.assertIn("Editorial standards:", prompt)
        self.assertIn("You are a ReAct-style news retrieval repair planner.", prompt)
        self.assertIn("You, the repair planner, must interpret the observation.", prompt)
        self.assertIn("The objective is good evidence plus maximum outlet coverage.", prompt)
        self.assertIn("Prefer continuing until 8-12 distinct relevant outlets", prompt)
        self.assertIn("choose allowed_outlets mostly from outlets_without_candidates", prompt)
        self.assertIn("Choose 3-6 outlets per repair search", prompt)
        self.assertIn('"diagnosis"', prompt)
        self.assertIn(
            'Do not choose "search" only because many configured outlets are missing.',
            prompt,
        )

    def test_configured_major_prompt_templates_render(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_app_config(root / "config" / "news_agent_openai.yaml")
        service = PromptService()

        prompts = {
            "question_analysis": self._render_prompt(
                service,
                "question_analysis",
                query="What changed?",
            ),
            "query_planning": self._render_prompt(
                service,
                "query_planning",
                query="What changed?",
                intent_json=json.dumps(
                    {
                        "topic": "test topic",
                        "requested_metric": "latest count",
                        "expected_answer_type": "count",
                        "time_sensitivity": "latest",
                        "must_find": ["count"],
                        "avoid": [],
                    },
                    indent=2,
                ),
            ),
            "candidate_filter": self._render_prompt(
                service,
                "candidate_filter",
                outlet_name="Example",
                query="What changed?",
                intent_json="{}",
                candidate_lines_json="[]",
            ),
            "article_curation": self._render_prompt(
                service,
                "article_curation",
                outlet_name="Example",
                outlet_domain="example.com",
                query="What changed?",
                candidate_lines_json="[]",
            ),
            "metric_extraction": self._render_prompt(
                service,
                "metric_extraction",
                query="What changed?",
                intent_json="{}",
                article_json="{}",
            ),
            "summarization": self._render_prompt(
                service,
                "summarization",
                query="What changed?",
                article_payload_json="[]",
            ),
            config.analysis.evidence_based_prompt: self._render_prompt(
                service,
                config.analysis.evidence_based_prompt,
                evidence_bundle_json="{}",
            ),
            config.analysis.speculative_red_team_prompt: self._render_prompt(
                service,
                config.analysis.speculative_red_team_prompt,
                evidence_bundle_json="{}",
            ),
            config.search.web_search_prompt: self._render_prompt(
                service,
                config.search.web_search_prompt,
                outlet_limit=1,
                days_back=config.search.days_back,
                outlets_text=(
                    "- Example | domain=example.com | country=Test | "
                    "type=newspaper | orientation=center"
                ),
                planned_queries_json='["What changed?"]',
                query="What changed?",
            ),
            config.search.adaptive_react_repair_prompt: self._render_prompt(
                service,
                config.search.adaptive_react_repair_prompt,
                query="What changed?",
                outlets_json='[{"name": "Example"}]',
                planned_queries_json='["What changed?"]',
                previous_actions_json="[]",
                remaining_repair_actions=2,
                intent_json="{}",
                observation_json="{}",
            ),
        }

        for template_name, prompt in prompts.items():
            with self.subTest(template_name=template_name):
                self.assertTrue(prompt.strip())
                self.assertIn("Editorial standards:", prompt)

    def _render_prompt(
        self,
        service: PromptService,
        template_name: str,
        **variables: object,
    ) -> str:
        try:
            return service.build(template_name, **variables)
        except KeyError as exc:
            missing = exc.args[0]
            self.fail(
                f"Prompt {template_name!r} is missing render variable {missing!r}"
            )


if __name__ == "__main__":
    unittest.main()
