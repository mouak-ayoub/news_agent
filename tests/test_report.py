from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from news_agent.models.analysis import AnalysisBundle
from news_agent.models.analysis import EvidenceBasedAnalysis
from news_agent.models.analysis import SpeculativeRedTeamAnalysis
from news_agent.models.triage import Entities
from news_agent.models.triage import FactInferenceSpeculation
from news_agent.models.triage import MainClaim
from news_agent.models.triage import SourceFinding
from news_agent.models.triage import SourceProfile
from news_agent.models.triage import TriageBrief
from news_agent.services.reporting import default_report_path
from news_agent.services.reporting import write_html_report


class ReportTests(unittest.TestCase):
    def test_report_writer_includes_dynamic_findings(self) -> None:
        brief = TriageBrief(
            query="Which dates and compliance deadlines are being reported for global AI regulation updates?",
            main_claims=[
                MainClaim(
                    claim="Reported timelines differ across outlets.",
                    status="partly confirmed",
                    evidence_level="moderate",
                )
            ],
            entities=Entities(countries=["European Union", "United States"]),
            source_profiles=[
                SourceProfile(
                    name="Reuters",
                    country="United Kingdom",
                    type="wire",
                    orientation="straight news",
                    tone="analytical",
                )
            ],
            source_findings=[
                SourceFinding(
                    outlet_name="Reuters",
                    country="United Kingdom",
                    headline="Reuters reports updated AI regulation timeline",
                    url="https://example.com/reuters",
                    source_position="Reuters says the latest compliance dates remain provisional.",
                    reported_numbers=["2026", "2027"],
                    judgment="Useful because it gives explicit dates, but the timeline remains provisional.",
                    notes="The article attributes several dates to officials and regulatory documents.",
                )
            ],
            framing_analysis=["The framing stays close to attributed reporting."],
            historical_context=["Past technology regulation rollouts also produced disputed timelines."],
            uncertainties=["Dates are still moving."],
            fact_inference_speculation=FactInferenceSpeculation(
                observation=["Reuters listed two provisional dates."],
                evidence_backed_inference=["The timeline is not yet fully settled."],
                speculation=["Final dates may shift."],
            ),
            final_brief="Reuters provides explicit dates, but they still require corroboration.",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = default_report_path(brief.query, temp_dir)
            written = write_html_report(brief, report_path)
            html = written.read_text(encoding="utf-8")

        self.assertTrue(str(written).endswith(".html"))
        self.assertIn("Reuters", html)
        self.assertIn("2026", html)
        self.assertIn("2027", html)
        self.assertIn("Open source", html)

    def test_report_renders_evidence_based_analysis_section(self) -> None:
        brief = TriageBrief(
            query="What changed?",
            final_brief="Summary.",
            analysis_bundle=AnalysisBundle(
                evidence_based=EvidenceBasedAnalysis(
                    title="Evidence-based analysis",
                    overall_assessment="Evidence supports the main point.",
                    facts=["A supported fact."],
                    evidence_backed_inferences=["A careful inference."],
                    uncertainties=["A remaining uncertainty."],
                    source_disagreements=["A framing difference."],
                    confidence="medium",
                )
            ),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            html = write_html_report(
                brief,
                default_report_path(brief.query, temp_dir),
            ).read_text(encoding="utf-8")

        self.assertIn("Evidence-based analysis", html)
        self.assertIn("A supported fact.", html)
        self.assertIn("Evidence-backed inferences", html)
        self.assertIn("Confidence:</strong> medium", html)

    def test_report_renders_speculative_red_team_section(self) -> None:
        brief = TriageBrief(
            query="What changed?",
            final_brief="Summary.",
            analysis_bundle=AnalysisBundle(
                speculative_red_team=SpeculativeRedTeamAnalysis(
                    title="Speculative red-team lens",
                    core_suspicion="Speculatively, timing deserves scrutiny.",
                    adversarial_reading="A harsher alternative account.",
                    who_benefits=["An institution may benefit narratively."],
                    suspicious_patterns=["A suspicious pattern."],
                    possible_hidden_actors_or_incentives=["A possible incentive."],
                    speculative_hypotheses=["A clearly labeled hypothesis."],
                    mainstream_blind_spots=["A question not being asked."],
                    weaknesses_in_this_reading=["A weakness."],
                    evidence_needed=["A document."],
                    confidence="speculative",
                )
            ),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            html = write_html_report(
                brief,
                default_report_path(brief.query, temp_dir),
            ).read_text(encoding="utf-8")

        self.assertIn("Speculative red-team lens", html)
        self.assertIn("speculative red-team exercise", html)
        self.assertIn("A harsher alternative account.", html)
        self.assertIn("An institution may benefit narratively.", html)
        self.assertIn("A clearly labeled hypothesis.", html)
        self.assertIn("A question not being asked.", html)
        self.assertIn("Confidence:</strong> speculative", html)

    def test_report_escapes_model_output(self) -> None:
        brief = TriageBrief(
            query="What changed?",
            final_brief="Summary.",
            analysis_bundle=AnalysisBundle(
                evidence_based=EvidenceBasedAnalysis(
                    title="Evidence-based analysis",
                    overall_assessment="<script>alert(1)</script>",
                    facts=["<b>not html</b>"],
                )
            ),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            html = write_html_report(
                brief,
                default_report_path(brief.query, temp_dir),
            ).read_text(encoding="utf-8")

        self.assertNotIn("<script>alert(1)</script>", html)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", html)
        self.assertIn("&lt;b&gt;not html&lt;/b&gt;", html)

    def test_analysis_disabled_preserves_existing_report(self) -> None:
        brief = TriageBrief(
            query="What changed?",
            final_brief="Summary.",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            html = write_html_report(
                brief,
                default_report_path(brief.query, temp_dir),
            ).read_text(encoding="utf-8")

        self.assertIn("Summary.", html)
        self.assertNotIn("Speculative red-team lens", html)
        self.assertNotIn("Evidence-based analysis", html)


if __name__ == "__main__":
    unittest.main()
