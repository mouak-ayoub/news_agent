from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from news_agent.html_report import default_report_path
from news_agent.html_report import write_html_report
from news_agent.schemas import Entities
from news_agent.schemas import FactInferenceSpeculation
from news_agent.schemas import MainClaim
from news_agent.schemas import SourceFinding
from news_agent.schemas import SourceProfile
from news_agent.schemas import TriageBrief


class ReportTests(unittest.TestCase):
    def test_report_writer_includes_dynamic_findings(self) -> None:
        brief = TriageBrief(
            query="What are the casualties in the Iran war across all participating countries?",
            main_claims=[
                MainClaim(
                    claim="Casualty figures differ across outlets.",
                    status="partly confirmed",
                    evidence_level="moderate",
                )
            ],
            entities=Entities(countries=["Iran", "Israel"]),
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
                    headline="Reuters reports updated casualty figures",
                    url="https://example.com/reuters",
                    source_position="Reuters says the latest available figures remain provisional.",
                    reported_numbers=["120", "430"],
                    judgment="Useful because it gives explicit figures, but the figures remain provisional.",
                    notes="The article attributes several figures to officials and monitoring groups.",
                )
            ],
            framing_analysis=["The framing stays close to attributed reporting."],
            historical_context=["Past regional wars also produced disputed casualty counts."],
            uncertainties=["Numbers are still moving."],
            fact_inference_speculation=FactInferenceSpeculation(
                observation=["Reuters listed two provisional figures."],
                evidence_backed_inference=["The totals are not yet fully settled."],
                speculation=["Final totals may rise."],
            ),
            final_brief="Reuters provides explicit figures, but they still require corroboration.",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = default_report_path(brief.query, temp_dir)
            written = write_html_report(brief, report_path)
            html = written.read_text(encoding="utf-8")

        self.assertTrue(str(written).endswith(".html"))
        self.assertIn("Reuters", html)
        self.assertIn("120", html)
        self.assertIn("430", html)
        self.assertIn("Open source", html)


if __name__ == "__main__":
    unittest.main()
