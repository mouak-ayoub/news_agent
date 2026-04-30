from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
