from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import json
import logging
from typing import Callable

from news_agent.models.analysis import AnalysisBundle
from news_agent.models.analysis import EvidenceBasedAnalysis
from news_agent.models.analysis import SpeculativeRedTeamAnalysis
from news_agent.models.triage import ResearchBundle
from news_agent.models.triage import SourceFinding
from news_agent.models.triage import TriageBrief
from news_agent.services.debug.debug_output import DebugOutput
from news_agent.services.llm.text_generation import ModelGenerationError
from news_agent.services.llm.text_generation import ModelOutputError
from news_agent.services.llm.text_generation import TextGenerator
from news_agent.services.llm.text_generation import extract_json_block
from news_agent.services.prompts.prompt_service import PromptService


logger = logging.getLogger(__name__)


class AnalysisService:
    def __init__(
        self,
        *,
        prompt_service: PromptService,
        text_generator: TextGenerator,
        debug_output: DebugOutput | None = None,
        evidence_prompt: str = "analysis/evidence_based_analysis",
        speculative_prompt: str = "analysis/speculative_red_team_analysis",
        run_parallel: bool = True,
    ) -> None:
        self.prompt_service = prompt_service
        self.text_generator = text_generator
        self.debug_output = debug_output
        self.evidence_prompt = evidence_prompt
        self.speculative_prompt = speculative_prompt
        self.run_parallel = run_parallel

    def analyze(
        self,
        *,
        query: str,
        bundle: ResearchBundle,
        brief: TriageBrief,
    ) -> AnalysisBundle:
        evidence_bundle = self._build_evidence_bundle(
            query=query,
            bundle=bundle,
            brief=brief,
        )
        if self.run_parallel:
            analysis_bundle = self._run_parallel(evidence_bundle)
        else:
            analysis_bundle = AnalysisBundle(
                evidence_based=self._run_evidence_based(evidence_bundle),
                speculative_red_team=self._run_speculative_red_team(evidence_bundle),
            )

        if self.debug_output:
            self.debug_output.write_json(
                "analysis_bundle.json",
                analysis_bundle.to_dict(),
            )
        return analysis_bundle

    def _run_parallel(self, evidence_bundle: dict[str, object]) -> AnalysisBundle:
        results: dict[str, object | None] = {
            "evidence_based": None,
            "speculative_red_team": None,
        }
        tasks: dict[object, str] = {}
        with ThreadPoolExecutor(max_workers=2) as executor:
            tasks[
                executor.submit(self._run_evidence_based, evidence_bundle)
            ] = "evidence_based"
            tasks[
                executor.submit(self._run_speculative_red_team, evidence_bundle)
            ] = "speculative_red_team"
            for future in as_completed(tasks):
                results[tasks[future]] = future.result()

        return AnalysisBundle(
            evidence_based=(
                results["evidence_based"]
                if isinstance(results["evidence_based"], EvidenceBasedAnalysis)
                else None
            ),
            speculative_red_team=(
                results["speculative_red_team"]
                if isinstance(
                    results["speculative_red_team"],
                    SpeculativeRedTeamAnalysis,
                )
                else None
            ),
        )

    def _run_evidence_based(
        self,
        evidence_bundle: dict[str, object],
    ) -> EvidenceBasedAnalysis | None:
        return self._run_agent(
            call_name="evidence_based_analysis",
            prompt_name=self.evidence_prompt,
            evidence_bundle=evidence_bundle,
            parser=_parse_evidence_based,
        )

    def _run_speculative_red_team(
        self,
        evidence_bundle: dict[str, object],
    ) -> SpeculativeRedTeamAnalysis | None:
        return self._run_agent(
            call_name="speculative_red_team_analysis",
            prompt_name=self.speculative_prompt,
            evidence_bundle=evidence_bundle,
            parser=_parse_speculative_red_team,
        )

    def _run_agent(
        self,
        *,
        call_name: str,
        prompt_name: str,
        evidence_bundle: dict[str, object],
        parser: Callable[[dict], object],
    ) -> object | None:
        evidence_bundle_json = json.dumps(
            evidence_bundle,
            ensure_ascii=False,
            indent=2,
        )
        prompt = self.prompt_service.build(
            prompt_name,
            evidence_bundle_json=evidence_bundle_json,
        )
        if self.debug_output:
            self.debug_output.write_text(f"{call_name}_input.txt", prompt)

        try:
            result = self.text_generator.generate(prompt)
            text = getattr(result, "text", str(result))
            if self.debug_output:
                self.debug_output.write_text(f"{call_name}_output.txt", text)
            payload = json.loads(extract_json_block(text))
            if not isinstance(payload, dict):
                raise ModelOutputError("Analysis agent returned non-object JSON.")
            return parser(payload)
        except (
            ModelGenerationError,
            ModelOutputError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            logger.info("%s skipped after analysis error: %s", call_name, exc)
            if self.debug_output:
                self.debug_output.write_text(
                    f"{call_name}_error.txt",
                    f"{type(exc).__name__}: {exc}",
                )
            return None

    def _build_evidence_bundle(
        self,
        *,
        query: str,
        bundle: ResearchBundle,
        brief: TriageBrief,
    ) -> dict[str, object]:
        findings_by_url = {
            finding.url: finding
            for finding in brief.source_findings
            if finding.url
        }
        findings_by_outlet = {
            finding.outlet_name: finding
            for finding in brief.source_findings
            if finding.outlet_name
        }
        return {
            "user_query": query,
            "final_summary": brief.final_brief,
            "requested_metric": (
                bundle.intent.requested_metric
                if bundle.intent is not None
                else ""
            ),
            "selected_sources": [
                _source_payload(
                    article=article,
                    finding=(
                        findings_by_url.get(article.url)
                        or findings_by_outlet.get(article.outlet_name)
                    ),
                )
                for article in bundle.articles
            ],
            "known_uncertainties": list(brief.uncertainties),
            "source_disagreements": list(brief.framing_analysis),
        }


def _source_payload(
    *,
    article: object,
    finding: SourceFinding | None,
) -> dict[str, object]:
    extracted_metric = ""
    if bool(getattr(article, "metric_found", False)):
        extracted_metric = " | ".join(
            part
            for part in [
                str(getattr(article, "metric_type", "")).strip(),
                str(getattr(article, "metric_value", "")).strip(),
                str(getattr(article, "metric_evidence", "")).strip(),
                str(getattr(article, "metric_confidence", "")).strip(),
            ]
            if part
        )

    source_framing = ""
    if finding is not None:
        source_framing = _trim_text(
            " ".join(
                part
                for part in [
                    finding.source_position,
                    finding.judgment,
                    finding.notes,
                ]
                if part
            ),
            limit=700,
        )

    return {
        "outlet_name": str(getattr(article, "outlet_name", "")),
        "title": str(getattr(article, "title", "")),
        "url": str(getattr(article, "url", "")),
        "published_at": str(getattr(article, "published_at", "") or ""),
        "article_text": _trim_text(
            str(
                getattr(article, "article_text", "")
                or getattr(article, "snippet", "")
            ),
            limit=1800,
        ),
        "extracted_metric": extracted_metric,
        "source_framing": source_framing,
    }


def _parse_evidence_based(payload: dict) -> EvidenceBasedAnalysis | None:
    bundle = AnalysisBundle.from_dict({"evidence_based": payload})
    return bundle.evidence_based if bundle else None


def _parse_speculative_red_team(payload: dict) -> SpeculativeRedTeamAnalysis | None:
    bundle = AnalysisBundle.from_dict({"speculative_red_team": payload})
    return bundle.speculative_red_team if bundle else None


def _trim_text(text: str, *, limit: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."
