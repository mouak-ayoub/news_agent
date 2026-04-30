from __future__ import annotations

from dataclasses import asdict
import json
import logging
import re

from ..models.config import AppConfig
from ..models.triage import Entities
from ..models.triage import FactInferenceSpeculation
from ..models.triage import MainClaim
from ..models.triage import ResearchBundle
from ..models.triage import SourceFinding
from ..models.triage import SourceProfile
from ..models.triage import TriageBrief
from ..models.generation import GenerationResult
from .prompt_service import PromptService
from .text_generation import ModelGenerationError
from .text_generation import ModelOutputError
from .text_generation import TextGenerator
from .text_generation import extract_json_block

NUMBER_PATTERN = re.compile(
    r"\b\d[\d,]*(?:\.\d+)?(?:\s?(?:k|m|b|million|billion|thousand|percent|%))?\b",
    re.IGNORECASE,
)

logger = logging.getLogger(__name__)


class SummarizationService:
    def __init__(
        self,
        config: AppConfig,
        text_generator: TextGenerator,
        prompt_service: PromptService | None = None,
    ) -> None:
        self.config = config
        self.text_generator = text_generator
        self.prompt_service = prompt_service or PromptService()

    def summarize(self, query: str, bundle: ResearchBundle) -> TriageBrief:
        logger.info(
            "summarization started query=%r articles=%d",
            query,
            len(bundle.articles),
        )
        try:
            result: GenerationResult = self.text_generator.generate(
                self._build_prompt(query, bundle)
            )
            payload = json.loads(extract_json_block(result.text))
            brief = self._normalize(query, bundle, payload)
            logger.info("summarization finished mode=model")
            return brief
        except ModelGenerationError:
            logger.exception("summarization failed because model generation failed")
            raise
        except (ModelOutputError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            if not self.config.fallback_to_heuristic:
                logger.exception("summarization failed because model output was unusable")
                raise
            logger.warning("summarization output unusable; using heuristic summary")
            brief = self._heuristic(query, bundle)
            logger.info("summarization finished mode=heuristic")
            return brief

    def _build_prompt(self, query: str, bundle: ResearchBundle) -> str:
        article_payload = [asdict(article) for article in bundle.articles]
        return self.prompt_service.build(
            "summarization",
            query=query,
            article_payload_json=json.dumps(article_payload, ensure_ascii=False, indent=2),
        )

    def _normalize(self, query: str, bundle: ResearchBundle, payload: dict) -> TriageBrief:
        payload["query"] = query
        payload.setdefault("main_claims", [])
        payload.setdefault("entities", {})
        payload.setdefault("source_profiles", [])
        payload.setdefault("source_findings", [])
        payload.setdefault("framing_analysis", [])
        payload.setdefault("historical_context", [])
        payload.setdefault("uncertainties", [])
        payload.setdefault("fact_inference_speculation", {})
        payload.setdefault("final_brief", "")

        if not isinstance(payload["main_claims"], list):
            payload["main_claims"] = []
        payload["main_claims"] = [
            (
                {
                    "claim": str(item.get("claim", "")),
                    "status": str(item.get("status", "unclear / not enough evidence")),
                    "evidence_level": str(item.get("evidence_level", "low")),
                }
                if isinstance(item, dict)
                else {
                    "claim": str(item),
                    "status": "unclear / not enough evidence",
                    "evidence_level": "low",
                }
            )
            for item in payload["main_claims"]
            if str(item).strip()
        ]

        if not isinstance(payload["entities"], dict):
            payload["entities"] = {}
        payload["entities"] = {
            "countries": _string_list(
                payload["entities"].get("countries", [article.country for article in bundle.articles])
            ),
            "people": _string_list(payload["entities"].get("people", [])),
            "organizations": _string_list(payload["entities"].get("organizations", [])),
            "locations": _string_list(payload["entities"].get("locations", [])),
        }

        if not payload["source_profiles"]:
            payload["source_profiles"] = self._build_source_profiles(bundle)
        else:
            payload["source_profiles"] = [
                {
                    "name": str(item.get("name", item.get("outlet_name", ""))),
                    "country": str(item.get("country", "")),
                    "type": str(item.get("type", item.get("medium_type", ""))),
                    "orientation": str(item.get("orientation", "")),
                    "tone": str(item.get("tone", "analytical")),
                }
                for item in payload["source_profiles"]
                if isinstance(item, dict)
            ]

        if not payload["source_findings"]:
            payload["source_findings"] = self._build_source_findings(bundle)
        else:
            payload["source_findings"] = [
                {
                    "outlet_name": str(item.get("outlet_name", "")),
                    "country": str(item.get("country", "")),
                    "headline": str(item.get("headline", "")),
                    "url": str(item.get("url", "")),
                    "source_position": str(item.get("source_position", "")),
                    "reported_numbers": [
                        str(number).strip()
                        for number in item.get("reported_numbers", [])
                        if str(number).strip()
                    ],
                    "judgment": str(item.get("judgment", "")),
                    "notes": str(item.get("notes", "")),
                }
                for item in payload["source_findings"]
                if isinstance(item, dict)
            ]

        framing_analysis = payload["framing_analysis"]
        if isinstance(framing_analysis, dict):
            payload["framing_analysis"] = _string_list(framing_analysis.values())
        else:
            payload["framing_analysis"] = _string_list(framing_analysis)

        payload["historical_context"] = _string_list(payload["historical_context"])
        payload["uncertainties"] = _string_list(payload["uncertainties"])

        if not isinstance(payload["fact_inference_speculation"], dict):
            payload["fact_inference_speculation"] = {}
        payload["fact_inference_speculation"] = {
            "observation": _string_list(
                payload["fact_inference_speculation"].get("observation", [])
            ),
            "evidence_backed_inference": _string_list(
                payload["fact_inference_speculation"].get(
                    "evidence_backed_inference",
                    [],
                )
            ),
            "speculation": _string_list(
                payload["fact_inference_speculation"].get("speculation", [])
            ),
        }
        payload["final_brief"] = str(payload.get("final_brief", ""))
        return TriageBrief.from_dict(payload)

    def _heuristic(self, query: str, bundle: ResearchBundle) -> TriageBrief:
        source_profiles = [
            SourceProfile(**profile) for profile in self._build_source_profiles(bundle)
        ]
        source_findings = [
            SourceFinding(**finding) for finding in self._build_source_findings(bundle)
        ]
        claims = [
            MainClaim(
                claim=article.title,
                status=(
                    "unclear / not enough evidence"
                    if len(bundle.articles) < 3
                    else "partly confirmed"
                ),
                evidence_level="low" if len(bundle.articles) < 3 else "moderate",
            )
            for article in bundle.articles[:3]
        ]
        observations = [
            f"{finding.outlet_name}: {finding.source_position or finding.headline}"
            for finding in source_findings[:3]
        ]
        inferences = []
        has_explicit_numbers = any(finding.reported_numbers for finding in source_findings)
        if len({article.country for article in bundle.articles}) > 1:
            inferences.append(
                "Coverage differences likely reflect each outlet's local context, audience, and editorial priorities."
            )
        if len({article.orientation for article in bundle.articles}) > 1:
            inferences.append(
                "Differences in outlet orientation suggest the story may be framed through different assumptions or priorities."
            )
        if has_explicit_numbers:
            inferences.append(
                "Some outlets provide explicit figures, but they still need cross-checking before being treated as settled totals."
            )
        if not inferences:
            inferences.append(
                "The current source set is narrow, so framing conclusions should stay tentative."
            )
        speculation = [
            "Some silence or emphasis may be strategic, but the available evidence is insufficient to present motive as fact."
        ]
        uncertainties = ["This v1 uses a single retrieval round and may miss later updates."]
        if len(bundle.articles) < 3:
            uncertainties.append("Fewer than three strong sources were retrieved.")

        return TriageBrief(
            query=query,
            main_claims=claims,
            entities=Entities(countries=sorted({a.country for a in bundle.articles})),
            source_profiles=source_profiles,
            source_findings=source_findings,
            framing_analysis=inferences,
            historical_context=[
                "Recent reporting can change quickly, so source dates and update times matter.",
                "Outlet framing often reflects audience, ownership, access to sources, and editorial standards.",
            ],
            uncertainties=uncertainties,
            fact_inference_speculation=FactInferenceSpeculation(
                observation=observations,
                evidence_backed_inference=inferences,
                speculation=speculation,
            ),
            final_brief=(
                "Retrieved outlets offer a partial answer, and the safest read is to compare each source's wording and treat uncross-checked figures cautiously."
                if bundle.articles and has_explicit_numbers
                else (
                    "The topic appears to be actively reported, but the main divergence is in what outlets emphasize, omit, or contextualize."
                    if bundle.articles
                    else "No strong recent sources were retrieved, so the brief cannot support a confident triage judgment."
                )
            ),
        )

    def _build_source_profiles(self, bundle: ResearchBundle) -> list[dict]:
        """Build profile rows for sources that were actually part of this run."""
        profiles: list[dict] = []
        if self._reports_configured_outlets():
            profiles.extend(
                {
                    "name": outlet.name,
                    "country": outlet.country,
                    "type": outlet.medium_type,
                    "orientation": outlet.orientation,
                    "tone": "analytical",
                }
                for outlet in self.config.outlets
            )
        seen_names = {profile["name"] for profile in profiles}
        for article in bundle.articles:
            if article.outlet_name in seen_names:
                continue
            profiles.append(
                {
                    "name": article.outlet_name,
                    "country": article.country,
                    "type": article.medium_type,
                    "orientation": article.orientation,
                    "tone": "analytical",
                }
            )
            seen_names.add(article.outlet_name)
        return profiles

    def _build_source_findings(self, bundle: ResearchBundle) -> list[dict]:
        """Build one finding per expected outlet or per retrieved publisher."""
        articles_by_outlet = {article.outlet_name: article for article in bundle.articles}
        findings = []
        covered_outlets: set[str] = set()

        if self._reports_configured_outlets():
            for outlet in self.config.outlets:
                article = articles_by_outlet.get(outlet.name)
                if article is None:
                    findings.append(
                        {
                            "outlet_name": outlet.name,
                            "country": outlet.country,
                            "headline": "No strong recent article retrieved",
                            "url": "",
                            "source_position": "No strong result was found for this outlet in the current search pass.",
                            "reported_numbers": [],
                            "judgment": "No judgment on the topic is possible from this outlet yet because retrieval did not return a usable article.",
                            "notes": "The report keeps this outlet visible so absence is explicit rather than hidden.",
                        }
                    )
                    continue

                covered_outlets.add(outlet.name)
                numbers = self._article_numbers(article)
                if article.metric_found:
                    judgment = "Useful because it reports the requested metric, but the figure should still be cross-checked."
                    source_position = article.metric_evidence or article.snippet.strip() or article.title.strip()
                elif numbers:
                    judgment = "Contains explicit figures, but they may not be the exact metric requested."
                    source_position = article.snippet.strip() or article.title.strip()
                else:
                    judgment = "Useful for framing or context, but it does not give a precise numeric answer to the query."
                    source_position = article.snippet.strip() or article.title.strip()
                findings.append(
                    {
                        "outlet_name": article.outlet_name,
                        "country": article.country,
                        "headline": article.title,
                        "url": article.url,
                        "source_position": source_position,
                        "reported_numbers": numbers,
                        "judgment": judgment,
                        "notes": article.metric_notes or self._trim_text(article.article_text or article.snippet),
                    }
                )

        for article in bundle.articles:
            if article.outlet_name in covered_outlets:
                continue
            numbers = self._article_numbers(article)
            if article.metric_found:
                judgment = "Useful because it reports the requested metric, but the figure should still be cross-checked."
                source_position = article.metric_evidence or article.snippet.strip() or article.title.strip()
            elif numbers:
                judgment = "Contains explicit figures, but they may not be the exact metric requested."
                source_position = article.snippet.strip() or article.title.strip()
            else:
                judgment = "Useful for framing or context, but it does not give a precise numeric answer to the query."
                source_position = article.snippet.strip() or article.title.strip()
            findings.append(
                {
                    "outlet_name": article.outlet_name,
                    "country": article.country,
                    "headline": article.title,
                    "url": article.url,
                    "source_position": source_position,
                    "reported_numbers": numbers,
                    "judgment": judgment,
                    "notes": article.metric_notes or self._trim_text(article.article_text or article.snippet),
                }
            )
        return findings

    def _reports_configured_outlets(self) -> bool:
        """Use configured outlet placeholders only for outlet-scoped providers."""
        return self.config.search.provider in {"google_news_rss", "openai_web_search"}

    def _article_numbers(self, article: object) -> list[str]:
        metric_value = str(getattr(article, "metric_value", "")).strip()
        if bool(getattr(article, "metric_found", False)) and metric_value:
            return [metric_value]
        return self._extract_numbers(
            " ".join(
                part
                for part in [
                    getattr(article, "title", ""),
                    getattr(article, "snippet", ""),
                    getattr(article, "article_text", ""),
                ]
                if part
            )
        )

    def _extract_numbers(self, text: str) -> list[str]:
        seen: set[str] = set()
        numbers: list[str] = []
        for match in NUMBER_PATTERN.findall(text):
            cleaned = " ".join(match.split())
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                numbers.append(cleaned)
        return numbers

    def _trim_text(self, text: str, limit: int = 220) -> str:
        normalized = " ".join(text.split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3].rstrip() + "..."


def _string_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []
