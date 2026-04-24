from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import json
import re

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.genai import types
from typing_extensions import override

from ..config import AppConfig
from ..model import ModelGenerationError
from ..model import TextGenerator
from ..model import extract_json_block
from ..schemas import Entities
from ..schemas import FactInferenceSpeculation
from ..schemas import MainClaim
from ..schemas import ResearchBundle
from ..schemas import SourceFinding
from ..schemas import SourceProfile
from ..schemas import TriageBrief
from ..usage import BudgetExceededError
from ..usage import UsageGuard

NUMBER_PATTERN = re.compile(
    r"\b\d[\d,]*(?:\.\d+)?(?:\s?(?:k|m|b|million|billion|thousand|percent|%))?\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class SummarizationService:
    config: AppConfig
    text_generator: TextGenerator
    usage_guard: UsageGuard

    def summarize(self, query: str, bundle: ResearchBundle) -> TriageBrief:
        try:
            result = self.text_generator.generate(self._build_prompt(query, bundle))
            self.usage_guard.record("summarize", result.usage)
            payload = json.loads(extract_json_block(result.text))
            return self._normalize(query, bundle, payload)
        except (BudgetExceededError, ModelGenerationError):
            if not self.config.model.fallback_to_heuristic:
                raise
            return self._heuristic(query, bundle)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            if not self.config.model.fallback_to_heuristic:
                raise
            return self._heuristic(query, bundle)

    def _build_prompt(self, query: str, bundle: ResearchBundle) -> str:
        article_payload = [asdict(article) for article in bundle.articles]
        return f"""
Return JSON only.

Task:
- analyze the recent reporting for the query
- extract main claims and entities
- list what each source says in relation to the query
- extract any explicit numbers reported by each source
- judge each source's usefulness and specificity for the query
- compare framing across sources
- separate observation, evidence-backed inference, and speculation
- keep historical context short

Required keys:
query, main_claims, entities, source_profiles, source_findings, framing_analysis,
historical_context, uncertainties, fact_inference_speculation, final_brief

For each item in source_findings return:
- outlet_name
- country
- headline
- url
- source_position
- reported_numbers (list of strings, empty list if none)
- judgment
- notes

Query:
{query}

Sources:
{json.dumps(article_payload, ensure_ascii=False, indent=2)}
""".strip()

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

        if not payload["source_profiles"]:
            payload["source_profiles"] = self._build_source_profiles(bundle)
        else:
            payload["source_profiles"] = [
                {
                    "name": str(item.get("name", "")),
                    "country": str(item.get("country", "")),
                    "type": str(item.get("type", "")),
                    "orientation": str(item.get("orientation", "")),
                    "tone": str(item.get("tone", "analytical")),
                }
                for item in payload["source_profiles"]
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
            ]
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
                status="unclear / not enough evidence" if len(bundle.articles) < 3 else "partly confirmed",
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
            inferences.append("Coverage differences likely reflect national context and alliance-sensitive framing.")
        if len({article.orientation for article in bundle.articles}) > 1:
            inferences.append("Differences in outlet orientation suggest competing narratives about justification, legality, or escalation.")
        if has_explicit_numbers:
            inferences.append("Some outlets provide explicit figures, but they still need cross-checking before being treated as settled totals.")
        if not inferences:
            inferences.append("The current source set is narrow, so framing conclusions should stay tentative.")
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
                "Regional conflicts are often framed differently through security, legality, and civilian-impact narratives.",
                "Repeated crises make outlet alignment and state interests especially visible in headline choices.",
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
                    "The event appears to be actively reported, but the main divergence is in how outlets justify, condemn, or contextualize the escalation."
                    if bundle.articles
                    else "No strong recent sources were retrieved, so the brief cannot support a confident triage judgment."
                )
            ),
        )

    def _build_source_profiles(self, bundle: ResearchBundle) -> list[dict]:
        profiles: list[dict] = [
            {
                "name": outlet.name,
                "country": outlet.country,
                "type": outlet.medium_type,
                "orientation": outlet.orientation,
                "tone": "analytical",
            }
            for outlet in self.config.outlets
        ]
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
        articles_by_outlet = {article.outlet_name: article for article in bundle.articles}
        findings = []
        covered_outlets: set[str] = set()

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
            numbers = self._extract_numbers(
                " ".join(
                    part
                    for part in [article.title, article.snippet, article.article_text]
                    if part
                )
            )
            if numbers:
                judgment = "Useful because it gives explicit figures, but those figures should still be cross-checked against other outlets."
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
                    "notes": self._trim_text(article.article_text or article.snippet),
                }
            )

        for article in bundle.articles:
            if article.outlet_name in covered_outlets:
                continue
            numbers = self._extract_numbers(
                " ".join(
                    part
                    for part in [article.title, article.snippet, article.article_text]
                    if part
                )
            )
            if numbers:
                judgment = "Useful because it gives explicit figures, but those figures should still be cross-checked against other outlets."
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
                    "notes": self._trim_text(article.article_text or article.snippet),
                }
            )
        return findings

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


class SummarizerAgent(BaseAgent):
    service: SummarizationService

    @override
    async def _run_async_impl(self, ctx: InvocationContext):
        query = str(ctx.session.state.get("query", ""))
        bundle = ResearchBundle.from_dict(
            ctx.session.state.get("research_bundle", {"query": query, "articles": []})
        )
        brief = self.service.summarize(query, bundle)
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            actions=EventActions(state_delta={"triage_brief": brief.to_dict()}),
            content=types.Content(
                role="model",
                parts=[types.Part(text="SummarizerAgent prepared the final brief.")],
            ),
        )
