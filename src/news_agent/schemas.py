from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
import json


@dataclass(slots=True)
class MainClaim:
    claim: str
    status: str
    evidence_level: str


@dataclass(slots=True)
class SourceProfile:
    name: str
    country: str
    type: str
    orientation: str
    tone: str


@dataclass(slots=True)
class SourceFinding:
    outlet_name: str
    country: str
    headline: str
    url: str
    source_position: str = ""
    reported_numbers: list[str] = field(default_factory=list)
    judgment: str = ""
    notes: str = ""


@dataclass(slots=True)
class Entities:
    countries: list[str] = field(default_factory=list)
    people: list[str] = field(default_factory=list)
    organizations: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FactInferenceSpeculation:
    observation: list[str] = field(default_factory=list)
    evidence_backed_inference: list[str] = field(default_factory=list)
    speculation: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ArticleRecord:
    title: str
    url: str
    outlet_name: str
    domain: str
    country: str
    medium_type: str
    orientation: str
    published_at: str | None
    snippet: str
    article_text: str
    search_query: str


@dataclass(slots=True)
class ResearchBundle:
    query: str
    articles: list[ArticleRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ResearchBundle":
        return cls(
            query=data["query"],
            articles=[ArticleRecord(**item) for item in data.get("articles", [])],
        )


@dataclass(slots=True)
class TriageBrief:
    query: str
    main_claims: list[MainClaim] = field(default_factory=list)
    entities: Entities = field(default_factory=Entities)
    source_profiles: list[SourceProfile] = field(default_factory=list)
    source_findings: list[SourceFinding] = field(default_factory=list)
    framing_analysis: list[str] = field(default_factory=list)
    historical_context: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    fact_inference_speculation: FactInferenceSpeculation = field(
        default_factory=FactInferenceSpeculation
    )
    final_brief: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_pretty_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "TriageBrief":
        return cls(
            query=data["query"],
            main_claims=[MainClaim(**item) for item in data.get("main_claims", [])],
            entities=Entities(**data.get("entities", {})),
            source_profiles=[
                SourceProfile(**item) for item in data.get("source_profiles", [])
            ],
            source_findings=[
                SourceFinding(**item) for item in data.get("source_findings", [])
            ],
            framing_analysis=data.get("framing_analysis", []),
            historical_context=data.get("historical_context", []),
            uncertainties=data.get("uncertainties", []),
            fact_inference_speculation=FactInferenceSpeculation(
                **data.get("fact_inference_speculation", {})
            ),
            final_brief=data.get("final_brief", ""),
        )
