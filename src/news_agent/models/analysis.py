from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from typing import Literal


Confidence = Literal["high", "medium", "low", "very low", "speculative"]


@dataclass(frozen=True, slots=True)
class EvidenceBasedAnalysis:
    title: str
    overall_assessment: str
    facts: list[str] = field(default_factory=list)
    evidence_backed_inferences: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    source_disagreements: list[str] = field(default_factory=list)
    confidence: Confidence = "medium"


@dataclass(frozen=True, slots=True)
class SpeculativeRedTeamAnalysis:
    title: str
    core_suspicion: str
    adversarial_reading: str = ""
    who_benefits: list[str] = field(default_factory=list)
    suspicious_patterns: list[str] = field(default_factory=list)
    possible_hidden_actors_or_incentives: list[str] = field(default_factory=list)
    speculative_hypotheses: list[str] = field(default_factory=list)
    mainstream_blind_spots: list[str] = field(default_factory=list)
    weaknesses_in_this_reading: list[str] = field(default_factory=list)
    evidence_needed: list[str] = field(default_factory=list)
    confidence: Confidence = "speculative"


@dataclass(frozen=True, slots=True)
class AnalysisBundle:
    evidence_based: EvidenceBasedAnalysis | None = None
    speculative_red_team: SpeculativeRedTeamAnalysis | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: object) -> "AnalysisBundle | None":
        if not isinstance(data, dict):
            return None

        evidence_payload = data.get("evidence_based")
        speculative_payload = data.get("speculative_red_team")
        return cls(
            evidence_based=(
                _evidence_based_from_dict(evidence_payload)
                if isinstance(evidence_payload, dict)
                else None
            ),
            speculative_red_team=(
                _speculative_from_dict(speculative_payload)
                if isinstance(speculative_payload, dict)
                else None
            ),
        )


def _evidence_based_from_dict(data: dict) -> EvidenceBasedAnalysis:
    return EvidenceBasedAnalysis(
        title=str(data.get("title", "Evidence-based analysis")),
        overall_assessment=str(data.get("overall_assessment", "")),
        facts=_string_list(data.get("facts", [])),
        evidence_backed_inferences=_string_list(
            data.get("evidence_backed_inferences", [])
        ),
        uncertainties=_string_list(data.get("uncertainties", [])),
        source_disagreements=_string_list(data.get("source_disagreements", [])),
        confidence=_confidence(
            data.get("confidence", "medium"),
            allowed={"high", "medium", "low"},
            fallback="medium",
        ),
    )


def _speculative_from_dict(data: dict) -> SpeculativeRedTeamAnalysis:
    return SpeculativeRedTeamAnalysis(
        title=str(data.get("title", "Speculative red-team lens")),
        core_suspicion=str(data.get("core_suspicion", "")),
        adversarial_reading=str(data.get("adversarial_reading", "")),
        who_benefits=_string_list(data.get("who_benefits", [])),
        suspicious_patterns=_string_list(data.get("suspicious_patterns", [])),
        possible_hidden_actors_or_incentives=_string_list(
            data.get("possible_hidden_actors_or_incentives", [])
        ),
        speculative_hypotheses=_string_list(data.get("speculative_hypotheses", [])),
        mainstream_blind_spots=_string_list(data.get("mainstream_blind_spots", [])),
        weaknesses_in_this_reading=_string_list(
            data.get("weaknesses_in_this_reading", [])
        ),
        evidence_needed=_string_list(data.get("evidence_needed", [])),
        confidence=_confidence(
            data.get("confidence", "speculative"),
            allowed={"low", "very low", "speculative"},
            fallback="speculative",
        ),
    )


def _string_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _confidence(
    value: object,
    *,
    allowed: set[str],
    fallback: Confidence,
) -> Confidence:
    normalized = str(value).strip().lower()
    if normalized in allowed:
        return normalized  # type: ignore[return-value]
    return fallback
