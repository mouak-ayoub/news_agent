from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field


@dataclass(slots=True)
class ResearchIntent:
    """Structured view of what the user is trying to find."""

    topic: str
    requested_metric: str
    expected_answer_type: str
    time_sensitivity: str
    must_find: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None, fallback_query: str = "") -> "ResearchIntent":
        if not isinstance(data, dict):
            return default_research_intent(fallback_query)
        return cls(
            topic=str(data.get("topic", fallback_query)),
            requested_metric=str(data.get("requested_metric", fallback_query)),
            expected_answer_type=str(data.get("expected_answer_type", "unknown")),
            time_sensitivity=str(data.get("time_sensitivity", "unspecified")),
            must_find=_string_list(data.get("must_find", [])),
            avoid=_string_list(data.get("avoid", [])),
        )


@dataclass(slots=True)
class SearchPlan:
    """Search queries derived from the intent, ordered from strict to broad."""

    queries: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None, fallback_query: str = "") -> "SearchPlan":
        if not isinstance(data, dict):
            return cls(queries=[fallback_query] if fallback_query else [])
        queries = _string_list(data.get("queries", []))
        if not queries and fallback_query:
            queries = [fallback_query]
        return cls(queries=queries)


@dataclass(slots=True)
class MetricExtraction:
    """Metric extracted from one selected article."""

    metric_found: bool
    value: str = ""
    metric_type: str = ""
    evidence: str = ""
    confidence: str = "low"
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "MetricExtraction":
        if not isinstance(data, dict):
            return cls(metric_found=False)
        return cls(
            metric_found=bool(data.get("metric_found", False)),
            value=str(data.get("value", "")),
            metric_type=str(data.get("metric_type", "")),
            evidence=str(data.get("evidence", "")),
            confidence=str(data.get("confidence", "low")),
            notes=str(data.get("notes", "")),
        )


def default_research_intent(query: str) -> ResearchIntent:
    return ResearchIntent(
        topic=query,
        requested_metric=query,
        expected_answer_type="unknown",
        time_sensitivity="latest" if _looks_current(query) else "unspecified",
        must_find=[],
        avoid=[],
    )


def _looks_current(query: str) -> bool:
    lowered = query.lower()
    return any(word in lowered for word in ("latest", "current", "today", "now", "recent"))


def _string_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []
