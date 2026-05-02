from __future__ import annotations

from dataclasses import dataclass

from ...models.config import OutletConfig
from ...models.research import SearchPlan


@dataclass(frozen=True, slots=True)
class WebSearchJob:
    """One concrete OpenAI web-search job."""

    search_query: str
    outlets: tuple[OutletConfig, ...]


class OpenAISearchJobPlanner:
    """Build deterministic site-filtered search jobs."""

    def build_jobs(
        self,
        *,
        query: str,
        plan: SearchPlan | None,
        outlets: list[OutletConfig],
        max_calls: int,
    ) -> list[WebSearchJob]:
        if not outlets:
            return []

        call_limit = max(1, max_calls)
        planned_queries = _planned_queries(query, plan)
        jobs: list[WebSearchJob] = [
            WebSearchJob(
                search_query=_scoped_query(planned_queries[0], outlets),
                outlets=tuple(outlets),
            )
        ]
        if len(jobs) >= call_limit:
            return jobs

        seen_queries: set[str] = {jobs[0].search_query}
        outlet_groups = _outlet_groups(outlets, call_limit - len(jobs))
        for planned_query in planned_queries:
            for group in outlet_groups:
                scoped_query = _scoped_query(planned_query, group)
                if scoped_query in seen_queries:
                    continue
                jobs.append(
                    WebSearchJob(
                        search_query=scoped_query,
                        outlets=tuple(group),
                    )
                )
                seen_queries.add(scoped_query)
                if len(jobs) >= call_limit:
                    return jobs
        return jobs


def _outlet_groups(
    outlets: list[OutletConfig],
    group_limit: int,
) -> list[list[OutletConfig]]:
    """Split outlets according to the configured search-call budget."""
    group_count = min(max(1, group_limit), len(outlets))
    base_size = len(outlets) // group_count
    remainder = len(outlets) % group_count

    groups: list[list[OutletConfig]] = []
    start = 0
    for index in range(group_count):
        size = base_size + (1 if index < remainder else 0)
        groups.append(outlets[start : start + size])
        start += size
    return groups


def _planned_queries(query: str, plan: SearchPlan | None) -> list[str]:
    """Prefer planner keyword queries and keep the raw question as fallback."""
    values: list[str] = []
    if plan:
        values.extend(plan.queries)
    values.append(query)

    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = " ".join(str(value).split())
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        deduped.append(normalized)
        seen.add(key)

    if not deduped:
        deduped = [query]
    return sorted(deduped, key=_looks_like_question)


def _looks_like_question(value: str) -> bool:
    lowered = value.strip().lower()
    return lowered.endswith("?") or lowered.startswith(
        (
            "what ",
            "why ",
            "how ",
            "when ",
            "where ",
            "who ",
            "which ",
            "is ",
            "are ",
            "do ",
            "does ",
            "did ",
            "can ",
            "could ",
        )
    )


def _scoped_query(query: str, outlets: list[OutletConfig]) -> str:
    """Add explicit outlet domain filters to one keyword query."""
    return f"{query} {_domain_filter(outlets)}".strip()


def _domain_filter(outlets: list[OutletConfig]) -> str:
    return " OR ".join(f"site:{outlet.domain}" for outlet in outlets)

