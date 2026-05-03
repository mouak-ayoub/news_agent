from __future__ import annotations

from dataclasses import dataclass

from news_agent.models.config import OutletConfig
from news_agent.models.research import SearchPlan
from .domain_utils import normalize_allowed_domain


@dataclass(frozen=True, slots=True)
class WebSearchJob:
    """One concrete OpenAI web-search job."""

    search_query: str
    outlets: tuple[OutletConfig, ...]
    allowed_domains: tuple[str, ...] = ()


class OpenAISearchJobPlanner:
    """Build deterministic site-filtered search jobs."""

    def build_jobs(
        self,
        *,
        query: str,
        plan: SearchPlan | None,
        outlets: list[OutletConfig],
        max_calls: int,
        use_allowed_domains: bool = True,
        use_site_query_filters: bool = False,
    ) -> list[WebSearchJob]:
        if not outlets:
            return []

        call_limit = max(1, max_calls)
        planned_queries = _planned_queries(query, plan)
        jobs: list[WebSearchJob] = [
            WebSearchJob(
                search_query=_scoped_query(
                    planned_queries[0],
                    outlets,
                    use_site_query_filters=use_site_query_filters,
                ),
                outlets=tuple(outlets),
                allowed_domains=_allowed_domains(
                    outlets,
                    use_allowed_domains=use_allowed_domains,
                ),
            )
        ]
        if len(jobs) >= call_limit:
            return jobs

        seen_jobs: set[tuple[str, tuple[str, ...]]] = {
            (jobs[0].search_query, jobs[0].allowed_domains)
        }
        outlet_groups = _outlet_groups(outlets, call_limit - len(jobs))
        for planned_query in planned_queries:
            for group in outlet_groups:
                scoped_query = _scoped_query(
                    planned_query,
                    group,
                    use_site_query_filters=use_site_query_filters,
                )
                allowed_domains = _allowed_domains(
                    group,
                    use_allowed_domains=use_allowed_domains,
                )
                job_key = (scoped_query, allowed_domains)
                if job_key in seen_jobs:
                    continue
                jobs.append(
                    WebSearchJob(
                        search_query=scoped_query,
                        outlets=tuple(group),
                        allowed_domains=allowed_domains,
                    )
                )
                seen_jobs.add(job_key)
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


def _scoped_query(
    query: str,
    outlets: list[OutletConfig],
    *,
    use_site_query_filters: bool,
) -> str:
    """Add explicit outlet domain filters to one keyword query."""
    if not use_site_query_filters:
        return query.strip()
    return f"{query} {_domain_filter(outlets)}".strip()


def _domain_filter(outlets: list[OutletConfig]) -> str:
    return " OR ".join(f"site:{outlet.domain}" for outlet in outlets)


def _allowed_domains(
    outlets: list[OutletConfig],
    *,
    use_allowed_domains: bool,
) -> tuple[str, ...]:
    if not use_allowed_domains:
        return ()

    domains: list[str] = []
    seen: set[str] = set()
    for outlet in outlets:
        domain = normalize_allowed_domain(outlet.domain)
        if not domain or domain in seen:
            continue
        domains.append(domain)
        seen.add(domain)
    return tuple(domains)


