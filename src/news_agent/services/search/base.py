from __future__ import annotations

from typing import Protocol

from ...models.research import ResearchIntent
from ...models.research import SearchPlan
from ...models.triage import ArticleRecord


class SearchClient(Protocol):
    """Provider strategy: fetch candidate articles for a planned research query."""

    def search_candidates(
        self,
        query: str,
        plan: SearchPlan | None = None,
        intent: ResearchIntent | None = None,
    ) -> list[ArticleRecord]:
        """Return provider-normalized article candidates for the user query."""
        ...
