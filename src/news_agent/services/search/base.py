from __future__ import annotations

from typing import Protocol

from ...models.research import ResearchIntent
from ...models.research import SearchPlan
from ...models.triage import ResearchBundle


class SearchClient(Protocol):
    """Provider strategy: fetch candidate articles for a planned research query."""

    def search(
        self,
        query: str,
        plan: SearchPlan | None = None,
        intent: ResearchIntent | None = None,
    ) -> ResearchBundle:
        """Return a research bundle for the user query."""
        ...
