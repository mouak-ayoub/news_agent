from __future__ import annotations

from typing import Protocol

from news_agent.services.research.context import ResearchContext


class ResearchStep(Protocol):
    def run(self, context: ResearchContext) -> ResearchContext:
        ...
