from __future__ import annotations

from news_agent.models.triage import ResearchBundle
from news_agent.services.research.context import ResearchContext
from news_agent.services.research.steps.base import ResearchStep


class ResearchPipeline:
    def __init__(self, steps: list[ResearchStep]) -> None:
        self.steps = steps

    def run(self, query: str) -> ResearchBundle:
        context = ResearchContext(query=query)
        for step in self.steps:
            context = step.run(context)

        if context.bundle is None:
            raise RuntimeError("Research pipeline did not produce a ResearchBundle.")

        return context.bundle
