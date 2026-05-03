from __future__ import annotations

from news_agent.models.triage import ResearchBundle
from news_agent.services.research.context import ResearchContext


class BuildResearchBundleStep:
    def run(self, context: ResearchContext) -> ResearchContext:
        context.bundle = ResearchBundle(
            query=context.query,
            articles=context.selected_articles,
            intent=context.intent,
            search_plan=context.search_plan,
        )
        return context
