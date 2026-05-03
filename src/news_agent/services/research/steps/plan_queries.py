from __future__ import annotations

from news_agent.services.research.context import ResearchContext
from news_agent.services.research.query_planner import QueryPlanner


class PlanQueriesStep:
    def __init__(self, query_planner: QueryPlanner | None) -> None:
        self.query_planner = query_planner

    def run(self, context: ResearchContext) -> ResearchContext:
        if self.query_planner and context.intent is not None:
            context.search_plan = self.query_planner.plan(
                context.query,
                context.intent,
            )
        return context
