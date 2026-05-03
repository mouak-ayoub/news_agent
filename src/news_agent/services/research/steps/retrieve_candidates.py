from __future__ import annotations

import logging

from news_agent.services.research.context import ResearchContext
from news_agent.services.search.base import SearchClient


logger = logging.getLogger(__name__)


class RetrieveCandidatesStep:
    def __init__(self, client: SearchClient) -> None:
        self.client = client

    def run(self, context: ResearchContext) -> ResearchContext:
        context.candidates = self.client.search_candidates(
            context.query,
            plan=context.search_plan,
            intent=context.intent,
        )
        logger.info("research retrieved candidates=%d", len(context.candidates))
        return context
