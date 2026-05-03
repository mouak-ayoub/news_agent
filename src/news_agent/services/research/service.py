from __future__ import annotations

import logging

from news_agent.models.triage import ResearchBundle
from news_agent.services.research.pipeline import ResearchPipeline


logger = logging.getLogger(__name__)


class ResearchService:
    """Public research use-case service backed by an explicit pipeline."""

    def __init__(self, pipeline: ResearchPipeline) -> None:
        self.pipeline = pipeline

    def research(self, query: str) -> ResearchBundle:
        logger.info("research started query=%r", query)
        bundle = self.pipeline.run(query)
        logger.info("research finished articles=%d", len(bundle.articles))
        for article in bundle.articles:
            logger.info(
                "research article outlet=%r title=%r url=%s",
                article.outlet_name,
                article.title,
                article.url,
            )
        return bundle
