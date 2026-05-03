from __future__ import annotations

from news_agent.services.research.context import ResearchContext
from news_agent.services.research.metric_extractor import MetricExtractor


class ExtractMetricsStep:
    def __init__(self, metric_extractor: MetricExtractor | None) -> None:
        self.metric_extractor = metric_extractor

    def run(self, context: ResearchContext) -> ResearchContext:
        if self.metric_extractor and context.bundle is not None:
            context.bundle = self.metric_extractor.enrich_bundle(context.bundle)
        return context
