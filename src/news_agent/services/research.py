from __future__ import annotations

import logging

from .search.base import SearchClient
from .article_content_fetcher import ArticleContentFetcher
from .metric_extractor import MetricExtractor
from .query_planner import QueryPlanner
from .question_analyzer import QuestionAnalyzer
from ..models.triage import ResearchBundle


logger = logging.getLogger(__name__)


class ResearchService:
    """Coordinates the research pipeline around a pluggable search provider."""

    def __init__(
        self,
        client: SearchClient,
        question_analyzer: QuestionAnalyzer | None = None,
        query_planner: QueryPlanner | None = None,
        article_content_fetcher: ArticleContentFetcher | None = None,
        metric_extractor: MetricExtractor | None = None,
    ) -> None:
        self.client = client
        self.question_analyzer = question_analyzer
        self.query_planner = query_planner
        self.article_content_fetcher = article_content_fetcher
        self.metric_extractor = metric_extractor

    def research(self, query: str) -> ResearchBundle:
        """Run intent analysis, planned search, content enrichment, and metric extraction."""
        logger.info("research started query=%r", query)
        intent = self.question_analyzer.analyze(query) if self.question_analyzer else None
        plan = (
            self.query_planner.plan(query, intent)
            if self.query_planner and intent is not None
            else None
        )
        bundle = self.client.search(query, plan=plan, intent=intent)
        if self.article_content_fetcher:
            bundle = self.article_content_fetcher.enrich_bundle(bundle)
        if self.metric_extractor:
            bundle = self.metric_extractor.enrich_bundle(bundle)
            if intent and _requires_direct_metric(intent.expected_answer_type):
                matching_articles = [
                    article for article in bundle.articles if article.metric_found
                ]
                if matching_articles:
                    removed_count = len(bundle.articles) - len(matching_articles)
                    if removed_count:
                        logger.info(
                            "research removed non-metric articles count=%d",
                            removed_count,
                        )
                    bundle.articles = matching_articles
                else:
                    logger.info(
                        "research found no articles containing requested metric"
                    )
                    bundle.articles = []
        logger.info("research finished articles=%d", len(bundle.articles))
        for article in bundle.articles:
            logger.info(
                "research article outlet=%r title=%r url=%s",
                article.outlet_name,
                article.title,
                article.url,
            )
        return bundle


def _requires_direct_metric(expected_answer_type: str) -> bool:
    """Decide when selected articles must contain the exact requested metric."""
    return expected_answer_type.strip().lower() in {
        "count",
        "number or range",
        "date",
    }
