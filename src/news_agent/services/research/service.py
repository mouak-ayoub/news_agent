from __future__ import annotations

import logging

from news_agent.models.config import OutletConfig
from news_agent.models.research import ResearchIntent
from news_agent.models.triage import ArticleRecord
from news_agent.models.triage import ResearchBundle
from news_agent.services.search.base import SearchClient
from news_agent.services.articles.article_selector import ArticleSelector
from news_agent.services.articles.article_content_fetcher import ArticleContentFetcher
from .metric_extractor import MetricExtractor
from .query_planner import QueryPlanner
from .question_analyzer import QuestionAnalyzer


logger = logging.getLogger(__name__)


class ResearchService:
    """Coordinates the research pipeline around a pluggable search provider."""

    def __init__(
        self,
        client: SearchClient,
        question_analyzer: QuestionAnalyzer | None = None,
        query_planner: QueryPlanner | None = None,
        article_content_fetcher: ArticleContentFetcher | None = None,
        article_selector: ArticleSelector | None = None,
        metric_extractor: MetricExtractor | None = None,
        outlets: list[OutletConfig] | None = None,
        max_articles: int | None = None,
    ) -> None:
        self.client = client
        self.question_analyzer = question_analyzer
        self.query_planner = query_planner
        self.article_content_fetcher = article_content_fetcher
        self.article_selector = article_selector
        self.metric_extractor = metric_extractor
        self.outlets = list(outlets or [])
        self.max_articles = max_articles

    def research(self, query: str) -> ResearchBundle:
        """Run intent analysis, planned search, content enrichment, and metric extraction."""
        logger.info("research started query=%r", query)
        intent = self.question_analyzer.analyze(query) if self.question_analyzer else None
        plan = (
            self.query_planner.plan(query, intent)
            if self.query_planner and intent is not None
            else None
        )
        candidates = self.client.search_candidates(query, plan=plan, intent=intent)
        logger.info("research retrieved candidates=%d", len(candidates))
        if self.article_content_fetcher and candidates:
            candidate_bundle = ResearchBundle(
                query=query,
                articles=candidates,
                intent=intent,
                search_plan=plan,
            )
            candidates = self.article_content_fetcher.enrich_bundle(candidate_bundle).articles
        bundle = ResearchBundle(
            query=query,
            articles=self._select_articles(
                query=query,
                candidates=candidates,
                intent=intent,
            ),
            intent=intent,
            search_plan=plan,
        )
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

    def _select_articles(
        self,
        *,
        query: str,
        candidates: list[ArticleRecord],
        intent: ResearchIntent | None,
    ) -> list[ArticleRecord]:
        """Select final articles from provider candidates."""
        if not candidates:
            return []
        if self.article_selector is None:
            return self._limit_articles(candidates)

        candidate_outlet_names = {article.outlet_name for article in candidates}
        target_outlets = [
            outlet for outlet in self.outlets if outlet.name in candidate_outlet_names
        ]
        if not target_outlets:
            logger.info(
                "research skipped outlet selection; candidates do not match configured outlets"
            )
            return self._limit_articles(candidates)

        return self.article_selector.choose_one_per_outlet(
            query=query,
            outlets=target_outlets,
            candidates=candidates,
            intent=intent,
        )

    def _limit_articles(self, articles: list[ArticleRecord]) -> list[ArticleRecord]:
        if self.max_articles is None or self.max_articles <= 0:
            return articles
        return articles[: self.max_articles]


def _requires_direct_metric(expected_answer_type: str) -> bool:
    """Decide when selected articles must contain the exact requested metric."""
    return expected_answer_type.strip().lower() in {
        "count",
        "number or range",
        "date",
    }


