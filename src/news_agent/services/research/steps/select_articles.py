from __future__ import annotations

import logging

from news_agent.models.config import OutletConfig
from news_agent.models.triage import ArticleRecord
from news_agent.services.articles.article_selector import ArticleSelector
from news_agent.services.research.context import ResearchContext


logger = logging.getLogger(__name__)


class SelectArticlesStep:
    def __init__(
        self,
        article_selector: ArticleSelector | None,
        outlets: list[OutletConfig],
        max_articles: int | None,
    ) -> None:
        self.article_selector = article_selector
        self.outlets = list(outlets)
        self.max_articles = max_articles

    def run(self, context: ResearchContext) -> ResearchContext:
        if not context.candidates:
            context.selected_articles = []
            return context

        if self.article_selector is None:
            context.selected_articles = self._limit(context.candidates)
            return context

        candidate_outlet_names = {article.outlet_name for article in context.candidates}
        target_outlets = [
            outlet for outlet in self.outlets if outlet.name in candidate_outlet_names
        ]
        if not target_outlets:
            logger.info(
                "research skipped outlet selection; candidates do not match configured outlets"
            )
            context.selected_articles = self._limit(context.candidates)
            return context

        context.selected_articles = self.article_selector.choose_one_per_outlet(
            query=context.query,
            outlets=target_outlets,
            candidates=context.candidates,
            intent=context.intent,
        )
        return context

    def _limit(self, articles: list[ArticleRecord]) -> list[ArticleRecord]:
        if self.max_articles is None or self.max_articles <= 0:
            return articles
        return articles[: self.max_articles]
