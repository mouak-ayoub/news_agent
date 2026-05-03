from __future__ import annotations

from news_agent.models.triage import ResearchBundle
from news_agent.services.articles.article_content_fetcher import ArticleContentFetcher
from news_agent.services.research.context import ResearchContext


class EnrichCandidatesStep:
    def __init__(self, article_content_fetcher: ArticleContentFetcher | None) -> None:
        self.article_content_fetcher = article_content_fetcher

    def run(self, context: ResearchContext) -> ResearchContext:
        if self.article_content_fetcher and context.candidates:
            bundle = ResearchBundle(
                query=context.query,
                articles=context.candidates,
                intent=context.intent,
                search_plan=context.search_plan,
            )
            context.candidates = self.article_content_fetcher.enrich_bundle(bundle).articles
        return context
