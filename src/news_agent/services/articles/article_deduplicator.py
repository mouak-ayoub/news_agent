from __future__ import annotations

from news_agent.models.triage import ArticleRecord


class ArticleDeduplicator:
    """Remove duplicate articles across overlapping search jobs."""

    def dedupe(self, articles: list[ArticleRecord]) -> list[ArticleRecord]:
        deduped: list[ArticleRecord] = []
        seen: set[str] = set()
        for article in articles:
            key = article.url.strip().lower() or (
                f"{article.outlet_name}:{article.title}".lower()
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(article)
        return deduped



