from __future__ import annotations

import logging

from news_agent.services.research.context import ResearchContext


logger = logging.getLogger(__name__)


class ApplyAnswerPolicyStep:
    def run(self, context: ResearchContext) -> ResearchContext:
        if context.bundle is None or context.intent is None:
            return context

        if not _requires_direct_metric(context.intent.expected_answer_type):
            return context

        matching_articles = [
            article for article in context.bundle.articles if article.metric_found
        ]
        if matching_articles:
            removed_count = len(context.bundle.articles) - len(matching_articles)
            if removed_count:
                logger.info(
                    "research removed non-metric articles count=%d",
                    removed_count,
                )
            context.bundle.articles = matching_articles
        else:
            logger.info("research found no articles containing requested metric")
            context.bundle.articles = []

        return context


def _requires_direct_metric(expected_answer_type: str) -> bool:
    """Decide when selected articles must contain the exact requested metric."""
    return expected_answer_type.strip().lower() in {
        "count",
        "number or range",
        "date",
    }
