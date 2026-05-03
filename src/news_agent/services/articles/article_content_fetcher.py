from __future__ import annotations

import html
import logging
import re

import requests

from news_agent.models.config import AppConfig
from news_agent.models.triage import ArticleRecord
from news_agent.models.triage import ResearchBundle


logger = logging.getLogger(__name__)


class ArticleContentFetcher:
    """Best-effort article body fetch for candidate URLs."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": config.search.user_agent})

    def enrich_bundle(self, bundle: ResearchBundle) -> ResearchBundle:
        """Replace weak snippets with fetched page text when a readable body is available."""
        self.enrich_articles(bundle.articles)
        return bundle

    def enrich_articles(self, articles: list[ArticleRecord]) -> list[ArticleRecord]:
        """Fetch readable page text for candidate articles before LLM selection."""
        logger.info("article content fetch started candidates=%d", len(articles))
        for article in articles:
            if len(article.article_text) >= 1000:
                continue
            if not article.url.startswith(("http://", "https://")):
                continue
            try:
                fetched_text = self._fetch_text(article.url)
            except requests.RequestException:
                logger.info("article content fetch failed url=%s", article.url)
                continue
            if len(fetched_text) > len(article.article_text):
                article.article_text = fetched_text[:8000]
        logger.info("article content fetch finished")
        return articles

    def _fetch_text(self, url: str) -> str:
        """Download one article URL and return cleaned visible text."""
        response = self.session.get(
            url,
            timeout=self.config.search.request_timeout_seconds,
            allow_redirects=True,
        )
        response.raise_for_status()
        if _is_unreadable_fetch(response.url, response.text):
            logger.info("article content fetch ignored unreadable page url=%s", response.url)
            return ""
        return _clean_html(_extract_article_region(response.text))


def _clean_html(raw_html: str) -> str:
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", raw_html)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = _trim_after_boilerplate(text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_article_region(raw_html: str) -> str:
    """Prefer article/main HTML over the full page to avoid nav/trending noise."""
    class_match = re.search(
        r'(?is)<[^>]+class=["\'][^"\']*(?:article-header|article-body|article-content|story-body|entry-content)[^"\']*["\'][^>]*>.*',
        raw_html,
    )
    if class_match:
        return class_match.group(0)

    main_match = re.search(r"(?is)<main\b.*?</main>", raw_html)
    if main_match:
        return main_match.group(0)

    article_blocks = re.findall(r"(?is)<article\b.*?</article>", raw_html)
    if article_blocks:
        return max(article_blocks, key=len)

    return raw_html


def _trim_after_boilerplate(text: str) -> str:
    """Cut common recommendation/share sections after the story text."""
    compact = re.sub(r"\s+", " ", text).strip()
    lowered = compact.lower()
    markers = (
        " more from the same show ",
        " more from this show ",
        " related stories ",
        " related topics ",
        " recommended stories ",
        " most read ",
        " trending ",
        " sign up ",
    )
    for marker in markers:
        index = lowered.find(marker, 220)
        if index != -1:
            return compact[:index]
    return compact


def _is_unreadable_fetch(final_url: str, raw_html: str) -> bool:
    """Reject consent/login wrapper pages that are longer than the real article signal."""
    lowered_url = final_url.lower()
    lowered_html = raw_html[:3000].lower()
    if "consent.google.com" in lowered_url:
        return True
    return any(
        marker in lowered_html
        for marker in (
            "before you continue to google",
            "enable javascript",
            "please enable js",
            "privacy policy</title>",
        )
    )


