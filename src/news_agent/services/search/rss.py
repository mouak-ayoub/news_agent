from __future__ import annotations

from datetime import datetime
from datetime import timedelta
from email.utils import parsedate_to_datetime
import html
import logging
import math
import re
from urllib.parse import urlencode
import xml.etree.ElementTree as ET

import requests

from ...models.config import AppConfig
from ...models.config import OutletConfig
from ...models.research import ResearchIntent
from ...models.research import SearchPlan
from ...models.triage import ArticleRecord
from ...models.triage import ResearchBundle
from ..article_content_fetcher import ArticleContentFetcher
from ..prompt_service import PromptService
from .article_selector import ArticleSelector


logger = logging.getLogger(__name__)

_PREFILTER_STOPWORDS = {
    "about",
    "after",
    "also",
    "conflict",
    "current",
    "figures",
    "from",
    "have",
    "into",
    "latest",
    "news",
    "over",
    "reported",
    "says",
    "that",
    "their",
    "this",
    "toll",
    "what",
    "when",
    "with",
}


def _clean_html(raw_html: str) -> str:
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", raw_html)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_pub_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None


class GoogleNewsRssSearchClient:
    """RSS provider: prefer direct outlet feeds, optionally fall back to Google News."""

    def __init__(
        self,
        config: AppConfig,
        prompt_service: PromptService | None = None,
    ) -> None:
        self.config = config
        self.search_config = config.search
        self.outlets = config.outlets
        self.prompt_service = prompt_service or PromptService()
        self.article_selector = ArticleSelector(
            config=config,
            prompt_service=self.prompt_service,
        )
        self.article_content_fetcher = ArticleContentFetcher(config)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.search_config.user_agent})

    def search(
        self,
        query: str,
        plan: SearchPlan | None = None,
        intent: ResearchIntent | None = None,
    ) -> ResearchBundle:
        """Run outlet-scoped RSS searches, with one broad RSS fallback pass."""
        outlet_limit = min(self.search_config.max_sources, len(self.outlets))
        target_outlets = self.outlets[:outlet_limit]
        articles_by_outlet: dict[str, ArticleRecord] = {}
        logger.info(
            "rss search started query=%r target_outlets=%d",
            query,
            len(target_outlets),
        )

        for outlet in target_outlets:
            try:
                outlet_articles = self.search_outlet(query, outlet, plan, intent)
            except requests.RequestException:
                logger.exception("rss outlet search failed outlet=%r", outlet.name)
                continue
            if outlet_articles:
                articles_by_outlet[outlet.name] = outlet_articles[0]
                logger.info(
                    "rss selected outlet=%r url=%s",
                    outlet.name,
                    outlet_articles[0].url,
                )
            else:
                logger.info("rss no selected article outlet=%r", outlet.name)

        missing_outlets = {
            outlet.name for outlet in target_outlets if outlet.name not in articles_by_outlet
        }
        if missing_outlets and self.search_config.allow_google_news_fallback:
            logger.info("rss curated fallback started missing_outlets=%s", sorted(missing_outlets))
            try:
                for article in self.search_curated(query, plan, intent):
                    if article.outlet_name in missing_outlets:
                        articles_by_outlet[article.outlet_name] = article
                        missing_outlets.remove(article.outlet_name)
                        logger.info(
                            "rss fallback selected outlet=%r url=%s",
                            article.outlet_name,
                            article.url,
                        )
            except requests.RequestException:
                logger.exception("rss curated fallback failed")
                pass
        elif missing_outlets:
            logger.info(
                "rss curated fallback disabled missing_outlets=%s",
                sorted(missing_outlets),
            )

        bundle = ResearchBundle(
            query=query,
            articles=[
                articles_by_outlet[outlet.name]
                for outlet in target_outlets
                if outlet.name in articles_by_outlet
            ],
            intent=intent,
            search_plan=plan,
        )
        logger.info(
            "rss search finished selected_articles=%d",
            len(bundle.articles),
        )
        return bundle

    def search_outlet(
        self,
        query: str,
        outlet: OutletConfig,
        plan: SearchPlan | None = None,
        intent: ResearchIntent | None = None,
    ) -> list[ArticleRecord]:
        """Search one outlet and return the first full-text candidate that answers."""
        if _feed_urls(outlet):
            candidates = self._search_outlet_feed(
                outlet=outlet,
                relevance_query=query,
            )
            logger.info(
                "rss outlet feed candidates outlet=%r count=%d",
                outlet.name,
                len(candidates),
            )
            candidates = _prefilter_candidates(candidates, query, intent)
            logger.info(
                "rss outlet feed prefiltered outlet=%r count=%d",
                outlet.name,
                len(candidates),
            )
            best_article = self._choose_after_full_text(
                query=query,
                outlet=outlet,
                candidates=candidates,
                intent=intent,
            )
            if best_article:
                return [best_article]
            if not self.search_config.allow_google_news_fallback:
                logger.info(
                    "rss outlet no matching direct-feed article; Google News fallback disabled outlet=%r",
                    outlet.name,
                )
                return []

        for planned_query in _planned_queries(query, plan):
            scoped_query = (
                f"site:{outlet.domain} {planned_query} when:{self.search_config.days_back}d"
            )
            logger.info("rss outlet query outlet=%r query=%r", outlet.name, scoped_query)
            candidates = self._search_query(
                scoped_query=scoped_query,
                outlet=outlet,
                relevance_query=query,
            )
            logger.info(
                "rss outlet candidates outlet=%r count=%d",
                outlet.name,
                len(candidates),
            )
            candidates = _prefilter_candidates(candidates, query, intent)
            logger.info(
                "rss outlet prefiltered outlet=%r count=%d",
                outlet.name,
                len(candidates),
            )
            best_article = self._choose_after_full_text(
                query=query,
                outlet=outlet,
                candidates=candidates,
                intent=intent,
            )
            if best_article:
                return [best_article]
        return []

    def _choose_after_full_text(
        self,
        query: str,
        outlet: OutletConfig,
        candidates: list[ArticleRecord],
        intent: ResearchIntent | None,
    ) -> ArticleRecord | None:
        """Read candidate pages before asking the model to select direct evidence."""
        if not candidates:
            return None
        self.article_content_fetcher.enrich_articles(candidates)
        return self.article_selector.choose_best_article(
            query,
            outlet,
            candidates,
            intent=intent,
        )

    def _search_outlet_feed(
        self,
        outlet: OutletConfig,
        relevance_query: str,
    ) -> list[ArticleRecord]:
        """Read configured publisher RSS feeds, yielding direct article URLs."""
        cutoff = datetime.now().astimezone() - timedelta(
            days=self.search_config.days_back
        )

        feed_urls = _feed_urls(outlet)
        per_feed_limit = max(1, math.ceil(self.search_config.candidate_pool_size / len(feed_urls)))
        results: list[ArticleRecord] = []
        seen_urls: set[str] = set()
        for feed_url in feed_urls:
            response = self.session.get(
                feed_url,
                timeout=self.search_config.request_timeout_seconds,
            )
            response.raise_for_status()
            root = ET.fromstring(response.text)
            feed_count = 0
            for item in _iter_feed_items(root):
                article = self._article_from_item(
                    item=item,
                    outlet=outlet,
                    search_query=feed_url,
                    relevance_query=relevance_query,
                    cutoff=cutoff,
                )
                if article is None or article.url in seen_urls:
                    continue
                seen_urls.add(article.url)
                results.append(article)
                feed_count += 1
                if feed_count >= per_feed_limit:
                    break
            logger.info(
                "rss outlet feed read outlet=%r feed_url=%s accepted=%d",
                outlet.name,
                feed_url,
                feed_count,
            )
        return results

    def search_curated(
        self,
        query: str,
        plan: SearchPlan | None = None,
        intent: ResearchIntent | None = None,
    ) -> list[ArticleRecord]:
        """Fallback: broad RSS search, then select one result per configured outlet."""
        logger.info("rss curated query query=%r", query)
        cutoff = datetime.now().astimezone() - timedelta(
            days=self.search_config.days_back
        )

        candidates_by_outlet: dict[str, list[ArticleRecord]] = {}
        for planned_query in _planned_queries(query, plan):
            params = {
                "q": planned_query,
                "hl": "en-US",
                "gl": "US",
                "ceid": "US:en",
            }
            url = "https://news.google.com/rss/search?" + urlencode(params)
            response = self.session.get(
                url,
                timeout=self.search_config.request_timeout_seconds,
            )
            response.raise_for_status()
            root = ET.fromstring(response.text)
            for item in root.findall("./channel/item"):
                outlet = self._match_outlet(item)
                if outlet is None:
                    continue
                article = self._article_from_item(
                    item=item,
                    outlet=outlet,
                    search_query=planned_query,
                    relevance_query=query,
                    cutoff=cutoff,
                )
                if article is None:
                    continue
                candidates_by_outlet.setdefault(outlet.name, []).append(article)

        candidates = [
            article
            for outlet in self.outlets
            for article in candidates_by_outlet.get(outlet.name, [])
        ]
        candidates = _prefilter_candidates(candidates, query, intent)
        self.article_content_fetcher.enrich_articles(candidates)
        return self.article_selector.choose_one_per_outlet(
            query=query,
            outlets=self.outlets,
            candidates=candidates,
            intent=intent,
        )

    def _search_query(
        self,
        scoped_query: str,
        outlet: OutletConfig,
        relevance_query: str,
    ) -> list[ArticleRecord]:
        """Fetch a small candidate pool for one outlet-scoped Google News query."""
        params = {
            "q": scoped_query,
            "hl": "en-US",
            "gl": "US",
            "ceid": "US:en",
        }
        url = "https://news.google.com/rss/search?" + urlencode(params)
        response = self.session.get(
            url,
            timeout=self.search_config.request_timeout_seconds,
        )
        response.raise_for_status()
        root = ET.fromstring(response.text)
        cutoff = datetime.now().astimezone() - timedelta(
            days=self.search_config.days_back
        )

        results: list[ArticleRecord] = []
        for item in root.findall("./channel/item"):
            article = self._article_from_item(
                item=item,
                outlet=outlet,
                search_query=scoped_query,
                relevance_query=relevance_query,
                cutoff=cutoff,
            )
            if article is None:
                continue
            results.append(article)
            if len(results) >= self.search_config.candidate_pool_size:
                break
        return results

    def _article_from_item(
        self,
        item: ET.Element,
        outlet: OutletConfig,
        search_query: str,
        relevance_query: str,
        cutoff: datetime,
    ) -> ArticleRecord | None:
        """Convert one RSS item into an article record if it is recent and non-empty."""
        _ = relevance_query
        published_at = _parse_pub_date(
            _find_child_text(item, "pubDate")
            or _find_child_text(item, "published")
            or _find_child_text(item, "updated")
            or _find_child_text(item, "date")
        )
        if published_at and published_at.astimezone() < cutoff:
            return None

        title = (_find_child_text(item, "title") or "").strip()
        snippet = _clean_html(
            _find_child_text(item, "description")
            or _find_child_text(item, "summary")
            or _find_child_text(item, "encoded")
            or ""
        )
        if not (title or snippet).strip():
            return None

        link = _find_child_text(item, "link") or _find_child_attr(item, "link", "href")
        article_url = (link or "").strip()
        return ArticleRecord(
            title=title,
            url=article_url,
            outlet_name=outlet.name,
            domain=outlet.domain,
            country=outlet.country,
            medium_type=outlet.medium_type,
            orientation=outlet.orientation,
            published_at=published_at.isoformat() if published_at else None,
            snippet=snippet[:500],
            article_text=snippet[:4000],
            search_query=search_query,
        )

    def _match_outlet(self, item: ET.Element) -> OutletConfig | None:
        """Map a Google News RSS item back to one configured outlet."""
        source = item.find("source")
        source_name = (source.text or "").strip().lower() if source is not None else ""
        source_url = (source.attrib.get("url", "") if source is not None else "").lower()
        title = (item.findtext("title") or "").lower()

        for outlet in self.outlets:
            domain = outlet.domain.lower().removeprefix("www.")
            name = outlet.name.lower()
            if domain and domain in source_url:
                return outlet
            if source_name and (source_name == name or name in source_name):
                return outlet
            if title.endswith(f" - {outlet.name}".lower()):
                return outlet
        return None


def _planned_queries(query: str, plan: SearchPlan | None) -> list[str]:
    """Use planned queries when available, otherwise fall back to the raw query."""
    if plan and plan.queries:
        return plan.queries
    return [query]


def _prefilter_candidates(
    candidates: list[ArticleRecord],
    query: str,
    intent: ResearchIntent | None,
) -> list[ArticleRecord]:
    """Drop obvious front-page/feed items before full-text and LLM filtering."""
    if not candidates:
        return []
    tokens = _intent_tokens(query, intent)
    if not tokens:
        return candidates
    return [
        article
        for article in candidates
        if _candidate_has_topic_overlap(article, tokens)
    ]


def _intent_tokens(query: str, intent: ResearchIntent | None) -> set[str]:
    text_parts = [query]
    if intent is not None:
        text_parts.extend(
            [
                intent.topic,
                intent.requested_metric,
                " ".join(intent.must_find),
            ]
        )
    tokens = {
        token
        for token in re.findall(r"[a-zA-Z]{3,}", " ".join(text_parts).lower())
        if token not in _PREFILTER_STOPWORDS
    }
    return {token for token in tokens if len(token) >= 4}


def _candidate_has_topic_overlap(article: ArticleRecord, tokens: set[str]) -> bool:
    haystack = f"{article.title} {article.snippet}".lower()
    return any(token in haystack for token in tokens)


def _feed_urls(outlet: OutletConfig) -> list[str]:
    """Return configured feed URLs while keeping old `rss_url` config valid."""
    urls = [url.strip() for url in outlet.rss_urls if url.strip()]
    if outlet.rss_url.strip():
        urls.insert(0, outlet.rss_url.strip())
    return list(dict.fromkeys(urls))


def _iter_feed_items(root: ET.Element) -> list[ET.Element]:
    """Return RSS/RDF/Atom entries without depending on namespace prefixes."""
    return [
        item
        for item in root.iter()
        if _local_name(item.tag) in {"item", "entry"}
    ]


def _find_child_text(item: ET.Element, local_name: str) -> str | None:
    for child in item:
        if _local_name(child.tag) == local_name:
            return child.text
    return None


def _find_child_attr(item: ET.Element, local_name: str, attr_name: str) -> str | None:
    for child in item:
        if _local_name(child.tag) == local_name:
            value = child.attrib.get(attr_name)
            if value:
                return value
    return None


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]
