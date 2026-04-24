from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from email.utils import parsedate_to_datetime
import html
import re
from urllib.parse import urlencode
import xml.etree.ElementTree as ET

import requests

from .config import OutletConfig
from .config import SearchConfig
from .schemas import ArticleRecord


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
        return None


@dataclass(slots=True)
class GoogleNewsSearchService:
    config: SearchConfig
    session: requests.Session = field(init=False)

    def __post_init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.config.user_agent})

    def search_outlet(self, query: str, outlet: OutletConfig) -> list[ArticleRecord]:
        scoped_query = f"site:{outlet.domain} {query} when:{self.config.days_back}d"
        params = {
            "q": scoped_query,
            "hl": "en-US",
            "gl": "US",
            "ceid": "US:en",
        }
        url = "https://news.google.com/rss/search?" + urlencode(params)
        response = self.session.get(url, timeout=self.config.request_timeout_seconds)
        response.raise_for_status()
        root = ET.fromstring(response.text)
        cutoff = datetime.now().astimezone() - timedelta(days=self.config.days_back)

        results: list[ArticleRecord] = []
        for item in root.findall("./channel/item"):
            published_at = _parse_pub_date(item.findtext("pubDate"))
            if published_at and published_at.astimezone() < cutoff:
                continue

            title = (item.findtext("title") or "").strip()
            google_link = (item.findtext("link") or "").strip()
            snippet = _clean_html(item.findtext("description") or "")
            final_url, article_text = self._resolve_article(google_link, snippet)
            results.append(
                ArticleRecord(
                    title=title,
                    url=final_url,
                    outlet_name=outlet.name,
                    domain=outlet.domain,
                    country=outlet.country,
                    medium_type=outlet.medium_type,
                    orientation=outlet.orientation,
                    published_at=published_at.isoformat() if published_at else None,
                    snippet=snippet[:500],
                    article_text=article_text[:4000],
                    search_query=scoped_query,
                )
            )
            if len(results) >= self.config.max_per_outlet:
                break
        return results

    def _resolve_article(self, link: str, fallback_text: str) -> tuple[str, str]:
        try:
            response = self.session.get(
                link,
                allow_redirects=True,
                timeout=self.config.request_timeout_seconds,
            )
            response.raise_for_status()
            cleaned = _clean_html(response.text)
            return str(response.url), cleaned or fallback_text
        except requests.RequestException:
            return link, fallback_text
