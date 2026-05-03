from __future__ import annotations

from urllib.parse import urlparse


def normalize_allowed_domain(value: str) -> str:
    """Normalize a configured domain for OpenAI web_search allowed_domains."""
    text = str(value).strip().lower()
    if not text:
        return ""

    parsed = urlparse(text if "://" in text else f"//{text}")
    host = parsed.hostname or text.split("/", 1)[0]
    return host.strip().rstrip(".")
