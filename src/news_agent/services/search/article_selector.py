from __future__ import annotations

from datetime import datetime
import json
import logging
import re

from ...models.config import AppConfig
from ...models.config import OutletConfig
from ...models.research import ResearchIntent
from ...models.triage import ArticleRecord
from ..prompt_service import PromptService
from ..text_generation import TextGenerator
from ..text_generation import build_text_generator
from ..text_generation import ModelGenerationError
from ..text_generation import ModelOutputError
from ..text_generation import extract_json_block
from .candidate_filter import CandidateFilter


logger = logging.getLogger(__name__)


class ArticleSelector:
    """Shared article selection step used after any search provider returns candidates."""

    def __init__(
        self,
        config: AppConfig,
        prompt_service: PromptService | None = None,
        text_generator: TextGenerator | None = None,
    ) -> None:
        self.config = config
        self.prompt_service = prompt_service or PromptService()
        self.text_generator = text_generator or build_text_generator(
            config.model,
            model_id=config.model.research_model_id or config.model.summary_model_id,
        )
        self.candidate_filter = CandidateFilter(
            config=config,
            prompt_service=self.prompt_service,
            text_generator=self.text_generator,
        )

    def choose_best_article(
        self,
        query: str,
        outlet: OutletConfig,
        candidates: list[ArticleRecord],
        intent: ResearchIntent | None = None,
    ) -> ArticleRecord | None:
        """Pick the best outlet article by prompt, with heuristic fallback when needed."""
        candidates = self.candidate_filter.filter(query, outlet, candidates, intent)
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        logger.info(
            "article selection started outlet=%r candidates=%d",
            outlet.name,
            len(candidates),
        )
        try:
            result = self.text_generator.generate(
                self._build_prompt(query, outlet, candidates)
            )
            payload = json.loads(extract_json_block(result.text))
            selected_index = _coerce_candidate_index(
                payload.get("selected_index"),
                len(candidates),
            )
            if selected_index is None:
                raise ModelOutputError(
                    "Article selector did not return a valid selected_index."
                )
            if selected_index == -1:
                logger.info(
                    "article selection selected mode=prompt outlet=%r index=-1 (no relevant candidate)",
                    outlet.name,
                )
                return None
            article = candidates[selected_index]
            logger.info(
                "article selection selected mode=prompt outlet=%r index=%d url=%s",
                outlet.name,
                selected_index,
                article.url,
            )
            return article
        except (
            ModelOutputError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ):
            if not self.config.fallback_to_heuristic:
                logger.exception("article selection failed because model output was unusable")
                raise
            article = _fallback_select_best_article(candidates, query)
            logger.warning("article selection output unusable; using heuristic selection")
            if article is not None:
                logger.info(
                    "article selection selected mode=fallback outlet=%r url=%s",
                    outlet.name,
                    article.url,
                )
            return article
        except ModelGenerationError:
            logger.exception("article selection failed because model generation failed")
            raise

    def choose_one_per_outlet(
        self,
        query: str,
        outlets: list[OutletConfig],
        candidates: list[ArticleRecord],
        intent: ResearchIntent | None = None,
    ) -> list[ArticleRecord]:
        """Keep one selected article per outlet, preserving configured outlet order."""
        candidates_by_outlet: dict[str, list[ArticleRecord]] = {}
        for article in candidates:
            candidates_by_outlet.setdefault(article.outlet_name, []).append(article)

        selected: list[ArticleRecord] = []
        for outlet in outlets:
            article = self.choose_best_article(
                query=query,
                outlet=outlet,
                candidates=candidates_by_outlet.get(outlet.name, []),
                intent=intent,
            )
            if article:
                selected.append(article)
        return selected

    def _build_prompt(
        self,
        query: str,
        outlet: OutletConfig,
        candidates: list[ArticleRecord],
    ) -> str:
        """Render the article-selection prompt for one outlet candidate pool."""
        candidate_lines = [
            {
                "index": index,
                "title": article.title,
                "url": article.url,
                "published_at": article.published_at,
                "snippet": article.snippet,
                "full_text_available": len(article.article_text) > len(article.snippet),
                "article_text": article.article_text[:1800],
            }
            for index, article in enumerate(candidates)
        ]
        return self.prompt_service.build(
            "article_curation",
            outlet_name=outlet.name,
            outlet_domain=outlet.domain,
            query=query,
            candidate_lines_json=json.dumps(candidate_lines, ensure_ascii=False, indent=2),
        )


def _coerce_candidate_index(value: object, candidate_count: int) -> int | None:
    try:
        index = int(value)
    except (TypeError, ValueError):
        return None
    if index == -1:
        return -1
    if 0 <= index < candidate_count:
        return index
    return None


def _fallback_select_best_article(
    candidates: list[ArticleRecord],
    query: str,
) -> ArticleRecord | None:
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda article: (_candidate_score(article, query), _published_timestamp(article)),
    )


def _candidate_score(article: ArticleRecord, query: str) -> float:
    text = f"{article.title} {article.snippet} {article.article_text}".lower()
    query_matches = sum(
        1 for token in dict.fromkeys(_important_query_tokens(query)) if token in text
    )
    detail_score = min(len(" ".join([article.title, article.snippet]).split()) / 40, 2)
    recency_score = _recency_score(article)
    return (query_matches * 3) + detail_score + recency_score


def _important_query_tokens(query: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z'-]{3,}", query.lower())


def _recency_score(article: ArticleRecord) -> float:
    if not article.published_at:
        return 0.0
    try:
        published_at = datetime.fromisoformat(article.published_at)
    except ValueError:
        return 0.0
    age_days = max((datetime.now().astimezone() - published_at.astimezone()).days, 0)
    if age_days <= 1:
        return 2.0
    if age_days <= 7:
        return 1.0
    if age_days <= 30:
        return 0.5
    return 0.0


def _published_timestamp(article: ArticleRecord) -> float:
    if not article.published_at:
        return 0.0
    try:
        return datetime.fromisoformat(article.published_at).timestamp()
    except ValueError:
        return 0.0
