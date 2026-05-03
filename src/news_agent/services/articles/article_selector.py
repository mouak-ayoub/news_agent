from __future__ import annotations

import json
import logging

from news_agent.models.config import AppConfig
from news_agent.models.config import OutletConfig
from news_agent.models.research import ResearchIntent
from news_agent.models.triage import ArticleRecord
from news_agent.services.debug.debug_output import DebugOutput
from news_agent.services.prompts.prompt_service import PromptService
from news_agent.services.llm.text_generation import TextGenerator
from news_agent.services.llm.text_generation import build_text_generator
from news_agent.services.llm.text_generation import ModelGenerationError
from news_agent.services.llm.text_generation import ModelOutputError
from news_agent.services.llm.text_generation import extract_json_block
from .candidate_filter import CandidateFilter


logger = logging.getLogger(__name__)


class ArticleSelector:
    """Shared article selection step used after any search provider returns candidates."""

    def __init__(
        self,
        config: AppConfig,
        prompt_service: PromptService | None = None,
        text_generator: TextGenerator | None = None,
        debug_output: DebugOutput | None = None,
    ) -> None:
        self.config = config
        self.prompt_service = prompt_service or PromptService()
        self.text_generator = text_generator or build_text_generator(
            config.model,
            model_id=config.model.model_id_for_step("article_selection"),
        )
        self.debug_output = debug_output
        self.candidate_filter = CandidateFilter(
            config=config,
            prompt_service=self.prompt_service,
            debug_output=self.debug_output,
        )

    def choose_best_article(
        self,
        query: str,
        outlet: OutletConfig,
        candidates: list[ArticleRecord],
        intent: ResearchIntent | None = None,
    ) -> ArticleRecord | None:
        """Pick the best outlet article by prompt."""
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
        prompt = self._build_prompt(query, outlet, candidates)
        debug_call = (
            self.debug_output.start_model_call(f"article_selection_{outlet.name}", prompt)
            if self.debug_output
            else None
        )
        try:
            result = self.text_generator.generate(prompt)
            if debug_call:
                debug_call.write_output(result.text)
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
        ) as exc:
            if debug_call:
                debug_call.write_error(exc)
            logger.exception("article selection failed because model output was unusable")
            raise
        except ModelGenerationError as exc:
            if debug_call:
                debug_call.write_error(exc)
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



