from __future__ import annotations

import logging
import traceback
from typing import Any

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.genai import types
from typing_extensions import override

from news_agent.services.research import ResearchService


logger = logging.getLogger(__name__)


def _extract_query(user_content: types.Content | None) -> str:
    if not user_content or not user_content.parts:
        return ""
    return " ".join(part.text or "" for part in user_content.parts).strip()


class ResearchAgent(BaseAgent):
    service: Any

    @override
    async def _run_async_impl(self, ctx: InvocationContext):
        query = _extract_query(ctx.user_content)
        logger.info("ResearchAgent launched query=%r", query)
        try:
            bundle = self.service.research(query)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            logger.exception("ResearchAgent failed")
            yield Event(
                invocation_id=ctx.invocation_id,
                author=self.name,
                branch=ctx.branch,
                actions=EventActions(
                    state_delta={
                        "query": query,
                        "workflow_error": message,
                        "workflow_error_stage": "research",
                        "workflow_error_traceback": traceback.format_exc(),
                    }
                ),
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=f"ResearchAgent failed: {message}")],
                ),
            )
            return
        logger.info("ResearchAgent completed articles=%d", len(bundle.articles))
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            actions=EventActions(
                state_delta={
                    "query": query,
                    "research_bundle": bundle.to_dict(),
                }
            ),
            content=types.Content(
                role="model",
                parts=[types.Part(text=f"ResearchAgent gathered {len(bundle.articles)} article(s).")],
            ),
        )
