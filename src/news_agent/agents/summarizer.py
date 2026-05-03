from __future__ import annotations

import logging
import traceback

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.genai import types
from typing_extensions import override

from news_agent.models.triage import ResearchBundle
from ..services.summarization import SummarizationService


logger = logging.getLogger(__name__)


class SummarizerAgent(BaseAgent):
    service: SummarizationService

    @override
    async def _run_async_impl(self, ctx: InvocationContext):
        if ctx.session.state.get("workflow_error"):
            logger.info("SummarizerAgent skipped due to upstream workflow error")
            return

        query = str(ctx.session.state.get("query", ""))
        bundle = ResearchBundle.from_dict(
            ctx.session.state.get("research_bundle", {"query": query, "articles": []})
        )
        logger.info(
            "SummarizerAgent launched query=%r articles=%d",
            query,
            len(bundle.articles),
        )
        try:
            brief = self.service.summarize(query, bundle)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            logger.exception("SummarizerAgent failed")
            yield Event(
                invocation_id=ctx.invocation_id,
                author=self.name,
                branch=ctx.branch,
                actions=EventActions(
                    state_delta={
                        "workflow_error": message,
                        "workflow_error_stage": "summarize",
                        "workflow_error_traceback": traceback.format_exc(),
                    }
                ),
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=f"SummarizerAgent failed: {message}")],
                ),
            )
            return
        logger.info("SummarizerAgent completed")
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            actions=EventActions(state_delta={"triage_brief": brief.to_dict()}),
            content=types.Content(
                role="model",
                parts=[types.Part(text="SummarizerAgent prepared the final brief.")],
            ),
        )


