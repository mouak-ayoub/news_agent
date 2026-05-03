from __future__ import annotations

import logging

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.utils.context_utils import Aclosing
from google.genai import types
from typing_extensions import override

from news_agent.models.triage import TriageBrief


logger = logging.getLogger(__name__)


class CoordinatorAgent(BaseAgent):
    @override
    async def _run_async_impl(self, ctx: InvocationContext):
        if not self.sub_agents:
            raise RuntimeError("CoordinatorAgent requires a sequential workflow sub-agent.")

        async with Aclosing(self.sub_agents[0].run_async(ctx)) as agen:
            async for event in agen:
                yield event

        workflow_error = ctx.session.state.get("workflow_error")
        if workflow_error:
            stage = ctx.session.state.get("workflow_error_stage", "unknown")
            logger.error(
                "CoordinatorAgent detected workflow failure stage=%s error=%s",
                stage,
                workflow_error,
            )
            return

        brief_data = ctx.session.state.get("triage_brief")
        if not brief_data:
            logger.error("CoordinatorAgent ended without triage brief in session state")
            return
        brief = TriageBrief.from_dict(brief_data)
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            content=types.Content(role="model", parts=[types.Part(text=brief.final_brief)]),
        )


