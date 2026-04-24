from __future__ import annotations

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.utils.context_utils import Aclosing
from google.genai import types
from typing_extensions import override

from ..schemas import TriageBrief


class CoordinatorAgent(BaseAgent):
    @override
    async def _run_async_impl(self, ctx: InvocationContext):
        if not self.sub_agents:
            raise RuntimeError("CoordinatorAgent requires a sequential workflow sub-agent.")

        async with Aclosing(self.sub_agents[0].run_async(ctx)) as agen:
            async for event in agen:
                yield event

        brief_data = ctx.session.state.get("triage_brief")
        if not brief_data:
            raise RuntimeError("No triage brief was written to the session state.")
        brief = TriageBrief.from_dict(brief_data)
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            content=types.Content(role="model", parts=[types.Part(text=brief.final_brief)]),
        )
