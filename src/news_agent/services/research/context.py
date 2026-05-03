from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

from news_agent.models.research import ResearchIntent
from news_agent.models.research import SearchPlan
from news_agent.models.triage import ArticleRecord
from news_agent.models.triage import ResearchBundle


@dataclass(slots=True)
class ResearchContext:
    query: str
    intent: ResearchIntent | None = None
    search_plan: SearchPlan | None = None
    candidates: list[ArticleRecord] = field(default_factory=list)
    selected_articles: list[ArticleRecord] = field(default_factory=list)
    bundle: ResearchBundle | None = None
