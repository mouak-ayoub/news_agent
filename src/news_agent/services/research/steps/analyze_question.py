from __future__ import annotations

from news_agent.services.research.context import ResearchContext
from news_agent.services.research.question_analyzer import QuestionAnalyzer


class AnalyzeQuestionStep:
    def __init__(self, question_analyzer: QuestionAnalyzer | None) -> None:
        self.question_analyzer = question_analyzer

    def run(self, context: ResearchContext) -> ResearchContext:
        if self.question_analyzer:
            context.intent = self.question_analyzer.analyze(context.query)
        return context
