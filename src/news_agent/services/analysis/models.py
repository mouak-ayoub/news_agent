"""Compatibility exports for analysis dataclasses."""

from news_agent.models.analysis import AnalysisBundle
from news_agent.models.analysis import Confidence
from news_agent.models.analysis import EvidenceBasedAnalysis
from news_agent.models.analysis import SpeculativeRedTeamAnalysis

__all__ = [
    "AnalysisBundle",
    "Confidence",
    "EvidenceBasedAnalysis",
    "SpeculativeRedTeamAnalysis",
]
