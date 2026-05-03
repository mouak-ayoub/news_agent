"""Text generation adapters and helpers."""

from news_agent.services.llm.text_generation import GenerationResult
from news_agent.services.llm.text_generation import ModelGenerationError
from news_agent.services.llm.text_generation import ModelOutputError
from news_agent.services.llm.text_generation import StaticTextGenerator
from news_agent.services.llm.text_generation import TextGenerator
from news_agent.services.llm.text_generation import build_text_generator
from news_agent.services.llm.text_generation import extract_json_block
from news_agent.services.llm.text_generation import openai_supports_reasoning_effort
from news_agent.services.llm.text_generation import openai_supports_temperature

__all__ = [
    "GenerationResult",
    "ModelGenerationError",
    "ModelOutputError",
    "StaticTextGenerator",
    "TextGenerator",
    "build_text_generator",
    "extract_json_block",
    "openai_supports_reasoning_effort",
    "openai_supports_temperature",
]


