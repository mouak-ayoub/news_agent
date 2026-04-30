# Architecture

Canonical architecture file for this repo.

## Runtime Flow

```mermaid
flowchart TD
    A["main.py"] --> B["news_agent.command.main"]
    B --> C["ConfigLoader.load(config/*.yaml)"]
    C --> D["AppConfig"]
    D --> E["run_triage(query, config)"]
    E --> F["AgentGraphBuilder(config).build()"]
    F --> G["CoordinatorAgent"]
    G --> H["SequentialAgent"]
    H --> I["ResearchAgent"]
    H --> J["SummarizerAgent"]
    I --> K["ResearchBundle"]
    K --> J
    J --> L["TriageBrief"]
    L --> M["services.reporting.write_html_report"]
```

## Package Structure

```mermaid
flowchart LR
    Root["src/news_agent"] --> Agents["agents/"]
    Root --> Models["models/"]
    Root --> Services["services/"]
    Root --> App["workflow.py"]
    Root --> Cli["command.py"]

    Agents --> AB["agent_builder.py"]
    Agents --> CO["coordinator.py"]
    Agents --> RE["researcher.py"]
    Agents --> SU["summarizer.py"]

    Models --> MC["config.py"]
    Models --> MG["generation.py"]
    Models --> MT["triage.py"]
    Models --> MU["usage.py"]

    Services --> CL["config_loader.py"]
    Services --> RS["research.py"]
    Services --> RP["reporting.py"]
    Services --> SM["summarization.py"]
    Services --> TG["text_generation.py"]
    Services --> UG["usage_guard.py"]
    Services --> SS["search/"]

    SS --> SB["base.py"]
    SS --> SF["factory.py"]
    SS --> SO["openai.py"]
    SS --> SR["rss.py"]
```

## Core Relationships

```mermaid
classDiagram
    class AppConfig
    class AgentGraphBuilder
    class CoordinatorAgent
    class SequentialAgent
    class ResearchAgent
    class ResearchService
    class SearchClient {
      <<interface>>
    }
    class OpenAIWebSearchClient
    class GoogleNewsRssSearchClient
    class SummarizerAgent
    class SummarizationService
    class TextGenerator {
      <<interface>>
    }
    class OpenAIResponsesTextGenerator
    class StaticTextGenerator
    class UsageGuard
    class ResearchBundle
    class TriageBrief

    AgentGraphBuilder --> AppConfig
    AgentGraphBuilder --> UsageGuard
    AgentGraphBuilder --> CoordinatorAgent
    CoordinatorAgent *-- SequentialAgent
    SequentialAgent --> ResearchAgent
    SequentialAgent --> SummarizerAgent

    ResearchAgent --> ResearchService
    ResearchService --> SearchClient
    SearchClient <|.. OpenAIWebSearchClient
    SearchClient <|.. GoogleNewsRssSearchClient
    ResearchAgent --> ResearchBundle

    SummarizerAgent --> SummarizationService
    SummarizationService --> TextGenerator
    TextGenerator <|.. OpenAIResponsesTextGenerator
    TextGenerator <|.. StaticTextGenerator
    SummarizerAgent --> TriageBrief
```

## Rules

- All dataclasses stay in `models/`.
- Provider-specific implementations stay at the edge (`services/search/openai.py`, `services/search/rss.py`).
- Agents do orchestration only; service logic lives in `services/`.
- `workflow.py` runs one workflow and should not contain provider-specific branching.
