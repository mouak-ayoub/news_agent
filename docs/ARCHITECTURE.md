# Architecture

Canonical architecture file for this repo.

## Runtime Flow

```mermaid
flowchart TD
    A["main.py"] --> B["news_agent.command.main"]
    B --> C["ConfigLoader.load(config/*.yaml)"]
    C --> D["AppConfig"]
    D --> E["run_triage(query, config)"]
    E --> P["PromptService"]
    E --> Q["QuestionAnalyzer"]
    E --> R["QueryPlanner"]
    E --> S["SearchClient"]
    E --> F["AgentGraphBuilder.build()"]
    F --> G["CoordinatorAgent"]
    G --> H["SequentialAgent"]
    H --> I["ResearchAgent"]
    H --> J["SummarizerAgent"]
    I --> K["ResearchService.research()"]
    K --> Q
    K --> R
    K --> S
    S --> AF["Article candidates"]
    R --> WSJ["Deterministic site-filtered search jobs"]
    WSJ --> S
    AF --> AC["ArticleContentFetcher"]
    AC --> ME["MetricExtractor"]
    ME --> RB["ResearchBundle"]
    RB --> J
    J --> SS["SummarizationService"]
    SS --> TB["TriageBrief"]
    TB --> HR["services.reporting.write_html_report"]
```

## Package Structure

```mermaid
flowchart LR
    Root["src/news_agent"] --> Agents["agents/"]
    Root --> Bootstrap["bootstrap/"]
    Root --> Configuration["configuration/"]
    Root --> Models["models/"]
    Root --> Services["services/"]
    Root --> Workflow["workflow.py"]
    Root --> Command["command.py"]

    Agents --> AB["agent_builder.py"]
    Agents --> CO["coordinator.py"]
    Agents --> RE["researcher.py"]
    Agents --> SU["summarizer.py"]

    Models --> MC["config.py"]
    Models --> MR["research.py"]
    Models --> MG["generation.py"]
    Models --> MT["triage.py"]

    Configuration --> CL["loader.py"]
    Services --> ResearchPkg["research/"]
    Services --> ArticlesPkg["articles/"]
    Services --> LLMPkg["llm/"]
    Services --> PromptsPkg["prompts/"]
    Services --> DebugPkg["debug/"]
    Services --> ReportingPkg["reporting/"]
    Services --> SummarizationPkg["summarization/"]
    Services --> Search["search/"]

    Configuration --> Settings["settings.py"]
    Configuration --> Validation["validation.py"]

    ResearchPkg --> RS["service.py"]
    ResearchPkg --> QA["question_analyzer.py"]
    ResearchPkg --> QP["query_planner.py"]
    ResearchPkg --> MX["metric_extractor.py"]

    ArticlesPkg --> ACF["article_content_fetcher.py"]
    ArticlesPkg --> AS["article_selector.py"]
    ArticlesPkg --> CF["candidate_filter.py"]
    ArticlesPkg --> AD["article_deduplicator.py"]

    LLMPkg --> TG["text_generation.py"]
    PromptsPkg --> PS["prompt_service.py"]
    DebugPkg --> DO["debug_output.py"]
    ReportingPkg --> RP["html_report.py"]
    SummarizationPkg --> SM["service.py"]

    Search --> SB["base.py"]
    Search --> SF["factory.py"]
    Search --> SR["rss.py"]
    Search --> SN["free_news_api.py"]
    Search --> OpenAI["openai/"]

    OpenAI --> SO["client.py"]
    OpenAI --> GW["gateway.py"]
    OpenAI --> JP["job_planner.py"]
    OpenAI --> PB["prompt_builder.py"]
    OpenAI --> ON["article_normalizer.py"]
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
    class QuestionAnalyzer
    class QueryPlanner
    class SearchClient {
      <<interface>>
    }
    class OpenAIWebSearchClient
    class WebSearchJob
    class GoogleNewsRssSearchClient
    class FreeNewsApiSearchClient
    class ArticleContentFetcher
    class MetricExtractor
    class SummarizerAgent
    class SummarizationService
    class TextGenerator {
      <<interface>>
    }
    class GeminiTextGenerator
    class OllamaTextGenerator
    class OpenAIResponsesTextGenerator
    class StaticTextGenerator
    class PromptService
    class DebugOutput
    class ResearchBundle
    class TriageBrief

    AgentGraphBuilder --> CoordinatorAgent
    CoordinatorAgent *-- SequentialAgent
    SequentialAgent --> ResearchAgent
    SequentialAgent --> SummarizerAgent

    ResearchAgent --> ResearchService
    ResearchService --> QuestionAnalyzer
    ResearchService --> QueryPlanner
    ResearchService --> SearchClient
    ResearchService --> ArticleContentFetcher
    ResearchService --> MetricExtractor
    SearchClient <|.. OpenAIWebSearchClient
    OpenAIWebSearchClient --> WebSearchJob
    SearchClient <|.. GoogleNewsRssSearchClient
    SearchClient <|.. FreeNewsApiSearchClient
    ResearchService --> ResearchBundle

    SummarizerAgent --> SummarizationService
    SummarizationService --> TextGenerator
    TextGenerator <|.. GeminiTextGenerator
    TextGenerator <|.. OllamaTextGenerator
    TextGenerator <|.. OpenAIResponsesTextGenerator
    TextGenerator <|.. StaticTextGenerator
    SummarizationService --> TriageBrief

    QuestionAnalyzer --> PromptService
    QueryPlanner --> PromptService
    OpenAIWebSearchClient --> PromptService
    MetricExtractor --> PromptService
    SummarizationService --> PromptService
    PromptService --> AppConfig
    DebugOutput --> QuestionAnalyzer
    DebugOutput --> QueryPlanner
    DebugOutput --> OpenAIWebSearchClient
    DebugOutput --> MetricExtractor
    DebugOutput --> SummarizationService
```

## Config And Prompt Flow

```mermaid
flowchart TD
    Cfg["config/news_agent_*.yaml"] --> Loader["ConfigLoader"]
    Outlets["config/outlets/*.yaml"] --> Loader
    Prompts["config/prompts/**/*.txt"] --> PS["PromptService"]
    Html["config/html/report.html"] --> Report["Reporting service"]
    Loader --> AppConfig["AppConfig"]
    AppConfig --> Models["Per-step model ids"]
    AppConfig --> SearchConfig["Search provider config"]
    PS --> ModelCalls["Model-call services"]
```

## Debug Flow

```mermaid
flowchart TD
    Run["--debug"] --> Folder["debug_output/<day>/<3-hour-bucket>/<timestamp>_<query>/"]
    Folder --> Context["run_context.json"]
    Folder --> Report["report.html"]
    Folder --> Calls["model_calls/"]
    Calls --> In["input.txt"]
    Calls --> Out["output.txt"]
    Calls --> Job["search_job.json for OpenAI web search"]
    Calls --> Resp["response.json for OpenAI web search"]
    Calls --> Err["error.txt when a model call fails"]
```

## Rules

- All dataclasses stay in `models/`.
- Provider-specific implementations stay at the edge under `services/search/`.
- OpenAI web-search internals live under `services/search/openai/`.
- Research workflow pieces live under `services/research/`.
- Summarization lives under `services/summarization/`.
- Article processing pieces live under `services/articles/`.
- Text generation infrastructure lives under `services/llm/`.
- Configuration loading, validation, and resolved settings live under `configuration/`.
- Prompt loading, debug output, and reporting each have focused subpackages.
- Agents do orchestration only; service logic lives in `services/`.
- Prompts live in `config/prompts/`, not embedded in provider code.
- HTML report structure lives in `config/html/report.html`.
- `workflow.py` wires the pipeline and should not contain provider-specific branching.
- Query-specific special cases are not allowed in code; behavior must come from prompts, config, and general services.
