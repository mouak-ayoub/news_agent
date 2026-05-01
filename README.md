# News Agent

News Agent is a local ADK workflow for current-event triage. It researches a question, retrieves recent source material, extracts requested facts such as numbers or dates, compares outlet framing, and writes a structured JSON brief plus an HTML report.

## Project Notes

- [Learning log](docs/LEARNING_LOG.md) tracks the current baseline, recent experiments, weaknesses, and next work.
- [Architecture](docs/ARCHITECTURE.md) is the canonical Mermaid architecture view.
- [Fine-tuning plan](docs/FINE_TUNING_PLAN.md) explains what should and should not be fine-tuned.
- [Free pipeline](docs/FREE_PIPELINE.md) explains the no-OpenAI and low-cost tracks.

## Running

Hybrid config: Gemma through Gemini API for reasoning, OpenAI only for web-search retrieval.

```powershell
$env:GEMINI_API_KEY="your-google-key"
$env:openai_news_api="your-openai-key"
.\.venv\Scripts\python.exe main.py "What are the latest casualty figures in the Iran-USA conflict?" --config config\news_agent_openai.yaml --html-out reports\openai-run.html --debug
```

Gemini + FreeNewsApi config:

```powershell
$env:GEMINI_API_KEY="your-google-key"
$env:news_triage_codex_app="your-freenewsapi-key"
.\.venv\Scripts\python.exe main.py "What are the latest verified updates on global AI regulation?" --config config\news_agent_gemini.yaml --html-out reports\gemini-run.html --debug
```

Local Ollama config:

```powershell
.\.venv\Scripts\python.exe main.py "What are the latest verified updates on global AI regulation?" --config config\news_agent_ollama.yaml --html-out reports\ollama-run.html
```

Direct PyCharm/simple run:

```powershell
.\.venv\Scripts\python.exe main.py
```

`main.py` contains editable defaults:

```python
DEFAULT_QUERY = "What are the latest casualty figures in the Iran-USA conflict?"
DEFAULT_CONFIG = "news_agent_openai.yaml"
DEFAULT_HTML_OUT = "reports/openai-run.html"
DEFAULT_DEBUG = True
```

## Current Flow

```text
User query
  -> command/main
  -> ConfigLoader
  -> QuestionAnalyzer
  -> QueryPlanner
  -> SearchClient
  -> ArticleContentFetcher
  -> MetricExtractor
  -> SummarizationService
  -> TriageBrief JSON
  -> HTML report
```

## Current Components

| Component | Responsibility |
| --- | --- |
| `ConfigLoader` | Loads YAML config and outlet files into typed dataclasses. |
| `PromptService` | Loads prompt templates from `config/prompts`. |
| `DebugOutput` | Writes per-run model-call artifacts under `debug_output/` when `--debug` is enabled. |
| `QuestionAnalyzer` | Extracts topic, requested metric, answer type, and avoid-list. |
| `QueryPlanner` | Produces short search-style queries from the analyzed intent. |
| `ResearchService` | Coordinates intent analysis, planned search, article enrichment, and metric extraction. |
| `SearchClient` | Interface implemented by OpenAI web search, Google News RSS, and FreeNewsApi. |
| `ArticleContentFetcher` | Attempts to enrich candidates with article body text. |
| `MetricExtractor` | Extracts the requested metric from article text and filters numeric/date questions to metric-bearing sources. |
| `SummarizationService` | Produces the structured `TriageBrief`. |
| `AgentGraphBuilder` | Builds the ADK coordinator and sequential research/summarization pipeline. |
| `services/reporting.py` | Renders the HTML report from `config/html/report.html`. |

## Config Strategy

Models are configured per step:

```yaml
model:
  question_analysis_model_id: gemma-4-31b-it
  query_planning_model_id: gemma-4-31b-it
  candidate_filter_model_id: gemma-4-31b-it
  article_selection_model_id: gemma-4-31b-it
  metric_extraction_model_id: gemma-4-31b-it
  summary_model_id: gemma-4-31b-it
```

Search is configured separately. In `news_agent_openai.yaml`, OpenAI is used only for retrieval:

```yaml
search:
  provider: openai_web_search
  api_key_env: openai_news_api
  web_search_model_id: gpt-4.1
  web_search_prompt: web_search/web_search_research_cot
```

Prompt variants live under `config/prompts/web_search/`:

- `web_search_research_baseline.txt`: loose recall-first retrieval.
- `web_search_research_cot.txt`: retrieval plus internal selection checklist.
- `web_search_research_self_consistency.txt`: prompt-level self-consistency only.
- `web_search_research_tot.txt`: prompt-level tree-of-thought only.

Self-consistency and ToT are currently prompt techniques, not multi-call algorithms.

## Debug Output

With `--debug`, each run writes a folder such as:

```text
debug_output/<timestamp>_<query-slug>/
  run_context.json
  report.html
  model_calls/
    001_question-analysis/
      input.txt
      output.txt
    002_query-planning/
      input.txt
      output.txt
    003_openai-web-search/
      input.txt
      output.txt
      response.json
```

For OpenAI web search:

- `input.txt` is the exact prompt sent to OpenAI.
- `output.txt` is the final JSON text parsed by the app.
- `response.json` is the full Responses API object, including web-search tool query and token usage.

## Session State

```text
query
  -> session.state["query"]
  -> session.state["research_bundle"]
  -> session.state["triage_brief"]
```
