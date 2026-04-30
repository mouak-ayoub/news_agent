# News Agent

## Project Notes

- [Learning log](docs/LEARNING_LOG.md) tracks the current baseline, weaknesses, target output quality, and next experiments as the project evolves with the course.
- [Architecture](docs/ARCHITECTURE.md) explains the current package structure and object relationships.
- [Fine-tuning plan](docs/FINE_TUNING_PLAN.md) explains what should and should not be fine-tuned for this news triage agent.
- [Free pipeline](docs/FREE_PIPELINE.md) explains the no-OpenAI baseline and its limits.

## Running

OpenAI-backed config:

```powershell
.\.venv\Scripts\python.exe main.py "What are the latest verified updates on global AI regulation?" --config config\news_agent.yaml --html-out reports\openai-run.html
```

Free baseline config:

```powershell
.\.venv\Scripts\python.exe main.py "What are the latest verified updates on global AI regulation?" --config config\news_agent_free.yaml --html-out reports\free-run.html
```

Hosted Gemma 4 config via Gemini API:

```powershell
$env:GEMINI_API_KEY="your-key"
.\.venv\Scripts\python.exe main.py "What are the latest verified updates on global AI regulation?" --config config\news_agent_gemini.yaml --html-out reports\gemma4-run.html
```

Local Ollama + Gemma 4 config:

```powershell
.\.venv\Scripts\python.exe main.py "What are the latest verified updates on global AI regulation?" --config config\news_agent_ollama.yaml --html-out reports\ollama-run.html
```

If no config is passed, `main.py` now chooses automatically:
- `config/news_agent_gemini.yaml` when `GEMINI_API_KEY` or `GOOGLE_API_KEY` is set
- otherwise `config/news_agent_free.yaml`

So this shorter form also works:

```powershell
.\.venv\Scripts\python.exe main.py "What are the latest verified updates on global AI regulation?" --html-out reports\free-run.html
```

You can also edit `DEFAULT_QUERY`, `DEFAULT_CONFIG`, and `DEFAULT_HTML_OUT` in `main.py`, then run:

```powershell
.\.venv\Scripts\python.exe main.py
```

The free baseline uses Google News RSS plus heuristic summarization. It does not need `NEWS_AGENT_KEY`.
The Gemini config uses Google News RSS plus hosted `gemini-2.5-flash-lite` for article curation/relevance and hosted `gemma-4-31b-it` for final summarization.
The Ollama config uses Google News RSS plus local `gemma4:e4b` for article curation and summarization.

## Current Flow

```text
User query
  -> main.py / news_agent.command
  -> ConfigLoader
  -> AppConfig
  -> run_triage()
  -> AgentGraphBuilder
  -> CoordinatorAgent
  -> ResearchAgent
  -> SummarizerAgent
  -> TriageBrief JSON
  -> HTML report
```

## Current Components

| Component | Responsibility |
| --- | --- |
| `ConfigLoader` | Loads YAML config into typed config models. |
| `models/config.py` | Holds config dataclasses such as `AppConfig` and `SearchConfig`. |
| `models/generation.py` | Holds generation result dataclasses used by model services. |
| `models/triage.py` | Holds result dataclasses such as `ResearchBundle` and `TriageBrief`. |
| `models/usage.py` | Holds usage and cost dataclasses. |
| `CoordinatorAgent` | Owns the ADK workflow and returns the final brief. |
| `AgentGraphBuilder` | Assembles the ADK coordinator, sequential pipeline, and agent services. |
| `ResearchAgent` | Calls `ResearchService` and stores a `ResearchBundle` in session state. |
| `ResearchService` | Depends on one `SearchClient` and delegates retrieval to it. |
| `services/search/openai.py` | OpenAI web-search implementation. |
| `services/search/rss.py` | Google News RSS implementation for the free pipeline. |
| `SummarizerAgent` | Calls `SummarizationService` and stores the final brief. |
| `SummarizationService` | Produces the structured `TriageBrief` from retrieved sources. |
| `services/text_generation.py` | Holds text-generator implementations used by summarization. |
| `services/reporting.py` | Renders and writes the HTML report. |
| `services/usage_guard.py` | Estimates and records model/search cost. |

## Session State

```text
query
  -> session.state["query"]
  -> session.state["research_bundle"]
  -> session.state["triage_brief"]
```
