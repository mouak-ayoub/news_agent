# Free And Low-Cost Pipeline

This project keeps several configs so retrieval/model choices can be swapped without changing Python code.

## Current Configs

| Config | Retrieval | Model backend | Keys | Status |
| --- | --- | --- | --- | --- |
| `config/news_agent_openai.yaml` | OpenAI web search | Gemini API hosting Gemma | `openai_news_api`, `GEMINI_API_KEY` | Best comparison pipeline. |
| `config/news_agent_gemini.yaml` | FreeNewsApi | Gemini API hosting Gemma | `news_triage_codex_app`, `GEMINI_API_KEY` | Low-cost search experiment. |
| `config/news_agent_ollama.yaml` | direct publisher RSS | local Ollama | none for model | Local-model track. |
| `config/news_agent_free.yaml` | direct publisher RSS | placeholder heuristic backend | none | Retrieval-only/free placeholder; not the quality target. |

## Running The Low-Cost Track

Gemini + FreeNewsApi:

```powershell
$env:GEMINI_API_KEY="your-google-key"
$env:news_triage_codex_app="your-freenewsapi-key"
.\.venv\Scripts\python.exe main.py "What are the latest verified updates on global AI regulation?" --config config\news_agent_gemini.yaml --html-out reports\gemini-run.html --debug
```

Local Ollama:

```powershell
.\.venv\Scripts\python.exe main.py "What are the latest verified updates on global AI regulation?" --config config\news_agent_ollama.yaml --html-out reports\ollama-run.html
```

## What Is Actually Free

Fully free means:

- no OpenAI API,
- no hosted Gemini API,
- no paid search API,
- local model inference after the model is downloaded,
- RSS or another free retrieval source.

The closest current config is `news_agent_ollama.yaml`, but retrieval still depends on what direct publisher RSS exposes.

## Current Search Lessons

Google News RSS is useful for titles and snippets, but in Europe it often redirects through Google consent pages. Because of that, the RSS configs now keep direct publisher feeds and disable Google News fallback.

FreeNewsApi can return full article details, but API search quality depends heavily on query planning, pagination, publisher coverage, and quota. It is useful for experiments but not yet a guaranteed replacement for web search.

OpenAI web search currently gives the best retrieval behavior when the prompt uses loose recall rules. The debug output showed that a stricter CoT prompt may return `[]` if it chooses a weak natural-language search query.

## Model Strategy

Gemma should not be expected to know current news. It should work from retrieved article text.

Use Gemma for:

- question analysis,
- query planning,
- candidate filtering,
- article selection,
- metric extraction,
- final summarization.

Keep retrieval separate:

- OpenAI web search for strong comparison runs,
- FreeNewsApi for lower-cost API experiments,
- RSS for no-key/local experiments.

## Current Limits

- RSS often exposes only titles and snippets.
- Full article fetching may hit consent pages, paywalls, or bot protection.
- FreeNewsApi may return irrelevant or low-coverage candidates for some outlets.
- Prompt-only self-consistency and ToT do not run multiple independent searches.
- The system currently has no automatic second search round when retrieval is weak.

## Recommended Next Step

Add a general retrieval retry policy:

```text
first pass: loose recall prompt
if zero or weak sources:
  retry with broader keyword/domain query strategy
then:
  fetch full article body
  extract requested metric
  keep only metric-bearing articles for numeric/date questions
```

This should stay general. Do not add query-specific code or vocabulary for one conflict, country, or topic.
