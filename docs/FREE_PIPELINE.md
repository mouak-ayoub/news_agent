# Free Pipeline

The project now has two runtime configs:

- `config/news_agent.yaml`: OpenAI-backed pipeline.
- `config/news_agent_free.yaml`: free baseline with Google News RSS and heuristic summarization.

Run the free baseline:

```powershell
.\.venv\Scripts\python.exe main.py "What are the latest verified updates on global AI regulation?" --config config\news_agent_free.yaml --html-out reports\free-run.html
```

## What Is Free Today

The free config uses:

- Google News RSS for retrieval,
- local Python filtering for relevance,
- heuristic extraction/summarization,
- local HTML report generation.

It does not require:

- `NEWS_AGENT_KEY`,
- OpenAI,
- paid Google Search grounding.

## Current Limits

The free baseline can find relevant article titles and snippets, but it usually cannot extract all numbers from the full article body.

Known limits:

- Google News RSS often exposes only titles and short snippets.
- Some RSS links point through Google News redirects.
- Full article fetching may hit consent pages, paywalls, or bot protection.
- The heuristic summarizer cannot reason like a strong model.

## Gemma Question

For a fully local/free setup, Gemma should run locally.

That means either:

- download Gemma weights and run them through a local inference stack, or
- use a local runner such as Ollama or LM Studio if it supports the target Gemma model.

If Gemma is used through a Google-hosted API, then the model is not downloaded locally. That may still be free within limits, but it depends on API availability and terms.

## Recommended Next Step

Add a local model backend:

```text
Google News RSS
  -> article/snippet extraction
  -> local Gemma extraction
  -> heuristic verification
  -> final report
```

Start with Gemma only for structured extraction:

- numbers,
- dates,
- country/geography,
- attribution,
- uncertainty,
- source URL.

Do not use Gemma to memorize current news.
