# News Agent Learning Log

This document is the living memory for the project. Update it whenever the agent changes, an experiment gives a useful result, or the course material suggests a better architecture.

## Project Direction

Build a news triage agent that can answer questions about current events with disciplined reasoning:

- search recent sources,
- extract the main claims and requested numbers/dates,
- compare framing across outlets and countries,
- separate observation, evidence-backed inference, and speculation,
- assess uncertainty instead of pretending to know hidden motives,
- return a structured brief that is useful for decision-making.

The goal is not just to summarize news. The goal is to explain what appears true, what remains uncertain, and why different outlets may frame the same event differently.

## Current Baseline

Current architecture:

- `CoordinatorAgent` owns the ADK workflow.
- `ResearchAgent` calls `ResearchService`.
- `ResearchService` runs question analysis, query planning, search, article enrichment, and metric extraction.
- `SummarizerAgent` calls `SummarizationService`.
- HTML report generation uses `config/html/report.html`.
- Prompts live under `config/prompts/`.
- Outlet definitions live under `config/outlets/`.
- Debug runs write model-call inputs and outputs under `debug_output/`.
- Unit tests cover config loading, pipeline behavior, and report rendering.

Current output schema includes:

- main claims,
- entities,
- source profiles,
- source findings,
- framing analysis,
- historical context,
- uncertainties,
- observation / inference / speculation,
- final brief.

## Current Service Layout

- Config loading lives in `src/news_agent/services/config_loader.py`.
- Data-only classes live under `src/news_agent/models/`.
- Search provider implementations live under `src/news_agent/services/search/`.
- `ResearchService` depends on one `SearchClient` interface, not directly on OpenAI, RSS, or FreeNewsApi.
- Question analysis lives in `src/news_agent/services/question_analyzer.py`.
- Query planning lives in `src/news_agent/services/query_planner.py`.
- Full article enrichment lives in `src/news_agent/services/article_content_fetcher.py`.
- Metric extraction lives in `src/news_agent/services/metric_extractor.py`.
- Summarization logic lives in `src/news_agent/services/summarization.py`.
- Text generation implementations live in `src/news_agent/services/text_generation.py`.
- ADK graph assembly lives in `src/news_agent/agents/agent_builder.py`.

## Model And Search Strategy

Current hybrid comparison config:

- reasoning backend: Gemini API,
- reasoning model: `gemma-4-31b-it`,
- web-search provider: OpenAI Responses API web search,
- web-search model: `gpt-4.1`,
- OpenAI API key env var: `openai_news_api`,
- Gemini API key env var: `GEMINI_API_KEY`.

Model calls are configured per step:

- `question_analysis_model_id`,
- `query_planning_model_id`,
- `candidate_filter_model_id`,
- `article_selection_model_id`,
- `metric_extraction_model_id`,
- `summary_model_id`.

Search provider is configured separately:

- `openai_web_search` for comparison/strong retrieval,
- `free_news_api` for lower-cost API experiments,
- `google_news_rss` for direct RSS/no-key retrieval.

## Prompt Strategy

Web-search prompt variants live in `config/prompts/web_search/`.

Current meanings:

- `web_search_research_baseline.txt`: loose recall-first search. It allows partial relevance so we do not miss useful candidates.
- `web_search_research_cot.txt`: prompt-level chain-of-thought style. It separates retrieval and selection but still runs only one OpenAI call.
- `web_search_research_self_consistency.txt`: prompt-level self-consistency only; it does not run multiple independent calls.
- `web_search_research_tot.txt`: prompt-level tree-of-thought only; it does not run separate branches in code.

Important lesson from 2026-05-01:

- Baseline loose recall found usable Al Jazeera and Jerusalem Post candidates for a casualty-number query.
- CoT returned `[]` when it chose the raw natural-language question as its web-search query.
- Search query quality matters before selection quality. Retrieval should prefer keyword/domain queries over raw natural-language questions.

## Debug Output

Use `--debug` for prompt engineering and provider inspection.

Debug run shape:

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
- `output.txt` is the final JSON parsed by the app.
- `response.json` is the full Responses API object, including the real web-search query chosen by the model.

This fixed a misleading earlier behavior where empty OpenAI final text was silently treated as `[]`.

## Current Weaknesses

Known issues:

- Retrieval can still return weak, partial, or too few sources.
- OpenAI web search may choose only one search query internally.
- Prompt-level CoT/Self-Consistency/ToT are not true multi-call algorithms.
- Source classification is mostly based on static outlet config, not article-specific tone.
- Veracity judgment is still too dependent on the summarizer model.
- There is no explicit verification loop yet.
- There is no source diversity scoring.
- The final summary schema can drift when no sources are found.
- The project depends on local environment setup and API keys for some configs.

## Target Output Quality

A good answer should be short, structured, and careful.

It should clearly say:

- what the source literally says,
- what is confirmed by multiple sources,
- what is only inferred,
- what is speculation,
- which outlets disagree and how,
- what evidence is missing,
- whether another search round was needed.

Bad output patterns to avoid:

- confident claims based on one weak source,
- vague phrases like "some experts say" without source grounding,
- treating country or media motives as facts,
- long historical background that distracts from the current event,
- final briefs that sound polished but do not cite concrete findings.

## Proposed Next Architecture

Keep the project simple, but make the research evidence path stronger:

1. `ResearchAgent`
   - retrieves relevant sources,
   - records search query and source metadata,
   - rejects duplicates and off-topic results.

2. `ExtractionAgent` or service stage
   - extracts claims, actors, dates, locations, numbers, and quoted positions.

3. `VerificationAgent` or service stage
   - checks whether the main claim is confirmed, partly confirmed, disputed, unsupported, or unclear.
   - decides whether a second search round is needed.

4. `BriefWriterAgent`
   - writes the final JSON and readable brief.
   - preserves observation / inference / speculation separation.

For now, do not add long-term memory, vector databases, or complex autonomous loops.

## Fine-Tuning Direction

See [FINE_TUNING_PLAN.md](FINE_TUNING_PLAN.md).

Do not fine-tune the model to memorize current news. Fine-tuning should target:

- JSON reliability,
- source finding normalization,
- number/date extraction,
- attribution,
- uncertainty labels,
- final brief discipline.

The first useful adapter remains:

> `news-extraction-lora`

## Free Pipeline Track

See [FREE_PIPELINE.md](FREE_PIPELINE.md).

The free/local goal is:

```text
direct publisher RSS or free retrieval
  -> local Gemma/Ollama model
  -> metric extraction
  -> final report
```

Gemma should learn task behavior from retrieved text, not memorize current events.

## Experiment Queue

| Date | Experiment | Expected Improvement | Result | Decision |
| --- | --- | --- | --- | --- |
| 2026-04-24 | Establish living project log | Make progress and failures trackable | Started | Keep |
| 2026-04-24 | Baseline query: Iran war death toll | Evaluate current output against manual web research | Agent retrieved too few usable sources and missed key current casualty figures | Improve retrieval and add verification |
| 2026-04-29 | Add free config | Run without OpenAI key or paid model calls | RSS retrieves titles/snippets; body-level extraction remains limited | Keep as no-key track |
| 2026-05-01 | Add per-step model ids | Make model routing explicit | Implemented in config and `ModelConfig.model_id_for_step()` | Keep |
| 2026-05-01 | Add prompt variants | Compare baseline, CoT, self-consistency, ToT | Baseline loose recall currently retrieves better than CoT for the casualty-number test | Keep variants, improve CoT retrieval rules |
| 2026-05-01 | Add model-call debug output | Inspect prompt/input/output for every model call | Implemented under `debug_output/<run>/model_calls/` | Keep |
| 2026-05-01 | OpenAI response dump | Explain empty/odd web-search results | `response.json` now records actual tool query and full Responses object | Keep |

## Historical Baseline Evaluation: Iran War Death Toll

This section records one old evaluation case. It must not become a special code path or vocabulary list in the agent.

Query tested:

> What is the current number of deaths in the Iran war, and how do casualty figures differ across Iranian, Israeli, US, French, and regional outlets?

Old agent output:

- Retrieved configured outlet slots, but too few usable article links.
- Numeric tracker values were missed or inconsistently extracted.
- Final brief was cautious, which was good, but not operational enough.

Newer lesson:

- Baseline loose OpenAI web search found Al Jazeera and Jerusalem Post candidate articles.
- Metric extraction found:
  - `1,250 killed and more than 12,000 wounded` from Al Jazeera,
  - `13 killed, about 200 wounded` from Jerusalem Post.
- The procedure still needs more outlet diversity and a general second search round.

Next fix suggested by this run:

- Add a second search round when a numeric question returns too few strong numeric sources.
- Do not let the final brief pass as successful when zero sources are found for a numeric/date query.
- Keep all logic general; do not add query-specific vocabulary or country-specific branches.

## Course Connection

Google/Kaggle agent course ideas to apply:

- sequential workflow first,
- manager/coordinator pattern second,
- loop pattern only when evidence is weak,
- tool use with clear stopping criteria,
- structured state passed between agents.

Older GenAI course ideas to apply:

- prompt discipline,
- structured JSON outputs,
- hallucination checks,
- clear evaluation examples,
- compare model output against expected behavior.

## Next Immediate Step

1. Keep baseline loose recall as the stable search prompt.
2. Improve CoT so it forces keyword/domain retrieval before selection.
3. Add a general second retrieval attempt when search returns zero or weak candidates.
4. Tighten summarization schema for empty-source cases.
5. Run the same debug query after each prompt/code change and compare model-call folders.
