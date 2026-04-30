# News Agent Learning Log

This document is the living memory for the project. Update it whenever the agent changes, an experiment gives a useful result, or the course material suggests a better architecture.

## Project Direction

Build a news triage agent that can answer questions about current events with disciplined reasoning:

- search recent sources,
- extract the main claims,
- compare framing across outlets and countries,
- separate observation, inference, and speculation,
- assess uncertainty instead of pretending to know hidden motives,
- return a structured brief that is useful for decision-making.

The goal is not just to summarize news. The goal is to explain what appears true, what remains uncertain, and why different outlets may frame the same event differently.

## Current Baseline

Current architecture:

- `CoordinatorAgent` owns the workflow.
- `ResearchAgent` searches for recent reporting using the configured search provider.
- `SummarizerAgent` creates the final triage brief.
- `UsageGuard` tracks estimated cost and protects paid runs from accidental overspend. The code comment in `src/news_agent/services/usage_guard.py` explains why it exists.
- HTML report generation exists.
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

## Current Weaknesses

The output is not yet where we want it.

Known issues:

- Retrieval can return weak or partial sources.
- The current architecture has only two real working agents: research and summarization.
- Source classification is mostly based on static outlet config, not actual article content.
- Veracity judgment is still too dependent on the summarizer model.
- There is no explicit verification loop yet.
- There is no scoring system for source diversity, evidence strength, or contradiction.
- The final brief can sound plausible without proving enough from the retrieved sources.
- The project depends on local environment setup; this PC currently needs dependencies installed in `.venv`.
- Keep Mermaid diagrams in `docs/ARCHITECTURE.md` as the canonical architecture view.

Current service layout:

- Config loading lives in `src/news_agent/services/config_loader.py`.
- Data-only classes live under `src/news_agent/models/`.
- Search provider implementations live under `src/news_agent/services/search/`.
- `ResearchService` depends on one `SearchClient` interface, not directly on OpenAI or RSS.
- Summarization logic lives in `src/news_agent/services/summarization.py`.
- Text generation implementations live in `src/news_agent/services/text_generation.py`.
- Usage accounting models live in `src/news_agent/models/usage.py`.
- Budget enforcement lives in `src/news_agent/services/usage_guard.py`.
- ADK graph assembly lives in `src/news_agent/agents/agent_builder.py`.

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

Keep the project simple, but make the agents more explicit:

1. `ResearchAgent`
   - retrieves 3 to 5 relevant sources,
   - records search query and source metadata,
   - rejects duplicates and off-topic results.

2. `ExtractionAgent`
   - extracts claims, actors, dates, locations, numbers, and quoted positions.

3. `SourceProfileAgent`
   - classifies outlet type, country, orientation, and tone.
   - separates static outlet metadata from article-specific tone.

4. `VerificationAgent`
   - checks whether the main claim is confirmed, partly confirmed, disputed, unsupported, or unclear.
   - decides whether a second search round is needed.

5. `BriefWriterAgent`
   - writes the final JSON and readable brief.
   - must preserve the observation / inference / speculation separation.

For now, do not add long-term memory, vector databases, or complex autonomous loops.

## Model Strategy

Ambition:

- Use a smaller open model, such as Gemma 4, as much as possible.
- Make the system feel strong through retrieval, tools, structured extraction, verification, and evaluation.
- Do not expect fine-tuning alone to make a small model generally equal to a frontier model.

Practical direction:

- Use the small model for narrow structured tasks:
  - entity extraction,
  - reported-number extraction,
  - source tone classification,
  - JSON normalization,
  - first-pass brief drafting.
- Use a stronger hosted model only as:
  - teacher for creating high-quality examples,
  - evaluator for judging outputs,
  - fallback for hard synthesis while the local model is weak.
- Keep the facts in retrieved sources and structured state, not inside the model weights.
- Fine-tune only after we have examples and evaluation cases.

Success should be measured against the task, not by general chat ability:

- valid JSON rate,
- number extraction accuracy,
- correct attribution of numbers,
- source diversity,
- hallucination rate,
- uncertainty quality,
- cost per run,
- latency per run.

The target is not "small model beats frontier model at everything." The target is "small model plus good agent design produces reliable news triage briefs for this project."

Fine-tuning plan:

- See [FINE_TUNING_PLAN.md](FINE_TUNING_PLAN.md).
- Fine-tune only after retrieval, extraction, and evaluation are stable enough to produce training examples.
- First target: normalize source findings with numbers, attribution, country, date, uncertainty, and source URL.

## Free Pipeline Track

We keep two configs:

- `config/news_agent.yaml`: OpenAI-backed pipeline for comparison and teacher/evaluator use.
- `config/news_agent_free.yaml`: free baseline using Google News RSS plus heuristic summarization.
- See [FREE_PIPELINE.md](FREE_PIPELINE.md).

The free config is intentionally weaker, but it gives us a zero-cost baseline. The next step is to replace the heuristic summarizer with a local Gemma backend.

Gemma options:

- Local Gemma: download weights or use a local runner such as Ollama/LM Studio. Best for a fully local/free setup after download.
- Gemma through Google API: no local download, but depends on API limits and does not provide Google Search grounding for Gemma itself.

For the free/local track, Gemma should learn task behavior from retrieved text, not memorize current events.

## Experiment Queue

Use this table as the project advances.

| Date | Experiment | Expected Improvement | Result | Decision |
| --- | --- | --- | --- | --- |
| 2026-04-24 | Establish living project log | Make progress and failures trackable | Started | Keep |
| 2026-04-24 | Baseline query: Iran war death toll | Evaluate current output against manual web research | Agent retrieved too few usable sources and missed key current casualty figures | Improve retrieval and add verification |
| 2026-04-24 | Add explicit verification stage | Better veracity labels and uncertainty | Not started | Pending |
| 2026-04-24 | Add source diversity scoring | Avoid one-sided retrieval | Not started | Pending |
| 2026-04-24 | Add extraction before summarization | Better grounding of final brief | Not started | Pending |
| 2026-04-29 | Add free config | Run without OpenAI key or paid model calls | Google News RSS now retrieves relevant titles, but body-level numeric extraction still needs article fetch/local model | Keep and improve |

## Historical Baseline Evaluation: Iran War Death Toll

This section records one old evaluation case. It must not become a special code
path or vocabulary list in the agent.

Query tested:

> What is the current number of deaths in the Iran war, and how do casualty figures differ across Iranian, Israeli, US, French, and regional outlets?

Agent output:

- Retrieved 11 configured outlet slots, but only 2 usable article links.
- Only Le Monde supplied explicit death numbers.
- Al Jazeera was retrieved but its numeric tracker values were not extracted.
- Reuters, CNN, Fox News, Le Figaro, Jerusalem Post, Haaretz, Tasnim, El Pais, and ABC all showed "No strong recent article retrieved."
- Final brief was cautious and avoided false certainty, which is good.
- The answer was not operational enough because it did not state the best current casualty range.

Manual comparison:

- A stronger answer should have found the Al Jazeera live tracker and extracted its country-by-country numbers.
- It should have cross-checked Iran's official toll against HRANA and later AFP-syndicated reports.
- It should have separated Iran-only deaths from regional deaths in Lebanon, Israel, Gulf states, Iraq, and US forces.
- It should have explained why Iranian official figures, HRANA figures, Israeli military claims, and regional outlet trackers may diverge.

Next fix suggested by this run:

- Add a verification/extraction step that explicitly asks: "What exact numbers did this source report, who does it attribute them to, and what geography/time period do they cover?"
- Do not let the final brief pass when fewer than 3 usable sources include numeric claims for a numeric query.
- Add a second search round when a numeric question returns too few strong numeric sources.

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

Before improving the agent, make the baseline runnable:

1. Install dependencies in `.venv`.
2. Run the unit tests.
3. Keep the new `services/search` provider split clean as retrieval improves.
4. Run one real query and save the output as a baseline.
5. Compare the baseline against the target output quality above.
