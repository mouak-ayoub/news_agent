# Fine-Tuning Plan

Fine-tuning is useful for this project, but only after the baseline pipeline is stable.

The model should not be fine-tuned to memorize current news. Current events must come from search/retrieval. Fine-tuning should teach the model how to transform retrieved source text into disciplined structured output.

## Current Pipeline Context

The app now routes model calls per step:

- question analysis,
- query planning,
- candidate filtering,
- article selection,
- metric extraction,
- final summarization.

The hybrid config uses Gemma through the Gemini API for those reasoning steps and OpenAI only for web-search retrieval. Fine-tuning should target the reasoning/extraction steps, not web search.

## What To Fine-Tune

Good fine-tuning targets:

- extract reported numbers,
- attach each number to a source and date,
- distinguish local totals from broader totals,
- classify claims as confirmed, partly confirmed, disputed, unsupported, or unclear,
- separate observation, evidence-backed inference, and speculation,
- produce valid JSON matching `TriageBrief`,
- write concise final briefs without overclaiming.
- keep the output schema stable when sources are empty or weak.

Bad fine-tuning targets:

- memorizing current facts,
- memorizing outlet political positions as permanent facts,
- learning hidden motives of countries or media outlets,
- replacing web search with model memory.
- deciding which live article exists without retrieval evidence.

## Dataset Shape

Use JSONL with one training example per task.

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You extract and verify news claims. Return JSON only."
    },
    {
      "role": "user",
      "content": "Query: ...\n\nSources:\n..."
    },
    {
      "role": "assistant",
      "content": "{ \"query\": \"...\", \"main_claims\": [], \"source_findings\": [], \"uncertainties\": [] }"
    }
  ]
}
```

## First Dataset Milestones

Start small and measurable.

| Milestone | Examples | Purpose |
| --- | ---: | --- |
| Seed set | 20 | Check whether the format and labels are learnable. |
| First useful set | 100 | Improve extraction and JSON reliability. |
| Evaluation set | 30 | Hold out examples that are never used for training. |
| Stronger set | 300-500 | Improve robustness across topics and source styles. |

## Evaluation Metrics

Measure before and after fine-tuning:

- valid JSON rate,
- source URL preservation,
- number extraction accuracy,
- attribution accuracy,
- geography/time-period accuracy,
- hallucinated number count,
- unsupported motive count,
- final brief usefulness.

## Recommended Path

1. Improve retrieval and verification first.
2. Use `--debug` runs as raw data: prompts, model outputs, and OpenAI response dumps.
3. Save successful and failed examples for each step separately.
4. Manually correct 20-50 outputs into ideal outputs.
5. Use a stronger model as a teacher to draft more examples, then review them.
6. Fine-tune a small Gemma model with LoRA or QLoRA.
7. Compare the tuned model against the held-out evaluation set.
8. Keep the tuned model only if it improves metrics, not just style.

## LoRA vs QLoRA

LoRA is the preferred first approach for this project.

| Method | What It Does | When To Use |
| --- | --- | --- |
| LoRA | Freezes the base model and trains small adapter weights. | Use when the model fits comfortably in GPU memory. |
| QLoRA | Loads the base model in 4-bit quantized form, then trains LoRA adapters. | Use when GPU memory is limited. |
| Full fine-tune | Updates all model weights. | Avoid for now; expensive and easier to overfit. |

Why LoRA fits this project:

- cheaper than full fine-tuning,
- faster to iterate,
- easy to keep multiple adapters for different tasks,
- less risk of damaging the base model,
- good enough for structured extraction and JSON behavior.

Possible adapters:

- `news-extraction-lora`: extracts claims, numbers, attribution, dates, and URLs.
- `source-tone-lora`: classifies tone and framing.
- `brief-writer-lora`: writes concise final briefs from verified structured facts.

Start with one adapter:

> `news-extraction-lora`

That is the highest-value adapter because the current agent fails most at extracting and normalizing numbers from retrieved sources.

## Debug Runs As Training Data

Debug folders have the shape:

```text
debug_output/<run>/
  run_context.json
  model_calls/
    001_question-analysis/input.txt
    001_question-analysis/output.txt
    ...
```

Use these files to build examples, but do not train directly on every raw output. First classify each run:

- good output,
- bad retrieval,
- bad schema,
- bad metric extraction,
- bad summary.

For fine-tuning, only use corrected examples. Raw failed outputs are useful for evaluation and regression tests.

## Decision Rule

Fine-tuning is worth doing when prompt engineering stops improving the same repeated failures.

For this project, the first fine-tuning target should be:

> Given query plus retrieved article snippets, extract normalized source findings with numbers, attribution, country, date, uncertainty, and source URL.
