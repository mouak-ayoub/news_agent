from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from news_agent.configuration.loader import load_app_config
from news_agent.configuration.settings import resolve_openai_web_search_settings
from news_agent.models.research import SearchPlan
from news_agent.models.research import default_research_intent
from news_agent.services.debug.debug_output import DebugOutput
from news_agent.services.llm.text_generation import extract_json_block
from news_agent.services.prompts.prompt_service import PromptService
from news_agent.services.search.openai.gateway import DebuggingOpenAIWebSearchGateway
from news_agent.services.search.openai.gateway import OpenAIWebSearchGateway
from news_agent.services.search.openai.gateway import OpenAIWebSearchRequest
from news_agent.services.search.openai.gateway import _web_search_call_summaries
from news_agent.services.search.openai.job_planner import OpenAISearchJobPlanner
from news_agent.services.search.openai.prompt_builder import OpenAIWebSearchPromptBuilder


QUESTIONS = [
    {
        "id": "musk_altman_lawsuit",
        "question": "Latest evolution of suit between Elon Musk and Sam Altman?",
    },
    {
        "id": "iran_usa_casualties",
        "question": "What are the latest casualty figures in the Iran-USA conflict?",
    },
    {
        "id": "gta_vi_release",
        "question": "When will GTA VI be released?",
    },
    {
        "id": "recession_forecast",
        "question": "Do outlets think there will be a recession?",
    },
]

VARIANTS = [
    {
        "id": "a_short",
        "prompt": "web_search/web_search_research_new_a_short",
        "description": "Short retrieval prompt",
    },
    {
        "id": "b_diversity",
        "prompt": "web_search/web_search_research_new_b_diversity",
        "description": "Outlet-diversity prompt",
    },
    {
        "id": "c_query_variants_first",
        "prompt": "web_search/web_search_research_new_c_query_variants_first",
        "description": "Query-variant-first prompt",
    },
    {
        "id": "d_minimal",
        "prompt": "web_search/web_search_research_new_d_minimal",
        "description": "Minimal prompt",
    },
]

CONFIG_PATH = ROOT / "config" / "news_agent_openai.yaml"
MAX_SEARCH_CALLS_PER_RUN = 1
WEB_SEARCH_MAX_TOOL_CALLS = 8
WEB_SEARCH_REASONING_EFFORT = "low"
WEB_SEARCH_TEXT_VERBOSITY = "low"
WEB_SEARCH_USE_ALLOWED_DOMAINS = True
WEB_SEARCH_INCLUDE_SOURCES = True
WEB_SEARCH_TOOL_CHOICE = "required"
WEB_SEARCH_SEARCH_CONTEXT_SIZE = "medium"
WEB_SEARCH_USE_SITE_QUERY_FILTERS = False
PREFERRED_OUTLETS = ("Reuters", "CNN", "France 24")


def main() -> int:
    run_root = (
        ROOT
        / "debug_output"
        / "prompt_experiments"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []

    for variant in VARIANTS:
        for question in QUESTIONS:
            summary = run_case(run_root, variant, question)
            summaries.append(summary)
            write_summary_files(run_root, summaries)

    write_summary_files(run_root, summaries)
    print(f"SUMMARY {run_root / 'summary.md'}", flush=True)
    return 0


def run_case(
    run_root: Path,
    variant: dict[str, str],
    question: dict[str, str],
) -> dict[str, Any]:
    case_dir = run_root / variant["id"] / question["id"]
    case_dir.mkdir(parents=True, exist_ok=True)
    print(f"START {variant['id']} :: {question['id']}", flush=True)

    started = time.perf_counter()
    summary: dict[str, Any] = {
        "variant_id": variant["id"],
        "variant_description": variant["description"],
        "prompt": variant["prompt"],
        "question_id": question["id"],
        "question": question["question"],
        "case_dir": str(case_dir),
        "status": "started",
    }

    try:
        config = load_app_config(CONFIG_PATH)
        config.search.max_search_calls_per_run = MAX_SEARCH_CALLS_PER_RUN
        config.search.web_search_max_tool_calls = WEB_SEARCH_MAX_TOOL_CALLS
        config.search.web_search_reasoning_effort = WEB_SEARCH_REASONING_EFFORT
        config.search.web_search_text_verbosity = WEB_SEARCH_TEXT_VERBOSITY
        config.search.web_search_use_allowed_domains = WEB_SEARCH_USE_ALLOWED_DOMAINS
        config.search.web_search_include_sources = WEB_SEARCH_INCLUDE_SOURCES
        config.search.web_search_tool_choice = WEB_SEARCH_TOOL_CHOICE
        config.search.web_search_search_context_size = WEB_SEARCH_SEARCH_CONTEXT_SIZE
        config.search.web_search_use_site_query_filters = WEB_SEARCH_USE_SITE_QUERY_FILTERS
        config.search.web_search_prompt = variant["prompt"]

        settings = resolve_openai_web_search_settings(config)
        outlets = config.outlets[: min(config.search.max_sources, len(config.outlets))]
        jobs = OpenAISearchJobPlanner().build_jobs(
            query=question["question"],
            plan=SearchPlan(queries=[question["question"]]),
            outlets=outlets,
            max_calls=config.search.max_search_calls_per_run,
            use_allowed_domains=config.search.web_search_use_allowed_domains,
            use_site_query_filters=config.search.web_search_use_site_query_filters,
        )
        if len(jobs) != 1:
            raise RuntimeError(f"expected one Python-level search job, got {len(jobs)}")
        job = jobs[0]

        prompt = OpenAIWebSearchPromptBuilder(PromptService()).build(
            template_name=config.search.web_search_prompt,
            query=question["question"],
            job=job,
            days_back=config.search.days_back,
            intent=default_research_intent(question["question"]),
        )

        debug_output = DebugOutput(case_dir)
        debug_output.write_json(
            "experiment_context.json",
            {
                "variant": variant,
                "question": question,
                "config_path": str(CONFIG_PATH),
                "config_overrides": {
                    "max_search_calls_per_run": MAX_SEARCH_CALLS_PER_RUN,
                    "web_search_max_tool_calls": WEB_SEARCH_MAX_TOOL_CALLS,
                    "web_search_reasoning_effort": WEB_SEARCH_REASONING_EFFORT,
                    "web_search_text_verbosity": WEB_SEARCH_TEXT_VERBOSITY,
                    "web_search_use_allowed_domains": WEB_SEARCH_USE_ALLOWED_DOMAINS,
                    "web_search_include_sources": WEB_SEARCH_INCLUDE_SOURCES,
                    "web_search_tool_choice": WEB_SEARCH_TOOL_CHOICE,
                    "web_search_search_context_size": WEB_SEARCH_SEARCH_CONTEXT_SIZE,
                    "web_search_use_site_query_filters": WEB_SEARCH_USE_SITE_QUERY_FILTERS,
                    "web_search_prompt": variant["prompt"],
                },
                "active_outlets": [asdict(outlet) for outlet in outlets],
                "python_search_job": {
                    "search_query": job.search_query,
                    "outlets": [outlet.name for outlet in job.outlets],
                    "allowed_domains": list(job.allowed_domains),
                },
            },
        )

        gateway = DebuggingOpenAIWebSearchGateway(
            OpenAIWebSearchGateway(api_key_env=settings.api_key_env),
            debug_output,
        )
        response = gateway.search(
            OpenAIWebSearchRequest(
                call_name="openai_web_search_01",
                prompt=prompt,
                search_query=job.search_query,
                outlet_names=tuple(outlet.name for outlet in job.outlets),
                model_id=settings.model_id,
                max_output_tokens=settings.max_output_tokens,
                temperature=settings.temperature,
                reasoning_effort=settings.reasoning_effort,
                max_tool_calls=settings.max_tool_calls,
                text_verbosity=settings.text_verbosity,
                allowed_domains=job.allowed_domains,
                include_sources=settings.include_sources,
                tool_choice=settings.tool_choice,
                search_context_size=settings.search_context_size,
                use_site_query_filters=settings.use_site_query_filters,
            )
        )

        raw_candidates = parse_candidates(response.raw_text)
        response_json = parse_json_or_none(response.response_dump)
        web_search_calls = _web_search_call_summaries(response.response_dump)
        completed_web_search_calls = [
            call for call in web_search_calls if call.get("status") == "completed"
        ]
        attempted_web_search_calls = len(web_search_calls)
        internal_queries = extract_internal_queries(completed_web_search_calls)
        usage = extract_usage(response_json)
        outlet_counts = count_outlets(raw_candidates)
        preferred_present = {
            outlet: outlet in outlet_counts for outlet in PREFERRED_OUTLETS
        }

        debug_output.write_json("raw_candidates.json", raw_candidates)
        summary.update(
            {
                "status": "ok",
                "latency_seconds": round(time.perf_counter() - started, 2),
                "raw_candidate_count": len(raw_candidates),
                "distinct_outlet_count": len(outlet_counts),
                "outlet_counts": outlet_counts,
                "preferred_outlets_present": preferred_present,
                "internal_search_queries": internal_queries,
                "completed_web_search_call_count": len(completed_web_search_calls),
                "attempted_web_search_call_count": attempted_web_search_calls,
                "internal_search_call_count": len(completed_web_search_calls),
                "allowed_domains": list(job.allowed_domains),
                "use_site_query_filters": config.search.web_search_use_site_query_filters,
                "include_sources": config.search.web_search_include_sources,
                "tool_choice": config.search.web_search_tool_choice,
                "search_context_size": config.search.web_search_search_context_size,
                "usage": usage,
                "python_search_query": job.search_query,
                "candidate_titles": candidate_titles(raw_candidates),
            }
        )
    except Exception as exc:
        summary.update(
            {
                "status": "error",
                "latency_seconds": round(time.perf_counter() - started, 2),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        (case_dir / "error.txt").write_text(summary["error"], encoding="utf-8")

    print(
        "DONE {variant} :: {question} status={status} raw={raw} outlets={outlets} "
        "completed_calls={calls} attempted_calls={attempted} latency={latency}s".format(
            variant=summary["variant_id"],
            question=summary["question_id"],
            status=summary["status"],
            raw=summary.get("raw_candidate_count", 0),
            outlets=summary.get("distinct_outlet_count", 0),
            calls=summary.get("completed_web_search_call_count", 0),
            attempted=summary.get("attempted_web_search_call_count", 0),
            latency=summary.get("latency_seconds", 0),
        ),
        flush=True,
    )
    return summary


def parse_candidates(raw_text: str) -> list[dict[str, Any]]:
    data = json.loads(extract_json_block(raw_text))
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def parse_json_or_none(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def extract_internal_queries(response: Any) -> list[str]:
    queries: list[str] = []

    def add(value: object) -> None:
        text = str(value).strip()
        if text and text not in queries:
            queries.append(text)

    if isinstance(response, list):
        for item in response:
            if not isinstance(item, dict):
                continue
            for query in item.get("queries", []) or []:
                add(query)
            if item.get("query"):
                add(item["query"])
        return queries

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            item_type = str(value.get("type", "")).lower()
            if "web_search" in item_type:
                action = value.get("action")
                if isinstance(action, dict):
                    for key in ("query", "search_query"):
                        if action.get(key):
                            add(action[key])
                for key in ("query", "search_query"):
                    if value.get(key):
                        add(value[key])
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(response)
    return queries


def extract_usage(response: Any) -> dict[str, Any]:
    usage = response.get("usage", {}) if isinstance(response, dict) else {}
    if not isinstance(usage, dict):
        return {}
    return {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "input_tokens_details": usage.get("input_tokens_details"),
        "output_tokens_details": usage.get("output_tokens_details"),
    }


def count_outlets(candidates: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in candidates:
        outlet = str(item.get("outlet_name") or item.get("domain") or "").strip()
        if not outlet:
            outlet = "unknown"
        counts[outlet] = counts.get(outlet, 0) + 1
    return dict(sorted(counts.items()))


def candidate_titles(candidates: list[dict[str, Any]]) -> list[dict[str, str]]:
    values: list[dict[str, str]] = []
    for item in candidates:
        values.append(
            {
                "outlet": str(item.get("outlet_name", "")),
                "title": str(item.get("title", "")),
                "url": str(item.get("url", "")),
                "search_query": str(item.get("search_query", "")),
                "article_text": str(item.get("article_text", "")),
            }
        )
    return values


def write_summary_files(run_root: Path, summaries: list[dict[str, Any]]) -> None:
    (run_root / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_root / "summary.md").write_text(render_markdown(summaries), encoding="utf-8")


def render_markdown(summaries: list[dict[str, Any]]) -> str:
    lines = [
        "# OpenAI web-search prompt experiment",
        "",
        f"- generated_at: {datetime.now().isoformat(timespec='seconds')}",
        f"- max_search_calls_per_run: {MAX_SEARCH_CALLS_PER_RUN}",
        f"- web_search_max_tool_calls: {WEB_SEARCH_MAX_TOOL_CALLS}",
        f"- web_search_reasoning_effort: {WEB_SEARCH_REASONING_EFFORT}",
        f"- web_search_text_verbosity: {WEB_SEARCH_TEXT_VERBOSITY}",
        f"- web_search_use_allowed_domains: {WEB_SEARCH_USE_ALLOWED_DOMAINS}",
        f"- web_search_include_sources: {WEB_SEARCH_INCLUDE_SOURCES}",
        f"- web_search_tool_choice: {WEB_SEARCH_TOOL_CHOICE}",
        f"- web_search_search_context_size: {WEB_SEARCH_SEARCH_CONTEXT_SIZE}",
        f"- web_search_use_site_query_filters: {WEB_SEARCH_USE_SITE_QUERY_FILTERS}",
        "- active OpenAI outlets: Reuters, CNN, France 24, Jerusalem Post, Tasnim News",
        "- note: Al Jazeera is not in the active OpenAI outlet config for these runs.",
        "",
        "| Variant | Question | Status | Raw | Outlets | Preferred | Completed calls | Attempted calls | Allowed domains | Site filters | Sources | Tool choice | Context | Latency | Tokens |",
        "| --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for item in summaries:
        usage = item.get("usage", {}) or {}
        preferred = item.get("preferred_outlets_present", {}) or {}
        preferred_text = ", ".join(
            outlet for outlet, present in preferred.items() if present
        ) or "-"
        tokens = usage.get("total_tokens")
        if tokens is None:
            tokens = "-"
        lines.append(
            "| {variant} | {question} | {status} | {raw} | {outlets} | {preferred} | {completed} | {attempted} | {allowed_domains} | {site_filters} | {sources} | {tool_choice} | {context} | {latency} | {tokens} |".format(
                variant=item.get("variant_id", ""),
                question=item.get("question_id", ""),
                status=item.get("status", ""),
                raw=item.get("raw_candidate_count", 0),
                outlets=item.get("distinct_outlet_count", 0),
                preferred=preferred_text,
                completed=item.get("completed_web_search_call_count", 0),
                attempted=item.get("attempted_web_search_call_count", 0),
                allowed_domains=len(item.get("allowed_domains", []) or []),
                site_filters=item.get("use_site_query_filters", ""),
                sources=item.get("include_sources", ""),
                tool_choice=item.get("tool_choice", ""),
                context=item.get("search_context_size", ""),
                latency=item.get("latency_seconds", 0),
                tokens=tokens,
            )
        )

    lines.extend(["", "## Details", ""])
    for item in summaries:
        lines.extend(
            [
                f"### {item.get('variant_id')} / {item.get('question_id')}",
                "",
                f"- status: {item.get('status')}",
                f"- case_dir: {item.get('case_dir')}",
                f"- python_search_query: {item.get('python_search_query', '')}",
                f"- allowed_domains: {json.dumps(item.get('allowed_domains', []), ensure_ascii=False)}",
                f"- use_site_query_filters: {item.get('use_site_query_filters', '')}",
                f"- include_sources: {item.get('include_sources', '')}",
                f"- tool_choice: {item.get('tool_choice', '')}",
                f"- search_context_size: {item.get('search_context_size', '')}",
                f"- completed_web_search_call_count: {item.get('completed_web_search_call_count', 0)}",
                f"- attempted_web_search_call_count: {item.get('attempted_web_search_call_count', 0)}",
                f"- outlet_counts: {json.dumps(item.get('outlet_counts', {}), ensure_ascii=False)}",
                f"- internal_search_queries: {json.dumps(item.get('internal_search_queries', []), ensure_ascii=False)}",
                f"- usage: {json.dumps(item.get('usage', {}), ensure_ascii=False)}",
            ]
        )
        if item.get("error"):
            lines.append(f"- error: {item['error']}")
        for candidate in item.get("candidate_titles", []):
            title = re.sub(r"\s+", " ", candidate.get("title", "")).strip()
            outlet = candidate.get("outlet", "")
            search_query = candidate.get("search_query", "")
            lines.append(f"- {outlet}: {title} | search_query={search_query}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
