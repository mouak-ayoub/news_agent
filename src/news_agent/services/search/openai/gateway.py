from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any
from urllib.parse import urlparse

from news_agent.services.debug.debug_output import DebugOutput
from news_agent.services.llm.text_generation import ModelGenerationError
from news_agent.services.llm.text_generation import ModelOutputError
from news_agent.services.llm.text_generation import openai_supports_reasoning_effort
from news_agent.services.llm.text_generation import openai_supports_temperature
from .domain_utils import normalize_allowed_domain


@dataclass(frozen=True, slots=True)
class OpenAIWebSearchRequest:
    call_name: str
    prompt: str
    search_query: str
    outlet_names: tuple[str, ...]
    model_id: str
    max_output_tokens: int
    temperature: float
    reasoning_effort: str = ""
    max_tool_calls: int = 0
    text_verbosity: str = ""
    allowed_domains: tuple[str, ...] = ()
    include_sources: bool = False
    tool_choice: str = ""
    search_context_size: str = ""
    use_site_query_filters: bool = False


@dataclass(frozen=True, slots=True)
class OpenAIWebSearchResponse:
    raw_text: str
    response_dump: str


class OpenAIWebSearchGateway:
    """Adapter for OpenAI Responses API web-search calls."""

    def __init__(self, *, api_key_env: str) -> None:
        if not api_key_env:
            raise ModelGenerationError("OpenAI web search requires an API key env var.")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ModelGenerationError(
                f"Environment variable `{api_key_env}` is not set."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ModelGenerationError(
                "The OpenAI package is not installed. Use provider `google_news_rss` or install the OpenAI dependency."
            ) from exc
        self.client: Any = OpenAI(api_key=api_key)

    def search(self, request: OpenAIWebSearchRequest) -> OpenAIWebSearchResponse:
        if not request.model_id:
            raise ModelGenerationError(
                "OpenAI web search requires `search.web_search_model_id`."
            )

        tool = _build_web_search_tool(request)
        request_kwargs: dict[str, Any] = {
            "model": request.model_id,
            "tools": [tool],
            "input": request.prompt,
            "max_output_tokens": request.max_output_tokens,
        }
        if request.include_sources:
            request_kwargs["include"] = ["web_search_call.action.sources"]

        normalized_tool_choice = _normalize_tool_choice(request.tool_choice)
        if normalized_tool_choice:
            request_kwargs["tool_choice"] = normalized_tool_choice

        normalized_max_tool_calls = max(0, int(request.max_tool_calls))
        if normalized_max_tool_calls:
            request_kwargs["max_tool_calls"] = normalized_max_tool_calls

        normalized_text_verbosity = _normalize_text_verbosity(request.text_verbosity)
        if normalized_text_verbosity and _openai_supports_text_verbosity(request.model_id):
            request_kwargs["text"] = {"verbosity": normalized_text_verbosity}

        normalized_reasoning_effort = _normalize_reasoning_effort(
            request.reasoning_effort
        )
        if normalized_reasoning_effort:
            if not openai_supports_reasoning_effort(request.model_id):
                raise ModelGenerationError(
                    "`search.web_search_reasoning_effort` is configured, "
                    f"but `{request.model_id}` does not support reasoning effort."
                )
            request_kwargs["reasoning"] = {"effort": normalized_reasoning_effort}

        if openai_supports_temperature(request.model_id):
            request_kwargs["temperature"] = request.temperature

        response = _create_openai_response(self.client, request_kwargs)
        response_dump = _serialize_openai_response(response)
        _raise_for_incomplete_openai_response(response)

        raw_text = _extract_openai_response_text(response)
        if not raw_text.strip():
            raise ModelOutputError(
                "OpenAI web search returned an empty final text response."
            )
        return OpenAIWebSearchResponse(
            raw_text=raw_text,
            response_dump=response_dump,
        )


class DebuggingOpenAIWebSearchGateway:
    """Decorator that writes OpenAI web-search request and response artifacts."""

    def __init__(
        self,
        inner: OpenAIWebSearchGateway,
        debug_output: DebugOutput,
    ) -> None:
        self.inner = inner
        self.debug_output = debug_output

    def search(self, request: OpenAIWebSearchRequest) -> OpenAIWebSearchResponse:
        debug_call = self.debug_output.start_model_call(
            request.call_name,
            request.prompt,
        )
        try:
            debug_call.write_artifact(
                "search_job.json",
                json.dumps(
                    {
                        "search_query": request.search_query,
                        "outlets": list(request.outlet_names),
                        "allowed_domains": list(request.allowed_domains),
                        "include_sources": request.include_sources,
                        "tool_choice": request.tool_choice,
                        "search_context_size": request.search_context_size,
                        "use_site_query_filters": request.use_site_query_filters,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )
            response = self.inner.search(request)
            debug_call.write_artifact("response.json", response.response_dump)
            web_search_calls = _web_search_call_summaries(response.response_dump)
            debug_call.write_artifact(
                "internal_web_search_calls.json",
                json.dumps(web_search_calls, ensure_ascii=False, indent=2),
            )
            debug_call.write_artifact(
                "web_search_sources.json",
                json.dumps(
                    _flatten_web_search_sources(web_search_calls),
                    ensure_ascii=False,
                    indent=2,
                ),
            )
            debug_call.write_output(response.raw_text)
            return response
        except Exception as exc:
            debug_call.write_error(exc)
            raise


def _normalize_reasoning_effort(value: str) -> str:
    normalized = value.strip().lower()
    if not normalized or normalized == "none":
        return ""
    allowed_values = {"minimal", "low", "medium", "high", "xhigh"}
    if normalized not in allowed_values:
        raise ModelGenerationError(
            "`search.web_search_reasoning_effort` must be one of: "
            + "none, "
            + ", ".join(sorted(allowed_values))
        )
    return normalized


def _normalize_text_verbosity(value: str) -> str:
    normalized = value.strip().lower()
    if not normalized:
        return ""
    allowed_values = {"low", "medium", "high"}
    if normalized not in allowed_values:
        raise ModelGenerationError(
            "`search.web_search_text_verbosity` must be one of: "
            + ", ".join(sorted(allowed_values))
        )
    return normalized


def _normalize_tool_choice(value: str) -> str:
    normalized = value.strip().lower()
    if not normalized:
        return ""
    allowed_values = {"auto", "required", "none"}
    if normalized not in allowed_values:
        raise ModelGenerationError(
            "`search.web_search_tool_choice` must be one of: "
            + ", ".join(sorted(allowed_values))
        )
    return normalized


def _normalize_search_context_size(value: str) -> str:
    normalized = value.strip().lower()
    if not normalized:
        return ""
    allowed_values = {"low", "medium", "high"}
    if normalized not in allowed_values:
        raise ModelGenerationError(
            "`search.web_search_search_context_size` must be one of: "
            + ", ".join(sorted(allowed_values))
        )
    return normalized


def _build_web_search_tool(request: OpenAIWebSearchRequest) -> dict[str, Any]:
    tool: dict[str, Any] = {"type": "web_search"}

    allowed_domains = tuple(
        domain
        for domain in (
            normalize_allowed_domain(value) for value in request.allowed_domains
        )
        if domain
    )
    if allowed_domains:
        tool["filters"] = {"allowed_domains": list(allowed_domains)}

    search_context_size = _normalize_search_context_size(request.search_context_size)
    if search_context_size:
        tool["search_context_size"] = search_context_size

    return tool


def _create_openai_response(client: Any, request_kwargs: dict[str, Any]) -> Any:
    """Create a Responses API result without depending on SDK response schemas."""
    raw_responses = getattr(client.responses, "with_raw_response", None)
    raw_create = getattr(raw_responses, "create", None) if raw_responses else None
    if not raw_create:
        return client.responses.create(**request_kwargs)

    raw_response = raw_create(**request_kwargs)
    raw_text = _read_raw_response_text(raw_response)
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return raw_text


def _read_raw_response_text(raw_response: Any) -> str:
    """Read raw SDK response text across OpenAI SDK minor/major variants."""
    text = getattr(raw_response, "text", None)
    if callable(text):
        text = text()
    if isinstance(text, str):
        return text

    http_response = getattr(raw_response, "http_response", None)
    if http_response is not None:
        text = getattr(http_response, "text", None)
        if callable(text):
            text = text()
        if isinstance(text, str):
            return text

    content = getattr(raw_response, "content", None)
    if callable(content):
        content = content()
    if isinstance(content, bytes):
        return content.decode("utf-8")
    if isinstance(content, str):
        return content

    raise TypeError("OpenAI raw response did not expose text content.")


def _raise_for_incomplete_openai_response(response: Any) -> None:
    """Surface truncated Responses API output before JSON parsing hides the cause."""
    if _field(response, "status", None) != "incomplete":
        return
    details = _field(response, "incomplete_details", {}) or {}
    reason = _field(details, "reason", "unknown")
    usage = _field(response, "usage", {}) or {}
    output_details = _field(usage, "output_tokens_details", {}) or {}
    reasoning_tokens = _field(output_details, "reasoning_tokens", None)
    output_tokens = _field(usage, "output_tokens", None)
    token_summary = ""
    if output_tokens is not None or reasoning_tokens is not None:
        token_summary = (
            f" output_tokens={output_tokens}, reasoning_tokens={reasoning_tokens}."
        )
    raise ModelOutputError(
        "OpenAI web search response was incomplete"
        f" ({reason}).{token_summary}"
    )


def _openai_supports_text_verbosity(model_id: str) -> bool:
    """Return whether the model family accepts text.verbosity."""
    return model_id.strip().lower().startswith("gpt-5")


def _extract_openai_response_text(response: Any) -> str:
    """Read final assistant text from the Responses object without hiding empties."""
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)

    chunks: list[str] = []
    for output_item in _field(response, "output", []) or []:
        for content_item in _field(output_item, "content", []) or []:
            text = _field(content_item, "text", None)
            if text:
                chunks.append(str(text))
    return "\n".join(chunks)


def _serialize_openai_response(response: Any) -> str:
    """Serialize the full provider response so debug can show tool calls and status."""
    if hasattr(response, "model_dump_json"):
        return str(response.model_dump_json(indent=2))
    return json.dumps(response, ensure_ascii=False, indent=2, default=str)


def _web_search_call_summaries(response_dump: str) -> list[dict[str, Any]]:
    try:
        response = json.loads(response_dump)
    except json.JSONDecodeError:
        return []

    calls: list[dict[str, Any]] = []
    for index, output_item in enumerate(_field(response, "output", []) or [], start=1):
        if _field(output_item, "type", "") != "web_search_call":
            continue
        action = _field(output_item, "action", {}) or {}
        sources = _source_summaries(_field(action, "sources", []) or [])
        calls.append(
            {
                "index": index,
                "status": _field(output_item, "status", ""),
                "action_type": _field(action, "type", ""),
                "query": _field(action, "query", ""),
                "queries": list(_field(action, "queries", []) or []),
                "domains": _domains_from_sources(sources),
                "sources": sources,
            }
        )
    return calls


def _source_summaries(sources: list[Any]) -> list[dict[str, str]]:
    summaries: list[dict[str, str]] = []
    for source in sources:
        url = str(_field(source, "url", "") or "")
        title = str(_field(source, "title", "") or "")
        if not url and not title:
            continue
        summaries.append({"url": url, "title": title})
    return summaries


def _domains_from_sources(sources: list[dict[str, str]]) -> list[str]:
    domains: list[str] = []
    seen: set[str] = set()
    for source in sources:
        domain = urlparse(source.get("url", "")).netloc.lower().removeprefix("www.")
        if not domain or domain in seen:
            continue
        domains.append(domain)
        seen.add(domain)
    return domains


def _flatten_web_search_sources(
    web_search_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for call in web_search_calls:
        for source in call.get("sources", []):
            values.append(
                {
                    "call_index": call.get("index"),
                    "call_status": call.get("status"),
                    "url": source.get("url", ""),
                    "title": source.get("title", ""),
                }
            )
    return values


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)

