from __future__ import annotations

from datetime import datetime
from datetime import timedelta
import json
from pathlib import Path
import tempfile
import unittest

from news_agent.models.config import AppConfig
from news_agent.models.config import ModelConfig
from news_agent.models.config import OutletConfig
from news_agent.models.config import SearchConfig
from news_agent.models.research import ResearchIntent
from news_agent.models.research import SearchPlan
from news_agent.models.triage import ArticleRecord
from news_agent.models.triage import ResearchBundle
from news_agent.configuration.settings import OpenAIWebSearchSettings
from news_agent.services.research import ResearchService
from news_agent.services.debug.debug_output import DebugOutput
from news_agent.services.search import build_search_client
from news_agent.services.search.free_news_api import FreeNewsApiSearchClient
from news_agent.services.search.openai import OpenAIWebSearchClient
from news_agent.services.search.openai.article_normalizer import OpenAIArticleNormalizer
from news_agent.services.search.openai.gateway import DebuggingOpenAIWebSearchGateway
from news_agent.services.search.openai.gateway import OpenAIWebSearchRequest
from news_agent.services.search.openai.gateway import OpenAIWebSearchResponse
from news_agent.services.search.openai.gateway import _create_openai_response
from news_agent.services.search.openai.gateway import _extract_openai_response_text
from news_agent.services.search.openai.gateway import _raise_for_incomplete_openai_response
from news_agent.services.search.openai.job_planner import OpenAISearchJobPlanner
from news_agent.services.search.rss import GoogleNewsRssSearchClient
from news_agent.services.llm.text_generation import ModelOutputError


class FakeSearchClient:
    def __init__(self, articles: list[ArticleRecord] | None = None) -> None:
        self.received_plan: SearchPlan | None = None
        self.received_intent: ResearchIntent | None = None
        self.articles = articles or [
            ArticleRecord(
                title="Example title",
                url="https://example.com/news",
                outlet_name="Example",
                domain="example.com",
                country="France",
                medium_type="newspaper",
                orientation="center",
                published_at=None,
                snippet="Example snippet",
                article_text="Example article text",
                search_query="query",
            )
        ]

    def search_candidates(
        self,
        query: str,
        plan: SearchPlan | None = None,
        intent: ResearchIntent | None = None,
    ) -> list[ArticleRecord]:
        self.received_plan = plan
        self.received_intent = intent
        for article in self.articles:
            article.search_query = query
        return list(self.articles)


class FakeQuestionAnalyzer:
    def analyze(self, query: str) -> ResearchIntent:
        return ResearchIntent(
            topic="AI regulation",
            requested_metric="number of approved laws",
            expected_answer_type="count",
            time_sensitivity="latest",
            must_find=["approved laws", "count"],
            avoid=["opinion only"],
        )


class FakeQueryPlanner:
    def plan(self, query: str, intent: ResearchIntent) -> SearchPlan:
        return SearchPlan(queries=["AI regulation approved laws count"])


class FakeMetricExtractor:
    def enrich_bundle(self, bundle: ResearchBundle) -> ResearchBundle:
        bundle.articles[0].metric_found = True
        bundle.articles[0].metric_value = "3 laws"
        return bundle


class FakeArticleSelector:
    def __init__(self, selected_index: int = 0) -> None:
        self.selected_index = selected_index
        self.calls: list[tuple[str, str, list[str]]] = []

    def choose_best_article(
        self,
        query: str,
        outlet: OutletConfig,
        candidates: list[ArticleRecord],
        intent: ResearchIntent | None = None,
    ) -> ArticleRecord | None:
        self.calls.append((query, outlet.name, [article.url for article in candidates]))
        if not candidates:
            return None
        if self.selected_index == -1:
            return None
        if self.selected_index >= len(candidates):
            return candidates[-1]
        return candidates[self.selected_index]

    def choose_one_per_outlet(
        self,
        query: str,
        outlets: list[OutletConfig],
        candidates: list[ArticleRecord],
        intent: ResearchIntent | None = None,
    ) -> list[ArticleRecord]:
        selected: list[ArticleRecord] = []
        for outlet in outlets:
            article = self.choose_best_article(
                query,
                outlet,
                [
                    candidate
                    for candidate in candidates
                    if candidate.outlet_name == outlet.name
                ],
                intent=intent,
            )
            if article:
                selected.append(article)
        return selected


class FakeRawResponse:
    def __init__(self, body: dict[str, object]) -> None:
        self.text = json.dumps(body)


class FakeRawResponses:
    def __init__(self, body: dict[str, object]) -> None:
        self.body = body
        self.create_kwargs: dict[str, object] | None = None

    def create(self, **kwargs: object) -> FakeRawResponse:
        self.create_kwargs = kwargs
        return FakeRawResponse(self.body)


class FakeResponses:
    def __init__(self, body: dict[str, object]) -> None:
        self.with_raw_response = FakeRawResponses(body)

    def create(self, **kwargs: object) -> object:
        raise AssertionError("raw response path should be used")


class FakeOpenAIClient:
    def __init__(self, body: dict[str, object]) -> None:
        self.responses = FakeResponses(body)


class FakeOpenAIWebSearchGateway:
    def __init__(self) -> None:
        self.request: OpenAIWebSearchRequest | None = None

    def search(self, request: OpenAIWebSearchRequest) -> OpenAIWebSearchResponse:
        self.request = request
        return OpenAIWebSearchResponse(
            raw_text='[{"title":"Example","url":"https://example.com/a"}]',
            response_dump='{"status":"completed"}',
        )


class FakeOpenAIWebSearchPromptBuilder:
    def build(self, **kwargs: object) -> str:
        _ = kwargs
        return "OpenAI web-search prompt"


class FakeFreeNewsApiSearchClient(FreeNewsApiSearchClient):
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.search_config = config.search

    def _collect_listing_items(
        self,
        query: str,
        plan: SearchPlan | None,
    ) -> list[dict[str, object]]:
        _ = query, plan
        return [{"uuid": "first"}, {"uuid": "second"}]

    def _fetch_article_details(
        self,
        listing_items: list[dict[str, object]],
    ) -> list[ArticleRecord]:
        return [
            ArticleRecord(
                title=f"FreeNewsApi article {index}",
                url=f"https://publisher.example.com/{index}",
                outlet_name="Publisher",
                domain="publisher.example.com",
                country="global",
                medium_type="news API / publisher",
                orientation="unknown",
                published_at=None,
                snippet="Snippet",
                article_text="Body",
                search_query=str(item["uuid"]),
            )
            for index, item in enumerate(listing_items, start=1)
        ]


class FakeRssSearchClient(GoogleNewsRssSearchClient):
    def __init__(
        self,
        config: AppConfig,
        article_selector: FakeArticleSelector | None = None,
    ) -> None:
        super().__init__(
            config=config,
        )
        self.article_selector = article_selector or FakeArticleSelector()
        self.direct_calls: list[str] = []

    def search_outlet(
        self,
        query: str,
        outlet: OutletConfig,
        plan: SearchPlan | None = None,
        intent: ResearchIntent | None = None,
    ) -> list[ArticleRecord]:
        self.direct_calls.append(outlet.name)
        if outlet.name != "Example":
            return []
        return [
            ArticleRecord(
                title="Direct outlet result about AI regulation",
                url="https://example.com/direct",
                outlet_name=outlet.name,
                domain=outlet.domain,
                country=outlet.country,
                medium_type=outlet.medium_type,
                orientation=outlet.orientation,
                published_at=None,
                snippet="Direct result",
                article_text="Direct result",
                search_query=query,
            )
        ]

    def search_curated(
        self,
        query: str,
        plan: SearchPlan | None = None,
        intent: ResearchIntent | None = None,
    ) -> list[ArticleRecord]:
        outlet = self.outlets[1]
        return [
            ArticleRecord(
                title="Curated fallback result about AI regulation",
                url="https://second.example.com/fallback",
                outlet_name=outlet.name,
                domain=outlet.domain,
                country=outlet.country,
                medium_type=outlet.medium_type,
                orientation=outlet.orientation,
                published_at=None,
                snippet="Fallback result",
                article_text="Fallback result",
                search_query=query,
            )
        ]


class ServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AppConfig(
            model=ModelConfig(
                backend="heuristic",
                api_key_env="",
                question_analysis_model_id="heuristic",
                query_planning_model_id="heuristic",
                candidate_filter_model_id="heuristic",
                article_selection_model_id="heuristic",
                metric_extraction_model_id="heuristic",
                summary_model_id="heuristic",
                max_output_tokens=0,
                temperature=0.0,
            ),
            search=SearchConfig(
                provider="google_news_rss",
                days_back=90,
                max_sources=5,
                max_search_calls_per_run=1,
            ),
            outlets=[
                OutletConfig(
                    name="Example",
                    domain="example.com",
                    country="France",
                    medium_type="newspaper",
                    orientation="center",
                    notes="test",
                ),
                OutletConfig(
                    name="Second Example",
                    domain="second.example.com",
                    country="United States",
                    medium_type="newspaper",
                    orientation="center",
                    notes="test",
                )
            ],
            config_path=Path("config/news_agent_free.yaml"),
        )

    def test_research_service_delegates_to_one_search_client(self) -> None:
        service = ResearchService(client=FakeSearchClient())

        bundle = service.research("query")

        self.assertEqual(bundle.query, "query")
        self.assertEqual(len(bundle.articles), 1)
        self.assertEqual(bundle.articles[0].outlet_name, "Example")

    def test_research_service_runs_intent_plan_and_metric_steps(self) -> None:
        client = FakeSearchClient()
        service = ResearchService(
            client=client,
            question_analyzer=FakeQuestionAnalyzer(),
            query_planner=FakeQueryPlanner(),
            metric_extractor=FakeMetricExtractor(),
        )

        bundle = service.research("How many AI laws were approved?")

        self.assertIsNotNone(client.received_intent)
        self.assertIsNotNone(client.received_plan)
        assert client.received_plan is not None
        self.assertEqual(client.received_plan.queries, ["AI regulation approved laws count"])
        self.assertTrue(bundle.articles[0].metric_found)
        self.assertEqual(bundle.articles[0].metric_value, "3 laws")

    def test_research_service_selects_articles_after_candidate_retrieval(self) -> None:
        article_selector = FakeArticleSelector(selected_index=1)
        client = FakeSearchClient(
            articles=[
                ArticleRecord(
                    title="First candidate",
                    url="https://example.com/first",
                    outlet_name="Example",
                    domain="example.com",
                    country="France",
                    medium_type="newspaper",
                    orientation="center",
                    published_at=None,
                    snippet="First snippet",
                    article_text="First article text",
                    search_query="query",
                ),
                ArticleRecord(
                    title="Second candidate",
                    url="https://example.com/second",
                    outlet_name="Example",
                    domain="example.com",
                    country="France",
                    medium_type="newspaper",
                    orientation="center",
                    published_at=None,
                    snippet="Second snippet",
                    article_text="Second article text",
                    search_query="query",
                ),
            ]
        )
        service = ResearchService(
            client=client,
            article_selector=article_selector,
            outlets=self.config.outlets,
            max_articles=self.config.search.max_sources,
        )

        bundle = service.research("Which article is best?")

        self.assertEqual([article.url for article in bundle.articles], [
            "https://example.com/second",
        ])
        self.assertEqual(len(article_selector.calls), 1)
        self.assertEqual(article_selector.calls[0][1], "Example")

    def test_free_config_builds_rss_search_client(self) -> None:
        client = build_search_client(self.config)

        self.assertIsInstance(client, GoogleNewsRssSearchClient)

    def test_freenewsapi_returns_candidates_without_provider_filtering(self) -> None:
        self.config.search.max_sources = 1
        client = FakeFreeNewsApiSearchClient(self.config)

        articles = client.search_candidates(
            "How many AI laws were approved?",
            intent=FakeQuestionAnalyzer().analyze("How many AI laws were approved?"),
        )

        self.assertFalse(hasattr(client, "candidate_filter"))
        self.assertEqual(len(articles), 2)
        self.assertEqual([article.url for article in articles], [
            "https://publisher.example.com/1",
            "https://publisher.example.com/2",
        ])

    def test_rss_search_attempts_each_outlet_and_fills_missing_from_curated(self) -> None:
        client = FakeRssSearchClient(
            config=self.config,
        )

        articles = client.search_candidates(
            "What are the latest verified updates on AI regulation?"
        )

        self.assertEqual(client.direct_calls, ["Example", "Second Example"])
        self.assertEqual([article.outlet_name for article in articles], [
            "Example",
            "Second Example",
        ])
        self.assertEqual(articles[0].url, "https://example.com/direct")
        self.assertEqual(
            articles[1].url,
            "https://second.example.com/fallback",
        )

    def test_rss_search_outlet_returns_prefiltered_candidates(self) -> None:
        class CandidateRssSearchClient(GoogleNewsRssSearchClient):
            def _search_query(
                self,
                scoped_query: str,
                outlet: OutletConfig,
                relevance_query: str,
            ) -> list[ArticleRecord]:
                return [
                    ArticleRecord(
                        title="Older partial article",
                        url="https://example.com/old",
                        outlet_name=outlet.name,
                        domain=outlet.domain,
                        country=outlet.country,
                        medium_type=outlet.medium_type,
                        orientation=outlet.orientation,
                        published_at=(datetime.now().astimezone() - timedelta(days=20)).isoformat(),
                        snippet="Short note",
                        article_text="Short note",
                        search_query=scoped_query,
                    ),
                    ArticleRecord(
                        title="More relevant recent article",
                        url="https://example.com/new",
                        outlet_name=outlet.name,
                        domain=outlet.domain,
                        country=outlet.country,
                        medium_type=outlet.medium_type,
                        orientation=outlet.orientation,
                        published_at=datetime.now().astimezone().isoformat(),
                        snippet="Detailed note about AI regulation",
                        article_text="Detailed note about AI regulation",
                        search_query=scoped_query,
                    ),
                ]

            def _prepare_candidates(
                self,
                candidates: list[ArticleRecord],
            ) -> list[ArticleRecord]:
                return candidates

        client = CandidateRssSearchClient(
            config=self.config,
        )

        result = client.search_outlet(
            "What are the latest verified updates on global AI regulation?",
            self.config.outlets[0],
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].url, "https://example.com/new")

    def test_rss_article_from_item_skips_empty_title_and_snippet(self) -> None:
        client = GoogleNewsRssSearchClient(
            config=self.config,
        )
        import xml.etree.ElementTree as ET
        from datetime import datetime
        from datetime import timedelta

        item = ET.fromstring(
            "<item><title></title><description></description><link>https://example.com/a</link></item>"
        )
        cutoff = datetime.now().astimezone() - timedelta(days=1)
        article = client._article_from_item(
            item=item,
            outlet=self.config.outlets[0],
            search_query="query",
            relevance_query="query",
            cutoff=cutoff,
        )
        self.assertIsNone(article)

    def test_openai_url_cleaner_accepts_markdown_links(self) -> None:
        url = OpenAIArticleNormalizer().clean_article_url(
            "[https://www.reuters.com/world/example](https://www.reuters.com/world/example)"
        )

        self.assertEqual(url, "https://www.reuters.com/world/example")

    def test_openai_search_jobs_add_site_filters_and_prefer_keyword_query(self) -> None:
        jobs = OpenAISearchJobPlanner().build_jobs(
            query="What are the latest casualty figures?",
            plan=SearchPlan(
                queries=[
                    "What are the latest casualty figures?",
                    "Iran USA conflict latest casualty figures",
                ]
            ),
            outlets=self.config.outlets,
            max_calls=1,
        )

        self.assertEqual(len(jobs), 1)
        self.assertTrue(
            jobs[0].search_query.startswith("Iran USA conflict latest casualty figures")
        )
        self.assertIn("site:example.com", jobs[0].search_query)
        self.assertIn("site:second.example.com", jobs[0].search_query)

    def test_openai_search_jobs_split_outlets_by_call_budget(self) -> None:
        outlets = [
            OutletConfig(
                name=f"Outlet {index}",
                domain=f"outlet{index}.example.com",
                country="Test",
                medium_type="newspaper",
                orientation="center",
                notes="test",
            )
            for index in range(6)
        ]

        jobs = OpenAISearchJobPlanner().build_jobs(
            query="What are the latest casualty figures?",
            plan=SearchPlan(queries=["Iran USA conflict latest casualty figures"]),
            outlets=outlets,
            max_calls=3,
        )

        self.assertEqual(len(jobs), 3)
        self.assertEqual([len(job.outlets) for job in jobs], [6, 3, 3])
        self.assertIn("site:outlet0.example.com", jobs[0].search_query)
        self.assertIn("site:outlet5.example.com", jobs[2].search_query)

    def test_openai_search_jobs_use_broad_plus_one_job_per_outlet_when_budget_allows(self) -> None:
        outlets = [
            OutletConfig(
                name=f"Outlet {index}",
                domain=f"outlet{index}.example.com",
                country="Test",
                medium_type="newspaper",
                orientation="center",
                notes="test",
            )
            for index in range(6)
        ]

        jobs = OpenAISearchJobPlanner().build_jobs(
            query="What are the latest casualty figures?",
            plan=SearchPlan(queries=["Iran USA conflict latest casualty figures"]),
            outlets=outlets,
            max_calls=7,
        )

        self.assertEqual(len(jobs), 7)
        self.assertEqual([len(job.outlets) for job in jobs], [6, 1, 1, 1, 1, 1, 1])

    def test_openai_normalizer_rejects_wrong_domain(self) -> None:
        articles = OpenAIArticleNormalizer().normalize(
            [
                {
                    "title": "Wrong outlet",
                    "url": "https://cnn.com/world/example",
                    "outlet_name": "Reuters",
                    "domain": "cnn.com",
                }
            ],
            allowed_outlets=(self.config.outlets[0],),
        )

        self.assertEqual(articles, [])

    def test_openai_normalizer_preserves_retrieval_metadata(self) -> None:
        articles = OpenAIArticleNormalizer().normalize(
            [
                {
                    "title": "Relevant result",
                    "url": "https://www.example.com/world/example",
                    "outlet_name": "Example",
                    "domain": "example.com",
                    "published_at": "2026-05-01",
                    "snippet": "Snippet",
                    "article_text": "Evidence sentence.",
                    "search_query": "query",
                    "answer_match_score": 5,
                    "evidence_match_score": 4,
                    "selection_reason": "Direct metric evidence.",
                }
            ],
            allowed_outlets=(self.config.outlets[0],),
        )

        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0].url, "https://www.example.com/world/example")
        self.assertEqual(articles[0].retrieval_metadata["answer_match_score"], 5)
        self.assertEqual(
            articles[0].retrieval_metadata["selection_reason"],
            "Direct metric evidence.",
        )

    def test_openai_debug_gateway_writes_request_and_response_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            inner = FakeOpenAIWebSearchGateway()
            gateway = DebuggingOpenAIWebSearchGateway(
                inner,
                DebugOutput(Path(tmp_dir)),
            )

            response = gateway.search(
                OpenAIWebSearchRequest(
                    call_name="openai_web_search_01",
                    prompt="Find the latest update.",
                    search_query="latest update site:example.com",
                    outlet_names=("Example",),
                    model_id="gpt-5.4-mini",
                    max_output_tokens=256,
                    temperature=0.0,
                    reasoning_effort="low",
                    max_tool_calls=1,
                    text_verbosity="low",
                )
            )

            call_dirs = sorted((Path(tmp_dir) / "model_calls").iterdir())
            self.assertEqual(len(call_dirs), 1)
            self.assertIsNotNone(inner.request)
            self.assertEqual(
                response.raw_text,
                '[{"title":"Example","url":"https://example.com/a"}]',
            )
            self.assertTrue((call_dirs[0] / "input.txt").exists())
            self.assertTrue((call_dirs[0] / "output.txt").exists())
            self.assertTrue((call_dirs[0] / "search_job.json").exists())
            self.assertTrue((call_dirs[0] / "response.json").exists())
            search_job = json.loads(
                (call_dirs[0] / "search_job.json").read_text()
            )
            self.assertEqual(search_job["search_query"], "latest update site:example.com")
            self.assertEqual(search_job["outlets"], ["Example"])

    def test_openai_client_uses_injected_settings_for_request(self) -> None:
        gateway = FakeOpenAIWebSearchGateway()
        settings = OpenAIWebSearchSettings(
            api_key_env="UNUSED_IN_TEST",
            model_id="gpt-5.4-mini",
            max_output_tokens=321,
            temperature=0.3,
            reasoning_effort="high",
            max_tool_calls=4,
            text_verbosity="medium",
        )
        client = OpenAIWebSearchClient(
            config=self.config,
            settings=settings,
            gateway=gateway,
            prompt_builder=FakeOpenAIWebSearchPromptBuilder(),
        )

        articles = client.search_candidates("What changed?")

        self.assertFalse(hasattr(client, "_web_search_model_id"))
        self.assertFalse(hasattr(client, "_api_key_env"))
        self.assertEqual(len(articles), 1)
        self.assertIsNotNone(gateway.request)
        assert gateway.request is not None
        self.assertEqual(gateway.request.model_id, "gpt-5.4-mini")
        self.assertEqual(gateway.request.max_output_tokens, 321)
        self.assertEqual(gateway.request.temperature, 0.3)
        self.assertEqual(gateway.request.reasoning_effort, "high")
        self.assertEqual(gateway.request.max_tool_calls, 4)
        self.assertEqual(gateway.request.text_verbosity, "medium")

    def test_openai_raw_response_path_preserves_reasoning_effort(self) -> None:
        body = {
            "output": [
                {
                    "id": "rs_1",
                    "type": "reasoning",
                    "summary": [],
                    "status": "completed",
                },
                {
                    "id": "msg_1",
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": '[{"title":"Example","url":"https://example.com/a"}]',
                        }
                    ],
                },
            ]
        }
        client = FakeOpenAIClient(body)

        response = _create_openai_response(
            client,
            {
                "model": "gpt-5.4-mini",
                "input": "prompt",
                "tools": [{"type": "web_search"}],
                "reasoning": {"effort": "high"},
            },
        )

        self.assertEqual(response, body)
        raw_kwargs = client.responses.with_raw_response.create_kwargs
        self.assertIsNotNone(raw_kwargs)
        assert raw_kwargs is not None
        self.assertEqual(raw_kwargs["reasoning"], {"effort": "high"})
        self.assertEqual(
            _extract_openai_response_text(response),
            '[{"title":"Example","url":"https://example.com/a"}]',
        )

    def test_openai_incomplete_response_reports_token_usage(self) -> None:
        with self.assertRaisesRegex(
            ModelOutputError,
            "max_output_tokens.*reasoning_tokens=9709",
        ):
            _raise_for_incomplete_openai_response(
                {
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "usage": {
                        "output_tokens": 10000,
                        "output_tokens_details": {"reasoning_tokens": 9709},
                    },
                }
            )


if __name__ == "__main__":
    unittest.main()

