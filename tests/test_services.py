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
from news_agent.models.generation import GenerationResult
from news_agent.models.research import ResearchIntent
from news_agent.models.research import SearchPlan
from news_agent.models.triage import ArticleRecord
from news_agent.models.triage import ResearchBundle
from news_agent.configuration.settings import OpenAIWebSearchSettings
from news_agent.services.articles.article_deduplicator import ArticleDeduplicator
from news_agent.services.research import ResearchService
from news_agent.services.debug.debug_output import DebugOutput
from news_agent.services.prompts.prompt_service import PromptService
from news_agent.services.research.pipeline import ResearchPipeline
from news_agent.services.research.steps import AnalyzeQuestionStep
from news_agent.services.research.steps import ApplyAnswerPolicyStep
from news_agent.services.research.steps import BuildResearchBundleStep
from news_agent.services.research.steps import EnrichCandidatesStep
from news_agent.services.research.steps import ExtractMetricsStep
from news_agent.services.research.steps import PlanQueriesStep
from news_agent.services.research.steps import RetrieveCandidatesStep
from news_agent.services.research.steps import SelectArticlesStep
from news_agent.services.search import build_search_client
from news_agent.services.search.free_news_api import FreeNewsApiSearchClient
from news_agent.services.search.openai import OpenAIWebSearchClient
from news_agent.services.search.openai.adaptive_react import AdaptiveReactRepairPlanner
from news_agent.services.search.openai.adaptive_react import RepairDecision
from news_agent.services.search.openai.article_normalizer import OpenAIArticleNormalizer
from news_agent.services.search.openai.gateway import DebuggingOpenAIWebSearchGateway
from news_agent.services.search.openai.gateway import OpenAIWebSearchGateway
from news_agent.services.search.openai.gateway import OpenAIWebSearchRequest
from news_agent.services.search.openai.gateway import OpenAIWebSearchResponse
from news_agent.services.search.openai.gateway import _create_openai_response
from news_agent.services.search.openai.gateway import _extract_openai_response_text
from news_agent.services.search.openai.gateway import _raise_for_incomplete_openai_response
from news_agent.services.search.openai.domain_utils import normalize_allowed_domain
from news_agent.services.search.openai.job_planner import OpenAISearchJobPlanner
from news_agent.services.search.openai.job_planner import WebSearchJob
from news_agent.services.search.openai.prompt_builder import OpenAIWebSearchPromptBuilder
from news_agent.services.search.rss import GoogleNewsRssSearchClient
from news_agent.services.llm.text_generation import ModelOutputError
from news_agent.services.llm.text_generation import StaticTextGenerator


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


def build_test_research_service(
    *,
    client: FakeSearchClient,
    question_analyzer: FakeQuestionAnalyzer | None = None,
    query_planner: FakeQueryPlanner | None = None,
    article_selector: FakeArticleSelector | None = None,
    metric_extractor: FakeMetricExtractor | None = None,
    outlets: list[OutletConfig] | None = None,
    max_articles: int | None = None,
) -> ResearchService:
    pipeline = ResearchPipeline(
        steps=[
            AnalyzeQuestionStep(question_analyzer),
            PlanQueriesStep(query_planner),
            RetrieveCandidatesStep(client),
            EnrichCandidatesStep(None),
            SelectArticlesStep(
                article_selector=article_selector,
                outlets=outlets or [],
                max_articles=max_articles,
            ),
            BuildResearchBundleStep(),
            ExtractMetricsStep(metric_extractor),
            ApplyAnswerPolicyStep(),
        ]
    )
    return ResearchService(pipeline=pipeline)


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


class SequencedFakeOpenAIWebSearchGateway:
    def __init__(self, responses: list[OpenAIWebSearchResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[OpenAIWebSearchRequest] = []

    def search(self, request: OpenAIWebSearchRequest) -> OpenAIWebSearchResponse:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("No fake OpenAI web-search response remains.")
        return self.responses.pop(0)


class SequencedTextGenerator:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> GenerationResult:
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("No fake text-generation response remains.")
        return GenerationResult(text=self.responses.pop(0))


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
        service = build_test_research_service(client=FakeSearchClient())

        bundle = service.research("query")

        self.assertEqual(bundle.query, "query")
        self.assertEqual(len(bundle.articles), 1)
        self.assertEqual(bundle.articles[0].outlet_name, "Example")

    def test_research_service_runs_intent_plan_and_metric_steps(self) -> None:
        client = FakeSearchClient()
        service = build_test_research_service(
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
        service = build_test_research_service(
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
        client = build_search_client(
            self.config,
            prompt_service=PromptService(),
        )

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

    def test_normalize_allowed_domain_removes_https(self) -> None:
        self.assertEqual(
            normalize_allowed_domain("https://www.reuters.com/world/"),
            "www.reuters.com",
        )

    def test_normalize_allowed_domain_removes_path(self) -> None:
        self.assertEqual(
            normalize_allowed_domain("https://cnn.com/world/latest"),
            "cnn.com",
        )

    def test_normalize_allowed_domain_keeps_root_domain(self) -> None:
        self.assertEqual(normalize_allowed_domain("reuters.com"), "reuters.com")

    def test_job_planner_uses_allowed_domains_without_site_filters(self) -> None:
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
        self.assertEqual(
            jobs[0].allowed_domains,
            ("example.com", "second.example.com"),
        )
        self.assertTrue(
            jobs[0].search_query.startswith("Iran USA conflict latest casualty figures")
        )
        self.assertNotIn("site:example.com", jobs[0].search_query)

    def test_job_planner_can_keep_legacy_site_filters_when_enabled(self) -> None:
        jobs = OpenAISearchJobPlanner().build_jobs(
            query="What are the latest casualty figures?",
            plan=SearchPlan(queries=["Iran USA conflict latest casualty figures"]),
            outlets=self.config.outlets,
            max_calls=1,
            use_allowed_domains=False,
            use_site_query_filters=True,
        )

        self.assertEqual(jobs[0].allowed_domains, ())
        self.assertIn("site:example.com", jobs[0].search_query)
        self.assertIn("site:second.example.com", jobs[0].search_query)

    def test_job_planner_can_use_both_allowed_domains_and_site_filters_for_experiment(self) -> None:
        jobs = OpenAISearchJobPlanner().build_jobs(
            query="What are the latest casualty figures?",
            plan=SearchPlan(queries=["Iran USA conflict latest casualty figures"]),
            outlets=self.config.outlets,
            max_calls=1,
            use_allowed_domains=True,
            use_site_query_filters=True,
        )

        self.assertEqual(
            jobs[0].allowed_domains,
            ("example.com", "second.example.com"),
        )
        self.assertIn("site:example.com", jobs[0].search_query)

    def test_prompt_builder_for_allowed_domain_job_forbids_site_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            prompt_path = Path(tmp_dir) / "test_prompt.txt"
            prompt_path.write_text(
                "Prompt {planned_queries_json} {query} {outlets_text}",
                encoding="utf-8",
            )
            prompt = OpenAIWebSearchPromptBuilder(
                PromptService(prompts_dir=Path(tmp_dir))
            ).build(
                template_name="test_prompt",
                query="What changed?",
                job=WebSearchJob(
                    search_query="What changed?",
                    outlets=(self.config.outlets[0],),
                    allowed_domains=("example.com",),
                ),
                days_back=7,
                intent=None,
            )

        self.assertIn("allowed_domains", prompt)
        self.assertIn("Do not add `site:` filters", prompt)
        self.assertNotIn("Keep any existing `site:<domain>`", prompt)

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
        self.assertNotIn("site:outlet0.example.com", jobs[0].search_query)
        self.assertEqual(
            jobs[0].allowed_domains,
            tuple(f"outlet{index}.example.com" for index in range(6)),
        )
        self.assertEqual(
            jobs[1].allowed_domains,
            tuple(f"outlet{index}.example.com" for index in range(3)),
        )
        self.assertEqual(
            jobs[2].allowed_domains,
            tuple(f"outlet{index}.example.com" for index in range(3, 6)),
        )

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
                    search_query="latest update",
                    outlet_names=("Example",),
                    model_id="gpt-5.4-mini",
                    max_output_tokens=256,
                    temperature=0.0,
                    reasoning_effort="low",
                    max_tool_calls=1,
                    text_verbosity="low",
                    allowed_domains=("example.com",),
                    include_sources=True,
                    tool_choice="required",
                    search_context_size="medium",
                    use_site_query_filters=False,
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
            self.assertTrue((call_dirs[0] / "internal_web_search_calls.json").exists())
            self.assertTrue((call_dirs[0] / "web_search_sources.json").exists())
            search_job = json.loads(
                (call_dirs[0] / "search_job.json").read_text()
            )
            self.assertEqual(search_job["search_query"], "latest update")
            self.assertEqual(search_job["outlets"], ["Example"])
            self.assertEqual(search_job["allowed_domains"], ["example.com"])
            self.assertTrue(search_job["include_sources"])
            self.assertEqual(search_job["tool_choice"], "required")
            self.assertEqual(search_job["search_context_size"], "medium")
            self.assertFalse(search_job["use_site_query_filters"])

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
            prompt_service=PromptService(),
            job_planner=OpenAISearchJobPlanner(),
            gateway=gateway,
            prompt_builder=FakeOpenAIWebSearchPromptBuilder(),
            normalizer=OpenAIArticleNormalizer(),
            deduplicator=ArticleDeduplicator(),
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
        self.assertEqual(
            gateway.request.allowed_domains,
            ("example.com", "second.example.com"),
        )
        self.assertTrue(gateway.request.include_sources)
        self.assertEqual(gateway.request.tool_choice, "required")
        self.assertEqual(gateway.request.search_context_size, "medium")
        self.assertFalse(gateway.request.use_site_query_filters)
        self.assertNotIn("site:example.com", gateway.request.search_query)

    def test_adaptive_observation_records_raw_facts_only(self) -> None:
        planner = self._adaptive_planner()

        observation = planner.build_observation(
            articles=[
                self._article("Example", "https://example.com/first"),
                self._article("Example", "https://example.com/second"),
            ],
            outlets=self.config.outlets,
            remaining_repair_actions=2,
        )

        self.assertEqual(observation.candidate_count, 2)
        self.assertEqual(observation.outlet_counts, {"Example": 2})
        self.assertEqual(observation.distinct_outlet_count, 1)
        self.assertEqual(observation.configured_outlets[0]["name"], "Example")
        self.assertEqual(observation.configured_outlets[0]["notes"], "test")
        self.assertEqual(len(observation.top_candidates), 2)
        self.assertEqual(observation.top_candidates[0]["snippet"], "Snippet")
        self.assertEqual(observation.top_candidates[0]["article_text"], "Article text")
        self.assertEqual(observation.previous_actions, [])
        self.assertEqual(observation.remaining_repair_actions, 2)
        self.assertNotIn("missing_configured_outlets", observation.to_dict())
        self.assertNotIn("dominant_outlet", observation.to_dict())

    def test_adaptive_decision_preserves_llm_diagnosis(self) -> None:
        planner = self._adaptive_planner(
            text=json.dumps(
                {
                    "diagnosis": {
                        "quality": "partial",
                        "is_outlet_diverse": True,
                        "is_answer_bearing": False,
                        "dominance_assessment": "No outlet dominates.",
                        "strong_candidates": ["Example result"],
                        "weak_candidates": ["Second result"],
                        "missing_useful_outlets": ["Second Example"],
                        "problem": "Needs a stronger second source.",
                    },
                    "action": "search",
                    "reason": "Target the missing useful outlet.",
                    "search_query": "focused repair query",
                    "allowed_outlets": ["Second Example"],
                }
            )
        )
        observation = planner.build_observation(
            articles=[self._article("Example", "https://example.com/first")],
            outlets=self.config.outlets,
        )

        decision = planner.decide(
            query="What changed?",
            plan=SearchPlan(queries=["global query"]),
            intent=None,
            observation=observation,
            outlets=self.config.outlets,
        )

        self.assertEqual(decision.action, "search")
        self.assertEqual(decision.diagnosis["quality"], "partial")
        self.assertEqual(decision.diagnosis["missing_useful_outlets"], ["Second Example"])
        self.assertEqual(decision.allowed_outlets, ["Second Example"])

    def test_adaptive_decision_finish_skips_repair(self) -> None:
        gateway = SequencedFakeOpenAIWebSearchGateway(
            [
                self._openai_response(
                    [
                        self._candidate_payload(
                            title="First result",
                            url="https://example.com/first",
                            search_query="global query",
                        )
                    ]
                )
            ]
        )
        client = self._openai_client(
            gateway=gateway,
            adaptive_enabled=True,
            repair_text=json.dumps(
                {
                    "action": "finish",
                    "reason": "The first result is good enough.",
                    "search_query": "",
                    "allowed_outlets": [],
                }
            ),
        )

        articles = client.search_candidates(
            "What changed?",
            plan=SearchPlan(queries=["global query"]),
        )

        self.assertEqual(len(articles), 1)
        self.assertEqual(len(gateway.requests), 1)

    def test_adaptive_decision_search_runs_one_repair(self) -> None:
        gateway = SequencedFakeOpenAIWebSearchGateway(
            [
                self._openai_response(
                    [
                        self._candidate_payload(
                            title="First result",
                            url="https://example.com/first",
                            search_query="global query",
                        )
                    ]
                ),
                self._openai_response(
                    [
                        self._candidate_payload(
                            title="Repair result",
                            url="https://second.example.com/repair",
                            outlet_name="Second Example",
                            domain="second.example.com",
                            search_query="focused repair query",
                        )
                    ]
                ),
            ]
        )
        client = self._openai_client(
            gateway=gateway,
            adaptive_enabled=True,
            repair_texts=[
                json.dumps(
                    {
                        "action": "search",
                        "reason": "Second Example is missing.",
                        "search_query": "focused repair query",
                        "allowed_outlets": ["Second Example"],
                    }
                ),
                json.dumps(
                    {
                        "action": "finish",
                        "reason": "The repair added enough diversity.",
                        "search_query": "",
                        "allowed_outlets": [],
                    }
                ),
            ],
        )

        articles = client.search_candidates(
            "What changed?",
            plan=SearchPlan(queries=["global query"]),
        )

        self.assertEqual(len(gateway.requests), 2)
        self.assertEqual(gateway.requests[1].search_query, "focused repair query")
        self.assertEqual(gateway.requests[1].max_tool_calls, 2)
        self.assertEqual(
            [article.outlet_name for article in articles],
            ["Example", "Second Example"],
        )

    def test_adaptive_rejects_unknown_outlets(self) -> None:
        repair_job = self._adaptive_planner().build_repair_job(
            decision=RepairDecision(
                action="search",
                reason="Need a focused repair.",
                search_query="focused query",
                allowed_outlets=["Example", "Unknown Outlet"],
            ),
            outlets=self.config.outlets,
        )

        self.assertIsNotNone(repair_job)
        assert repair_job is not None
        self.assertEqual([outlet.name for outlet in repair_job.outlets], ["Example"])

    def test_adaptive_decision_finishes_when_no_allowed_outlets_are_valid(self) -> None:
        planner = self._adaptive_planner(
            text=json.dumps(
                {
                    "diagnosis": {
                        "quality": "partial",
                        "is_outlet_diverse": False,
                        "is_answer_bearing": False,
                        "dominance_assessment": "The set is weak.",
                        "strong_candidates": [],
                        "weak_candidates": ["Example result"],
                        "missing_useful_outlets": ["Unknown Outlet"],
                        "problem": "The requested repair outlet is not configured.",
                    },
                    "action": "search",
                    "reason": "Try an unknown outlet.",
                    "search_query": "focused repair query",
                    "allowed_outlets": ["Unknown Outlet"],
                }
            )
        )
        observation = planner.build_observation(
            articles=[self._article("Example", "https://example.com/first")],
            outlets=self.config.outlets,
        )

        decision = planner.decide(
            query="What changed?",
            plan=None,
            intent=None,
            observation=observation,
            outlets=self.config.outlets,
        )

        self.assertEqual(decision.action, "finish")
        self.assertEqual(decision.allowed_outlets, [])
        self.assertEqual(decision.diagnosis["quality"], "partial")

    def test_adaptive_repair_job_uses_allowed_domains(self) -> None:
        repair_job = self._adaptive_planner().build_repair_job(
            decision=RepairDecision(
                action="search",
                reason="Need a focused repair.",
                search_query="focused query",
                allowed_outlets=["Example", "Second Example"],
            ),
            outlets=self.config.outlets,
        )

        self.assertIsNotNone(repair_job)
        assert repair_job is not None
        self.assertEqual(
            repair_job.allowed_domains,
            ("example.com", "second.example.com"),
        )
        self.assertNotIn("site:", repair_job.search_query)

    def test_adaptive_caps_candidates_per_outlet(self) -> None:
        self.config.search.adaptive_react_max_candidates_per_outlet = 2
        articles = self._adaptive_planner().cap_per_outlet(
            [
                self._article("Example", "https://example.com/1"),
                self._article("Example", "https://example.com/2"),
                self._article("Example", "https://example.com/3"),
                self._article("Second Example", "https://second.example.com/1"),
            ]
        )

        self.assertEqual(
            [article.url for article in articles],
            [
                "https://example.com/1",
                "https://example.com/2",
                "https://second.example.com/1",
            ],
        )

    def test_adaptive_writes_observation_decision_and_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            debug_output = DebugOutput(Path(tmp_dir))
            gateway = SequencedFakeOpenAIWebSearchGateway(
                [
                    self._openai_response(
                        [
                            self._candidate_payload(
                                title="First result",
                                url="https://example.com/first",
                            )
                        ]
                    )
                ]
            )
            client = self._openai_client(
                gateway=gateway,
                adaptive_enabled=True,
                repair_text=json.dumps(
                    {
                        "action": "finish",
                        "reason": "The first search is good enough.",
                        "search_query": "",
                        "allowed_outlets": [],
                    }
                ),
                debug_output=debug_output,
            )

            client.search_candidates("What changed?")

            run_dir = Path(tmp_dir)
            self.assertTrue((run_dir / "adaptive_react_observation.json").exists())
            self.assertTrue((run_dir / "adaptive_react_decision.json").exists())
            trace = (run_dir / "adaptive_react_trace.txt").read_text()
            self.assertIn("Action 1: GlobalSearch", trace)
            self.assertIn("Reason 2:", trace)
            self.assertIn("Final:", trace)

    def test_non_adaptive_mode_keeps_existing_behavior(self) -> None:
        gateway = SequencedFakeOpenAIWebSearchGateway(
            [
                self._openai_response(
                    [
                        self._candidate_payload(
                            title="First result",
                            url="https://example.com/first",
                        )
                    ]
                )
            ]
        )
        client = self._openai_client(
            gateway=gateway,
            adaptive_enabled=False,
            repair_text="not json",
        )

        articles = client.search_candidates("What changed?")

        self.assertEqual(len(articles), 1)
        self.assertEqual(len(gateway.requests), 1)

    def test_repair_search_uses_max_tool_calls_override(self) -> None:
        gateway = SequencedFakeOpenAIWebSearchGateway(
            [
                self._openai_response(
                    [
                        self._candidate_payload(
                            title="First result",
                            url="https://example.com/first",
                        )
                    ]
                ),
                self._openai_response(
                    [
                        self._candidate_payload(
                            title="Repair result",
                            url="https://second.example.com/repair",
                            outlet_name="Second Example",
                            domain="second.example.com",
                        )
                    ]
                ),
            ]
        )
        client = self._openai_client(
            gateway=gateway,
            adaptive_enabled=True,
            repair_texts=[
                json.dumps(
                    {
                        "action": "search",
                        "reason": "Second Example is missing.",
                        "search_query": "focused repair query",
                        "allowed_outlets": ["Second Example"],
                    }
                ),
                json.dumps(
                    {
                        "action": "finish",
                        "reason": "The repair added enough diversity.",
                        "search_query": "",
                        "allowed_outlets": [],
                    }
                ),
            ],
        )

        client.search_candidates("What changed?")

        self.assertEqual(gateway.requests[0].max_tool_calls, 8)
        self.assertEqual(gateway.requests[1].max_tool_calls, 2)

    def test_adaptive_loop_can_run_two_repairs(self) -> None:
        gateway = SequencedFakeOpenAIWebSearchGateway(
            [
                self._openai_response(
                    [
                        self._candidate_payload(
                            title="First result",
                            url="https://example.com/first",
                        )
                    ]
                ),
                self._openai_response(
                    [
                        self._candidate_payload(
                            title="First repair",
                            url="https://second.example.com/repair",
                            outlet_name="Second Example",
                            domain="second.example.com",
                            search_query="first repair query",
                        )
                    ]
                ),
                self._openai_response(
                    [
                        self._candidate_payload(
                            title="Second repair",
                            url="https://example.com/second",
                            search_query="second repair query",
                        )
                    ]
                ),
            ]
        )
        client = self._openai_client(
            gateway=gateway,
            adaptive_enabled=True,
            repair_texts=[
                self._decision_text(
                    action="search",
                    reason="Second Example is missing.",
                    search_query="first repair query",
                    allowed_outlets=["Second Example"],
                ),
                self._decision_text(
                    action="search",
                    reason="Example needs another candidate.",
                    search_query="second repair query",
                    allowed_outlets=["Example"],
                ),
            ],
        )

        articles = client.search_candidates("What changed?")

        self.assertEqual(len(gateway.requests), 3)
        self.assertEqual(
            [request.search_query for request in gateway.requests],
            ["What changed?", "first repair query", "second repair query"],
        )
        self.assertEqual(len(articles), 3)

    def test_adaptive_loop_stops_when_planner_finishes(self) -> None:
        generator = SequencedTextGenerator(
            [
                self._decision_text(
                    action="search",
                    reason="Second Example is missing.",
                    search_query="first repair query",
                    allowed_outlets=["Second Example"],
                ),
                self._decision_text(
                    action="finish",
                    reason="The repair added enough diversity.",
                ),
            ]
        )
        gateway = SequencedFakeOpenAIWebSearchGateway(
            [
                self._openai_response(
                    [
                        self._candidate_payload(
                            title="First result",
                            url="https://example.com/first",
                        )
                    ]
                ),
                self._openai_response(
                    [
                        self._candidate_payload(
                            title="Repair result",
                            url="https://second.example.com/repair",
                            outlet_name="Second Example",
                            domain="second.example.com",
                        )
                    ]
                ),
            ]
        )
        client = self._openai_client(
            gateway=gateway,
            adaptive_enabled=True,
            repair_generator=generator,
        )

        client.search_candidates("What changed?")

        self.assertEqual(len(gateway.requests), 2)
        self.assertEqual(len(generator.prompts), 2)

    def test_adaptive_loop_stops_when_repair_adds_no_new_candidates(self) -> None:
        generator = SequencedTextGenerator(
            [
                self._decision_text(
                    action="search",
                    reason="Try the missing outlet.",
                    search_query="duplicate repair query",
                    allowed_outlets=["Example"],
                ),
                self._decision_text(
                    action="search",
                    reason="This should not be requested.",
                    search_query="unused query",
                    allowed_outlets=["Second Example"],
                ),
            ]
        )
        gateway = SequencedFakeOpenAIWebSearchGateway(
            [
                self._openai_response(
                    [
                        self._candidate_payload(
                            title="First result",
                            url="https://example.com/first",
                        )
                    ]
                ),
                self._openai_response(
                    [
                        self._candidate_payload(
                            title="Duplicate result",
                            url="https://example.com/first",
                        )
                    ]
                ),
            ]
        )
        client = self._openai_client(
            gateway=gateway,
            adaptive_enabled=True,
            repair_generator=generator,
        )

        articles = client.search_candidates("What changed?")

        self.assertEqual(len(articles), 1)
        self.assertEqual(len(gateway.requests), 2)
        self.assertEqual(len(generator.prompts), 1)

    def test_adaptive_loop_does_not_exceed_max_repair_actions(self) -> None:
        generator = SequencedTextGenerator(
            [
                self._decision_text(
                    action="search",
                    reason="First repair.",
                    search_query="first repair query",
                    allowed_outlets=["Second Example"],
                ),
                self._decision_text(
                    action="search",
                    reason="Second repair.",
                    search_query="second repair query",
                    allowed_outlets=["Example"],
                ),
                self._decision_text(
                    action="search",
                    reason="This should not be requested.",
                    search_query="third repair query",
                    allowed_outlets=["Example"],
                ),
            ]
        )
        gateway = SequencedFakeOpenAIWebSearchGateway(
            [
                self._openai_response(
                    [
                        self._candidate_payload(
                            title="First result",
                            url="https://example.com/first",
                        )
                    ]
                ),
                self._openai_response(
                    [
                        self._candidate_payload(
                            title="First repair",
                            url="https://second.example.com/repair",
                            outlet_name="Second Example",
                            domain="second.example.com",
                        )
                    ]
                ),
                self._openai_response(
                    [
                        self._candidate_payload(
                            title="Second repair",
                            url="https://example.com/second",
                        )
                    ]
                ),
            ]
        )
        client = self._openai_client(
            gateway=gateway,
            adaptive_enabled=True,
            repair_generator=generator,
        )

        client.search_candidates("What changed?")

        self.assertEqual(len(gateway.requests), 3)
        self.assertEqual(len(generator.prompts), 2)

    def test_adaptive_loop_passes_previous_actions_to_planner(self) -> None:
        generator = SequencedTextGenerator(
            [
                self._decision_text(
                    action="search",
                    reason="First repair.",
                    search_query="first repair query",
                    allowed_outlets=["Second Example"],
                ),
                self._decision_text(
                    action="finish",
                    reason="The previous repair is enough.",
                ),
            ]
        )
        gateway = SequencedFakeOpenAIWebSearchGateway(
            [
                self._openai_response(
                    [
                        self._candidate_payload(
                            title="First result",
                            url="https://example.com/first",
                        )
                    ]
                ),
                self._openai_response(
                    [
                        self._candidate_payload(
                            title="Repair result",
                            url="https://second.example.com/repair",
                            outlet_name="Second Example",
                            domain="second.example.com",
                        )
                    ]
                ),
            ]
        )
        client = self._openai_client(
            gateway=gateway,
            adaptive_enabled=True,
            repair_generator=generator,
        )

        client.search_candidates("What changed?")

        self.assertEqual(len(generator.prompts), 2)
        self.assertIn("Previous repair actions:", generator.prompts[1])
        self.assertIn("first repair query", generator.prompts[1])
        self.assertIn("Remaining repair actions:\n1", generator.prompts[1])

    def test_adaptive_loop_writes_multistep_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            debug_output = DebugOutput(Path(tmp_dir))
            gateway = SequencedFakeOpenAIWebSearchGateway(
                [
                    self._openai_response(
                        [
                            self._candidate_payload(
                                title="First result",
                                url="https://example.com/first",
                            )
                        ]
                    ),
                    self._openai_response(
                        [
                            self._candidate_payload(
                                title="First repair",
                                url="https://second.example.com/repair",
                                outlet_name="Second Example",
                                domain="second.example.com",
                            )
                        ]
                    ),
                    self._openai_response(
                        [
                            self._candidate_payload(
                                title="Second repair",
                                url="https://example.com/second",
                            )
                        ]
                    ),
                ]
            )
            client = self._openai_client(
                gateway=gateway,
                adaptive_enabled=True,
                repair_texts=[
                    self._decision_text(
                        action="search",
                        reason="First repair.",
                        search_query="first repair query",
                        allowed_outlets=["Second Example"],
                    ),
                    self._decision_text(
                        action="search",
                        reason="Second repair.",
                        search_query="second repair query",
                        allowed_outlets=["Example"],
                    ),
                ],
                debug_output=debug_output,
            )

            client.search_candidates("What changed?")

            trace = (Path(tmp_dir) / "adaptive_react_trace.txt").read_text()
            self.assertIn("Observation 1:", trace)
            self.assertIn("Reason 2:", trace)
            self.assertIn("Action 2: Search", trace)
            self.assertIn("Observation 2:", trace)
            self.assertIn("Reason 3:", trace)
            self.assertIn("Action 3: Search", trace)
            self.assertIn("Observation 3:", trace)
            self.assertIn("Final:", trace)

    def test_gateway_passes_allowed_domains_to_web_search_tool(self) -> None:
        gateway = object.__new__(OpenAIWebSearchGateway)
        body = _openai_completed_body()
        gateway.client = FakeOpenAIClient(body)

        gateway.search(
            OpenAIWebSearchRequest(
                call_name="call",
                prompt="prompt",
                search_query="query",
                outlet_names=("Example",),
                model_id="gpt-5.4-mini",
                max_output_tokens=256,
                temperature=0.0,
                allowed_domains=("https://www.example.com/world/",),
            )
        )

        raw_kwargs = gateway.client.responses.with_raw_response.create_kwargs
        self.assertIsNotNone(raw_kwargs)
        assert raw_kwargs is not None
        self.assertEqual(
            raw_kwargs["tools"][0]["filters"],
            {"allowed_domains": ["www.example.com"]},
        )

    def test_gateway_sets_search_context_size(self) -> None:
        gateway = object.__new__(OpenAIWebSearchGateway)
        gateway.client = FakeOpenAIClient(_openai_completed_body())

        gateway.search(
            OpenAIWebSearchRequest(
                call_name="call",
                prompt="prompt",
                search_query="query",
                outlet_names=("Example",),
                model_id="gpt-5.4-mini",
                max_output_tokens=256,
                temperature=0.0,
                search_context_size="high",
            )
        )

        raw_kwargs = gateway.client.responses.with_raw_response.create_kwargs
        assert raw_kwargs is not None
        self.assertEqual(raw_kwargs["tools"][0]["search_context_size"], "high")

    def test_gateway_includes_web_search_sources_when_enabled(self) -> None:
        gateway = object.__new__(OpenAIWebSearchGateway)
        gateway.client = FakeOpenAIClient(_openai_completed_body())

        gateway.search(
            OpenAIWebSearchRequest(
                call_name="call",
                prompt="prompt",
                search_query="query",
                outlet_names=("Example",),
                model_id="gpt-5.4-mini",
                max_output_tokens=256,
                temperature=0.0,
                include_sources=True,
            )
        )

        raw_kwargs = gateway.client.responses.with_raw_response.create_kwargs
        assert raw_kwargs is not None
        self.assertEqual(raw_kwargs["include"], ["web_search_call.action.sources"])

    def test_gateway_sets_tool_choice_required(self) -> None:
        gateway = object.__new__(OpenAIWebSearchGateway)
        gateway.client = FakeOpenAIClient(_openai_completed_body())

        gateway.search(
            OpenAIWebSearchRequest(
                call_name="call",
                prompt="prompt",
                search_query="query",
                outlet_names=("Example",),
                model_id="gpt-5.4-mini",
                max_output_tokens=256,
                temperature=0.0,
                tool_choice="required",
            )
        )

        raw_kwargs = gateway.client.responses.with_raw_response.create_kwargs
        assert raw_kwargs is not None
        self.assertEqual(raw_kwargs["tool_choice"], "required")

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

    def _adaptive_planner(
        self,
        *,
        text: str | None = None,
        debug_output: DebugOutput | None = None,
    ) -> AdaptiveReactRepairPlanner:
        return AdaptiveReactRepairPlanner(
            config=self.config,
            prompt_service=PromptService(),
            text_generator=StaticTextGenerator(text) if text is not None else None,
            debug_output=debug_output,
        )

    def _openai_client(
        self,
        *,
        gateway: SequencedFakeOpenAIWebSearchGateway,
        adaptive_enabled: bool,
        repair_text: str = "",
        repair_texts: list[str] | None = None,
        repair_generator: SequencedTextGenerator | None = None,
        debug_output: DebugOutput | None = None,
    ) -> OpenAIWebSearchClient:
        self.config.search.provider = "openai_web_search"
        self.config.search.max_sources = 2
        self.config.search.adaptive_react_enabled = adaptive_enabled
        self.config.search.adaptive_react_repair_max_tool_calls = 2
        self.config.search.adaptive_react_max_repair_actions = 2
        self.config.search.adaptive_react_max_candidates_per_outlet = 2
        repair_planner_generator = None
        if repair_generator is not None:
            repair_planner_generator = repair_generator
        elif repair_texts is not None:
            repair_planner_generator = SequencedTextGenerator(repair_texts)
        elif repair_text:
            repair_planner_generator = StaticTextGenerator(repair_text)
        settings = OpenAIWebSearchSettings(
            api_key_env="UNUSED_IN_TEST",
            model_id="gpt-5.4-mini",
            max_output_tokens=256,
            temperature=0.0,
            reasoning_effort="low",
            max_tool_calls=8,
            text_verbosity="low",
        )
        return OpenAIWebSearchClient(
            config=self.config,
            settings=settings,
            prompt_service=PromptService(),
            job_planner=OpenAISearchJobPlanner(),
            gateway=gateway,
            prompt_builder=FakeOpenAIWebSearchPromptBuilder(),
            normalizer=OpenAIArticleNormalizer(),
            deduplicator=ArticleDeduplicator(),
            repair_planner_generator=repair_planner_generator,
            debug_output=debug_output,
        )

    def _article(self, outlet_name: str, url: str) -> ArticleRecord:
        outlet = next(
            outlet for outlet in self.config.outlets if outlet.name == outlet_name
        )
        return ArticleRecord(
            title=f"{outlet_name} title",
            url=url,
            outlet_name=outlet.name,
            domain=outlet.domain,
            country=outlet.country,
            medium_type=outlet.medium_type,
            orientation=outlet.orientation,
            published_at="2026-05-01",
            snippet="Snippet",
            article_text="Article text",
            search_query="query",
        )

    def _candidate_payload(
        self,
        *,
        title: str,
        url: str,
        outlet_name: str = "Example",
        domain: str = "example.com",
        search_query: str = "query",
    ) -> dict[str, object]:
        return {
            "title": title,
            "url": url,
            "outlet_name": outlet_name,
            "domain": domain,
            "published_at": "2026-05-01",
            "snippet": "Snippet",
            "article_text": "Article text",
            "search_query": search_query,
        }

    def _decision_text(
        self,
        *,
        action: str,
        reason: str,
        search_query: str = "",
        allowed_outlets: list[str] | None = None,
    ) -> str:
        return json.dumps(
            {
                "action": action,
                "reason": reason,
                "search_query": search_query,
                "allowed_outlets": allowed_outlets or [],
            }
        )

    def _openai_response(
        self,
        candidates: list[dict[str, object]],
    ) -> OpenAIWebSearchResponse:
        return OpenAIWebSearchResponse(
            raw_text=json.dumps(candidates),
            response_dump='{"status":"completed"}',
        )


def _openai_completed_body() -> dict[str, object]:
    return {
        "status": "completed",
        "output": [
            {
                "id": "msg_1",
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "[]",
                    }
                ],
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
