from __future__ import annotations

from datetime import datetime
from datetime import timedelta
from pathlib import Path
import unittest

from news_agent.models.config import AppConfig
from news_agent.models.config import ModelConfig
from news_agent.models.config import OutletConfig
from news_agent.models.config import SearchConfig
from news_agent.models.research import ResearchIntent
from news_agent.models.research import SearchPlan
from news_agent.models.triage import ArticleRecord
from news_agent.models.triage import ResearchBundle
from news_agent.services.research import ResearchService
from news_agent.services.search import build_search_client
from news_agent.services.search.openai import _build_search_jobs
from news_agent.services.search.openai import _clean_article_url
from news_agent.services.search.rss import GoogleNewsRssSearchClient


class FakeSearchClient:
    def __init__(self) -> None:
        self.received_plan: SearchPlan | None = None
        self.received_intent: ResearchIntent | None = None

    def search(
        self,
        query: str,
        plan: SearchPlan | None = None,
        intent: ResearchIntent | None = None,
    ) -> ResearchBundle:
        self.received_plan = plan
        self.received_intent = intent
        return ResearchBundle(
            query=query,
            articles=[
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
                    search_query=query,
                )
            ],
        )


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
                max_search_calls_per_run=0,
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

    def test_free_config_builds_rss_search_client(self) -> None:
        client = build_search_client(self.config)

        self.assertIsInstance(client, GoogleNewsRssSearchClient)

    def test_rss_search_attempts_each_outlet_and_fills_missing_from_curated(self) -> None:
        client = FakeRssSearchClient(
            config=self.config,
        )

        bundle = client.search("What are the latest verified updates on AI regulation?")

        self.assertEqual(client.direct_calls, ["Example", "Second Example"])
        self.assertEqual([article.outlet_name for article in bundle.articles], [
            "Example",
            "Second Example",
        ])
        self.assertEqual(bundle.articles[0].url, "https://example.com/direct")
        self.assertEqual(
            bundle.articles[1].url,
            "https://second.example.com/fallback",
        )

    def test_rss_search_uses_article_selector(self) -> None:
        article_selector = FakeArticleSelector(selected_index=1)

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

        client = CandidateRssSearchClient(
            config=self.config,
        )
        client.article_selector = article_selector

        result = client.search_outlet(
            "What are the latest verified updates on global AI regulation?",
            self.config.outlets[0],
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].url, "https://example.com/new")
        self.assertEqual(len(article_selector.calls), 1)
        self.assertEqual(article_selector.calls[0][1], "Example")

    def test_rss_article_from_item_skips_empty_title_and_snippet(self) -> None:
        client = GoogleNewsRssSearchClient(
            config=self.config,
        )
        client.article_selector = FakeArticleSelector()
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
        url = _clean_article_url(
            "[https://www.reuters.com/world/example](https://www.reuters.com/world/example)"
        )

        self.assertEqual(url, "https://www.reuters.com/world/example")

    def test_openai_search_jobs_add_site_filters_and_prefer_keyword_query(self) -> None:
        jobs = _build_search_jobs(
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

        jobs = _build_search_jobs(
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

        jobs = _build_search_jobs(
            query="What are the latest casualty figures?",
            plan=SearchPlan(queries=["Iran USA conflict latest casualty figures"]),
            outlets=outlets,
            max_calls=7,
        )

        self.assertEqual(len(jobs), 7)
        self.assertEqual([len(job.outlets) for job in jobs], [6, 1, 1, 1, 1, 1, 1])


if __name__ == "__main__":
    unittest.main()
