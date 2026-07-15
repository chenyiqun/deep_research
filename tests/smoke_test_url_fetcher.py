from __future__ import annotations

import asyncio

from drb_qwen.url_fetcher import (
    URLFetchConfig,
    extract_text_from_html,
    normalize_visit_endpoint,
    select_relevant_excerpt,
    should_try_fetch_url,
)
from drb_qwen.visit_server import VisitService, is_probably_pdf
from drb_qwen.web_search import (
    DEFAULT_SEARCH_ENGINE,
    SUPPORTED_SEARCH_ENGINES,
    WebSearchClient,
    WebSearchConfig,
    get_search_engine_profile,
    is_unknown_search_engine_error,
    parse_search_results,
    should_fetch_result_pages,
)


async def assert_visit_service_rejects_unsafe_urls() -> None:
    service = VisitService(URLFetchConfig())
    service.fetcher = object()  # type: ignore[assignment]
    try:
        await service.visit("http://127.0.0.1/private", "test")
    except ValueError as exc:
        assert "unsafe URL" in str(exc)
    else:
        raise AssertionError("visit service must validate URLs before selecting a fetch backend")


class FakeSearchResponse:
    def __init__(self, status: int, body: str, data: dict | None = None) -> None:
        self.status = status
        self.body = body
        self.data = data or {}

    async def __aenter__(self) -> "FakeSearchResponse":
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    async def text(self) -> str:
        return self.body

    async def json(self) -> dict:
        return self.data


class FakeSearchSession:
    def __init__(self) -> None:
        self.api_engines: list[str] = []

    def post(self, endpoint: str, json: dict, headers: dict) -> FakeSearchResponse:
        api_engine = str(json["search_engine"])
        self.api_engines.append(api_engine)
        if api_engine == "search_pro_sogou":
            return FakeSearchResponse(
                400,
                '{"error":{"code":"1211","message":"模型不存在，请检查模型代码。"}}',
            )
        return FakeSearchResponse(
            200,
            "{}",
            {
                "search_result": [
                    {
                        "title": "legacy alias result",
                        "content": "direct search content",
                        "link": "https://example.com/result",
                    }
                ]
            },
        )


async def assert_search_engine_alias_fallback() -> None:
    client = WebSearchClient(
        WebSearchConfig(access_key="test", search_engine="search_live", count=10, max_retries=1)
    )
    session = FakeSearchSession()
    client._session = session
    results = await client.search("test", top_k=1)
    assert session.api_engines == ["search_pro_sogou", "search_live"]
    assert len(results) == 1
    assert results[0].search_engine == "search_live"
    assert results[0].api_search_engine == "search_live"


def main() -> None:
    html = """
    <html>
      <head><title>Ignored title</title><style>.x{color:red}</style></head>
      <body>
        <script>window.secret = "bad";</script>
        <article>
          <h1>Research title</h1>
          <p>First paragraph with useful evidence.</p>
          <p>Second paragraph with 2026 data.</p>
        </article>
      </body>
    </html>
    """
    text, method = extract_text_from_html(html)
    assert "Research title" in text
    assert "First paragraph with useful evidence." in text
    assert "Second paragraph with 2026 data." in text
    assert "window.secret" not in text
    assert method.startswith("html_")
    long_text = "\n".join(
        [
            "opening unrelated paragraph " * 30,
            "中国保险公司 信用评级 分红 ROE 这些是关键数据。",
            "closing unrelated paragraph " * 30,
        ]
    )
    excerpt = select_relevant_excerpt(long_text, "保险公司 信用评级 ROE", max_chars=120)
    assert "信用评级" in excerpt
    assert should_try_fetch_url("https://example.com/a")
    assert not should_try_fetch_url("ftp://example.com/a")
    assert not should_try_fetch_url("http://127.0.0.1/private")
    assert not should_try_fetch_url("http://169.254.169.254/latest/meta-data")
    assert not should_try_fetch_url("https://user:password@example.com/private")
    try:
        URLFetchConfig(max_concurrent_requests=0)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid fetch concurrency must fail before creating a deadlocked semaphore")
    assert is_probably_pdf("https://example.com/report.pdf")
    assert is_probably_pdf("https://example.com/download?id=1", "application/pdf")
    assert not is_probably_pdf("https://example.com/article.html", "text/html")
    assert normalize_visit_endpoint("http://localhost:8765") == "http://localhost:8765/visit"
    assert normalize_visit_endpoint("http://localhost:8765/visit") == "http://localhost:8765/visit"
    assert DEFAULT_SEARCH_ENGINE == "search_prime"
    assert set(SUPPORTED_SEARCH_ENGINES) == {
        "search_pro_jina",
        "search_prime",
        "search_pro_ms",
        "search_live",
        "search_lite",
        "search_plus",
    }
    for engine in SUPPORTED_SEARCH_ENGINES:
        assert WebSearchConfig(search_engine=engine).search_engine == engine
    assert should_fetch_result_pages("search_live", "auto") is False
    assert should_fetch_result_pages("search_prime", "auto") is True
    assert should_fetch_result_pages("search_live", "always") is True
    assert should_fetch_result_pages("search_prime", "never") is False
    assert get_search_engine_profile("search_live").api_engines == (
        "search_pro_sogou",
        "search_live",
    )
    assert get_search_engine_profile("search_lite").api_engines == (
        "search_pro_quark",
        "search_lite",
    )
    assert is_unknown_search_engine_error(
        400,
        '{"error":{"code":"1211","message":"模型不存在，请检查模型代码。"}}',
    )
    parsed = parse_search_results(
        {
            "data": {
                "results": [
                    {
                        "name": "Sogou result",
                        "summary": "Reader-ready result content.",
                        "url": "https://example.com/sogou",
                        "published_at": "2026-07-16",
                    }
                ]
            }
        },
        "test query",
        search_engine="search_live",
    )
    assert len(parsed) == 1
    assert parsed[0].search_engine == "search_live"
    assert parsed[0].api_search_engine == "search_pro_sogou"
    assert parsed[0].content_kind == "native_content"
    assert parsed[0].source_quality == "search_native_content"
    assert parsed[0].extraction_method == "search_live_content"
    assert parsed[0].link == "https://example.com/sogou"
    try:
        WebSearchConfig(search_engine="unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown search engines must fail before an API request")
    try:
        WebSearchConfig(search_engine="search_live", count=15)
    except ValueError:
        pass
    else:
        raise AssertionError("Sogou count must use the API-supported 10-result increments")
    asyncio.run(assert_visit_service_rejects_unsafe_urls())
    asyncio.run(assert_search_engine_alias_fallback())
    print("smoke_test_url_fetcher passed")


if __name__ == "__main__":
    main()
