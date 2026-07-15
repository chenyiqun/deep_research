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
    WebSearchConfig,
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
    assert DEFAULT_SEARCH_ENGINE == "search_live"
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
    asyncio.run(assert_visit_service_rejects_unsafe_urls())
    print("smoke_test_url_fetcher passed")


if __name__ == "__main__":
    main()
