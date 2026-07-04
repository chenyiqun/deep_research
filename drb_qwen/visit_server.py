from __future__ import annotations

import argparse
import asyncio
import os
from dataclasses import asdict
from typing import Any

from .url_fetcher import URLContentFetcher, URLFetchConfig, clean_text, select_relevant_excerpt


try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except Exception:  # pragma: no cover - import error is surfaced at runtime.
    FastAPI = None  # type: ignore[assignment]
    HTTPException = RuntimeError  # type: ignore[assignment]
    BaseModel = object  # type: ignore[assignment]


if FastAPI is None:
    app = None
else:
    app = FastAPI(title="DeepResearch Visit Service", version="1.0.0")


class VisitRequest(BaseModel):  # type: ignore[misc]
    url: str
    goal: str = ""


class VisitResponse(BaseModel):  # type: ignore[misc]
    url: str
    content: str
    ok: bool
    source: str
    extraction_method: str = ""
    status: int | None = None
    content_type: str = ""
    final_url: str = ""
    raw_text_chars: int = 0
    cached: bool = False
    error: str = ""


class VisitService:
    def __init__(
        self,
        fetch_config: URLFetchConfig,
        content_length: int = 12_000,
        min_content_chars: int = 500,
        enable_crawl4ai: bool = False,
        crawl4ai_timeout_s: int = 45,
    ) -> None:
        self.fetch_config = fetch_config
        self.content_length = content_length
        self.min_content_chars = min_content_chars
        self.enable_crawl4ai = enable_crawl4ai
        self.crawl4ai_timeout_s = crawl4ai_timeout_s
        self.fetcher: URLContentFetcher | None = None

    async def start(self) -> None:
        self.fetcher = await URLContentFetcher(self.fetch_config).__aenter__()

    async def close(self) -> None:
        if self.fetcher is not None:
            await self.fetcher.__aexit__(None, None, None)
        self.fetcher = None

    async def visit(self, url: str, goal: str) -> dict[str, Any]:
        if self.fetcher is None:
            raise RuntimeError("VisitService has not started.")

        direct = await self.fetcher.fetch(url, goal="")
        best = direct
        browser_error = ""

        should_try_browser = (
            self.enable_crawl4ai
            and not is_probably_pdf(url, direct.content_type)
            and (not direct.ok or len(direct.text.strip()) < self.min_content_chars)
        )
        if should_try_browser:
            browser_result = await fetch_with_crawl4ai(url, timeout_s=self.crawl4ai_timeout_s)
            if browser_result["ok"]:
                browser_text = clean_text(browser_result["text"])
                if len(browser_text) > len(direct.text):
                    direct.text = browser_text
                    direct.ok = True
                    direct.error = ""
                    direct.source = "visit_server"
                    direct.extraction_method = browser_result["extraction_method"]
                    direct.raw_text_chars = len(browser_text)
                    direct.final_url = browser_result.get("final_url") or direct.final_url or url
                    best = direct
            else:
                browser_error = browser_result["error"]

        text = clean_text(best.text)
        if text:
            text = select_relevant_excerpt(text, goal=goal or url, max_chars=self.content_length)

        payload = asdict(best)
        payload.update(
            {
                "content": text,
                "ok": bool(text),
                "raw_text_chars": best.raw_text_chars or len(best.text),
                "error": "" if text else (best.error or browser_error or "empty content"),
            }
        )
        payload.pop("text", None)
        return payload


async def fetch_with_crawl4ai(url: str, timeout_s: int) -> dict[str, Any]:
    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
        from crawl4ai.content_filter_strategy import PruningContentFilter
        from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
    except Exception as exc:
        return {"ok": False, "text": "", "error": f"crawl4ai unavailable: {exc}"}

    prune_filter = PruningContentFilter(threshold=0.4, threshold_type="dynamic", min_word_threshold=3)
    md_generator = DefaultMarkdownGenerator(
        content_filter=prune_filter,
        options={"ignore_links": False},
    )
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox", "--disable-extensions"],
    )
    crawler_config = CrawlerRunConfig(
        markdown_generator=md_generator,
        page_timeout=min(timeout_s * 1000, 60_000),
        verbose=False,
    )

    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await asyncio.wait_for(crawler.arun(url=url, config=crawler_config), timeout=timeout_s)
        if not result.success:
            return {"ok": False, "text": "", "error": result.error_message or "crawl4ai failed"}
        markdown = result.markdown.fit_markdown or result.markdown.raw_markdown or ""
        return {
            "ok": bool(markdown.strip()),
            "text": markdown,
            "error": "" if markdown.strip() else "crawl4ai returned empty content",
            "extraction_method": "html_crawl4ai",
            "final_url": url,
        }
    except asyncio.TimeoutError:
        return {"ok": False, "text": "", "error": "crawl4ai timeout"}
    except Exception as exc:
        return {"ok": False, "text": "", "error": str(exc)}


def is_probably_pdf(url: str, content_type: str = "") -> bool:
    lowered_type = (content_type or "").lower()
    if "pdf" in lowered_type:
        return True
    lowered_url = str(url).lower().split("?", 1)[0].split("#", 1)[0]
    return lowered_url.endswith(".pdf")


def configure_app(args: argparse.Namespace) -> None:
    if app is None:
        raise RuntimeError("Install fastapi and uvicorn to run the visit server.")
    app.state.service = VisitService(
        fetch_config=URLFetchConfig(
            timeout_s=args.fetch_timeout_s,
            max_concurrent_requests=args.max_concurrent_fetches,
            max_retries=args.fetch_max_retries,
            max_bytes=args.fetch_max_bytes,
            max_extracted_chars=args.fetch_max_extracted_chars,
            cache_dir=args.cache_dir,
            cache_errors=args.cache_errors,
        ),
        content_length=args.content_length,
        min_content_chars=args.min_content_chars,
        enable_crawl4ai=args.enable_crawl4ai,
        crawl4ai_timeout_s=args.crawl4ai_timeout_s,
    )


if app is not None:

    @app.on_event("startup")
    async def startup() -> None:
        if not hasattr(app.state, "service"):
            default_cache = os.environ.get("VISIT_CACHE_DIR", "")
            configure_app(
                argparse.Namespace(
                    fetch_timeout_s=int(os.environ.get("VISIT_FETCH_TIMEOUT_S", "30")),
                    max_concurrent_fetches=int(os.environ.get("VISIT_MAX_CONCURRENT_FETCHES", "16")),
                    fetch_max_retries=int(os.environ.get("VISIT_FETCH_MAX_RETRIES", "2")),
                    fetch_max_bytes=int(os.environ.get("VISIT_FETCH_MAX_BYTES", "2000000")),
                    fetch_max_extracted_chars=int(os.environ.get("VISIT_FETCH_MAX_EXTRACTED_CHARS", "50000")),
                    cache_dir=default_cache,
                    cache_errors=os.environ.get("VISIT_CACHE_ERRORS", "0").lower() in {"1", "true", "yes"},
                    content_length=int(os.environ.get("VISIT_CONTENT_LENGTH", "12000")),
                    min_content_chars=int(os.environ.get("VISIT_MIN_CONTENT_CHARS", "500")),
                    enable_crawl4ai=os.environ.get("VISIT_ENABLE_CRAWL4AI", "0").lower()
                    in {"1", "true", "yes"},
                    crawl4ai_timeout_s=int(os.environ.get("VISIT_CRAWL4AI_TIMEOUT_S", "45")),
                )
            )
        await app.state.service.start()

    @app.on_event("shutdown")
    async def shutdown() -> None:
        await app.state.service.close()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/visit", response_model=VisitResponse)
    async def visit(request: VisitRequest) -> dict[str, Any]:
        try:
            return await app.state.service.visit(request.url, request.goal)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DeepResearch URL visit service.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--fetch-timeout-s", type=int, default=30)
    parser.add_argument("--max-concurrent-fetches", type=int, default=16)
    parser.add_argument("--fetch-max-retries", type=int, default=2)
    parser.add_argument("--fetch-max-bytes", type=int, default=2_000_000)
    parser.add_argument("--fetch-max-extracted-chars", type=int, default=50_000)
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--cache-errors", action="store_true")
    parser.add_argument("--content-length", type=int, default=12_000)
    parser.add_argument("--min-content-chars", type=int, default=500)
    parser.add_argument("--enable-crawl4ai", action="store_true")
    parser.add_argument("--crawl4ai-timeout-s", type=int, default=45)
    args = parser.parse_args()

    configure_app(args)

    try:
        import uvicorn
    except Exception as exc:
        raise RuntimeError("Install uvicorn to run the visit server.") from exc
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
