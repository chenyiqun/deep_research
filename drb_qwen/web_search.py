from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from typing import Any


PROD_WEB_SEARCH_ENDPOINT = "http://edithai.devops.xiaohongshu.com/ext-tools/zhipu-web-search-vip"


@dataclass
class WebSearchConfig:
    endpoint: str = PROD_WEB_SEARCH_ENDPOINT
    access_key: str = ""
    search_engine: str = "search_prime"
    count: int = 15
    search_domain_filter: str = ""
    search_recency_filter: str = "noLimit"
    content_size: str = "high"
    timeout_s: int = 120
    max_concurrent_requests: int = 8
    max_retries: int = 3
    retry_sleep_s: float = 1.5


@dataclass
class SearchResult:
    title: str
    content: str
    link: str
    media: str = ""
    icon: str = ""
    refer: str = ""
    publish_date: str = ""
    search_query: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class WebSearchClient:
    """Async client for the Xiaohongshu EdithAI web search endpoint."""

    def __init__(self, config: WebSearchConfig) -> None:
        if not config.access_key:
            raise ValueError("WebSearchConfig.access_key is required. Pass it through WEB_SEARCH_API_KEY.")
        self.config = config
        self._session: Any | None = None
        self._semaphore = asyncio.Semaphore(config.max_concurrent_requests)

    async def __aenter__(self) -> "WebSearchClient":
        try:
            import aiohttp
        except ImportError as exc:
            raise RuntimeError("Install aiohttp to use async web search.") from exc

        timeout = aiohttp.ClientTimeout(total=self.config.timeout_s)
        self._session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._session is not None:
            await self._session.close()
        self._session = None

    async def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if self._session is None:
            raise RuntimeError("WebSearchClient must be used as an async context manager.")

        payload: dict[str, Any] = {
            "search_engine": self.config.search_engine,
            "search_query": query,
            "count": self.config.count,
            "search_recency_filter": self.config.search_recency_filter,
            "content_size": self.config.content_size,
        }
        if self.config.search_domain_filter:
            payload["search_domain_filter"] = self.config.search_domain_filter

        headers = {
            "X-EdithAI-Access-Key": self.config.access_key,
            "Content-Type": "application/json",
        }

        async with self._semaphore:
            last_error: Exception | None = None
            for attempt in range(1, self.config.max_retries + 1):
                try:
                    async with self._session.post(
                        self.config.endpoint,
                        json=payload,
                        headers=headers,
                    ) as response:
                        text = await response.text()
                        if response.status >= 400:
                            raise RuntimeError(
                                f"web search failed with HTTP {response.status}: {text[:1000]}"
                            )
                        data = await response.json()
                        return parse_search_results(data, query)[:top_k]
                except Exception as exc:
                    last_error = exc
                    if attempt >= self.config.max_retries:
                        break
                    await asyncio.sleep(self.config.retry_sleep_s * attempt)
            raise RuntimeError(f"web search failed after retries: {last_error}") from last_error


def parse_search_results(data: dict[str, Any], query: str) -> list[SearchResult]:
    results: list[SearchResult] = []
    raw_results = data.get("search_result", [])
    if not isinstance(raw_results, list):
        return results

    seen_links: set[str] = set()
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        link = str(item.get("link", "")).strip()
        title = str(item.get("title", "")).strip()
        content = str(item.get("content", "")).strip()
        if not link and not content:
            continue
        dedupe_key = link or f"{title}:{content[:120]}"
        if dedupe_key in seen_links:
            continue
        seen_links.add(dedupe_key)
        results.append(
            SearchResult(
                title=title,
                content=content,
                link=link,
                media=str(item.get("media", "")).strip(),
                icon=str(item.get("icon", "")).strip(),
                refer=str(item.get("refer", "")).strip(),
                publish_date=str(item.get("publish_date", "")).strip(),
                search_query=query,
            )
        )
    return results
