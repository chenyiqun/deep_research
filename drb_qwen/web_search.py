from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from typing import Any


PROD_WEB_SEARCH_ENDPOINT = "http://edithai.devops.xiaohongshu.com/ext-tools/zhipu-web-search-vip"
DEFAULT_SEARCH_ENGINE = "search_prime"


@dataclass(frozen=True)
class SearchEngineProfile:
    key: str
    provider: str
    native_content: bool
    fetch_pages_by_default: bool
    api_engines: tuple[str, ...]

    @property
    def result_source_quality(self) -> str:
        return "search_native_content" if self.native_content else "search_snippet"

    @property
    def result_extraction_method(self) -> str:
        return f"{self.key}_content" if self.native_content else "search_snippet"


# `search_live` returns reader-ready Sogou result content. The other engines
# are treated as discovery/snippet providers in auto mode and therefore use
# the URL fetcher. Callers can override this with --url-fetch-mode.
SEARCH_ENGINE_PROFILES: dict[str, SearchEngineProfile] = {
    "search_pro_jina": SearchEngineProfile("search_pro_jina", "jina", False, True, ("search_pro_jina",)),
    "search_prime": SearchEngineProfile("search_prime", "google", False, True, ("search_prime",)),
    "search_pro_ms": SearchEngineProfile("search_pro_ms", "bing", False, True, ("search_pro_ms",)),
    "search_live": SearchEngineProfile(
        "search_live",
        "sogou",
        True,
        False,
        ("search_pro_sogou", "search_live"),
    ),
    "search_lite": SearchEngineProfile(
        "search_lite",
        "quark",
        False,
        True,
        ("search_pro_quark", "search_lite"),
    ),
    "search_plus": SearchEngineProfile("search_plus", "baidu", False, True, ("search_plus",)),
}
SUPPORTED_SEARCH_ENGINES = tuple(SEARCH_ENGINE_PROFILES)
URL_FETCH_MODES = ("auto", "always", "never")


def get_search_engine_profile(search_engine: str) -> SearchEngineProfile:
    key = str(search_engine or DEFAULT_SEARCH_ENGINE).strip().lower()
    try:
        return SEARCH_ENGINE_PROFILES[key]
    except KeyError as exc:
        supported = ", ".join(SUPPORTED_SEARCH_ENGINES)
        raise ValueError(f"Unsupported search engine {search_engine!r}. Choose one of: {supported}") from exc


def should_fetch_result_pages(search_engine: str, mode: str = "auto") -> bool:
    profile = get_search_engine_profile(search_engine)
    normalized_mode = str(mode or "auto").strip().lower()
    if normalized_mode not in URL_FETCH_MODES:
        raise ValueError(f"Unsupported URL fetch mode {mode!r}. Choose one of: {', '.join(URL_FETCH_MODES)}")
    if normalized_mode == "always":
        return True
    if normalized_mode == "never":
        return False
    return profile.fetch_pages_by_default


@dataclass
class WebSearchConfig:
    endpoint: str = PROD_WEB_SEARCH_ENDPOINT
    access_key: str = ""
    search_engine: str = DEFAULT_SEARCH_ENGINE
    count: int = 10
    search_domain_filter: str = ""
    search_recency_filter: str = "noLimit"
    content_size: str = "high"
    timeout_s: int = 120
    max_concurrent_requests: int = 8
    max_retries: int = 3
    retry_sleep_s: float = 1.5

    def __post_init__(self) -> None:
        self.search_engine = get_search_engine_profile(self.search_engine).key
        positive = {
            "count": self.count,
            "timeout_s": self.timeout_s,
            "max_concurrent_requests": self.max_concurrent_requests,
            "max_retries": self.max_retries,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"WebSearchConfig fields must be positive: {', '.join(invalid)}")
        if self.search_engine == "search_live" and self.count not in {10, 20, 30, 40, 50}:
            raise ValueError("search_live count must be one of: 10, 20, 30, 40, 50")
        if self.retry_sleep_s < 0:
            raise ValueError("WebSearchConfig.retry_sleep_s cannot be negative")


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
    search_engine: str = ""
    api_search_engine: str = ""
    content_kind: str = ""
    source_quality: str = ""
    extraction_method: str = ""

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

        profile = get_search_engine_profile(self.config.search_engine)
        engine_errors: list[str] = []
        async with self._semaphore:
            for api_engine in profile.api_engines:
                try:
                    return await self._search_with_api_engine(
                        payload=payload,
                        headers=headers,
                        query=query,
                        top_k=top_k,
                        api_engine=api_engine,
                    )
                except SearchEngineUnavailableError as exc:
                    engine_errors.append(str(exc))
                    continue
        details = "; ".join(engine_errors) or "no compatible API engine was attempted"
        raise RuntimeError(
            f"web search engine {self.config.search_engine} is unavailable; tried "
            f"{', '.join(profile.api_engines)}: {details}"
        )

    async def _search_with_api_engine(
        self,
        *,
        payload: dict[str, Any],
        headers: dict[str, str],
        query: str,
        top_k: int,
        api_engine: str,
    ) -> list[SearchResult]:
        if self._session is None:
            raise RuntimeError("WebSearchClient must be used as an async context manager.")
        request_payload = {**payload, "search_engine": api_engine}
        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                async with self._session.post(
                    self.config.endpoint,
                    json=request_payload,
                    headers=headers,
                ) as response:
                    text = await response.text()
                    if response.status >= 400:
                        message = f"api_engine={api_engine} HTTP {response.status}: {text[:1000]}"
                        if is_unknown_search_engine_error(response.status, text):
                            raise SearchEngineUnavailableError(message)
                        raise RuntimeError(f"web search failed with {message}")
                    data = await response.json()
                    return parse_search_results(
                        data,
                        query,
                        search_engine=self.config.search_engine,
                        api_search_engine=api_engine,
                    )[:top_k]
            except SearchEngineUnavailableError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    break
                await asyncio.sleep(self.config.retry_sleep_s * attempt)
        raise RuntimeError(f"web search failed after retries with api_engine={api_engine}: {last_error}") from last_error


class SearchEngineUnavailableError(RuntimeError):
    pass


def is_unknown_search_engine_error(status: int, response_text: str) -> bool:
    lowered = str(response_text or "").lower()
    return status in {400, 404} and (
        '"code":"1211"' in lowered
        or '"code": "1211"' in lowered
        or "模型不存在" in response_text
        or "model not found" in lowered
        or "model does not exist" in lowered
    )


def parse_search_results(
    data: Any,
    query: str,
    search_engine: str = DEFAULT_SEARCH_ENGINE,
    api_search_engine: str = "",
) -> list[SearchResult]:
    results: list[SearchResult] = []
    profile = get_search_engine_profile(search_engine)
    raw_results = find_raw_results(data)

    seen_links: set[str] = set()
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        link = first_text(item, "link", "url", "source_url", "href")
        title = first_text(item, "title", "name", "source_title")
        content = first_text(item, "content", "text", "snippet", "summary", "description")
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
                media=first_text(item, "media", "site_name", "source"),
                icon=first_text(item, "icon", "favicon"),
                refer=first_text(item, "refer", "reference"),
                publish_date=first_text(item, "publish_date", "published_at", "date", "time"),
                search_query=query,
                search_engine=profile.key,
                api_search_engine=api_search_engine or profile.api_engines[0],
                content_kind="native_content" if profile.native_content else "snippet",
                source_quality=profile.result_source_quality,
                extraction_method=profile.result_extraction_method,
            )
        )
    return results


def find_raw_results(data: Any) -> list[Any]:
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    for key in ("search_result", "search_results", "results", "result", "items"):
        value = data.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            nested_results = find_raw_results(value)
            if nested_results:
                return nested_results
    nested = data.get("data")
    if nested is not data:
        return find_raw_results(nested)
    return []


def first_text(item: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""
