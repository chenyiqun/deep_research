from __future__ import annotations

import asyncio
import html
import re
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from io import BytesIO
from typing import Any
from urllib.parse import urlparse


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


@dataclass
class URLFetchConfig:
    timeout_s: int = 30
    visit_endpoint: str = ""
    visit_timeout_s: int = 60
    visit_fallback_to_direct_fetch: bool = True
    max_concurrent_requests: int = 16
    max_retries: int = 2
    retry_sleep_s: float = 1.0
    max_bytes: int = 2_000_000
    max_extracted_chars: int = 50_000
    user_agent: str = DEFAULT_USER_AGENT


@dataclass
class URLFetchResult:
    url: str
    ok: bool
    status: int | None = None
    content_type: str = ""
    final_url: str = ""
    text: str = ""
    error: str = ""
    source: str = "direct"
    visit_error: str = ""
    truncated_bytes: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["text_chars"] = len(self.text)
        data.pop("text", None)
        return data


class URLContentFetcher:
    """Async best-effort fetcher for turning search result URLs into reader text."""

    def __init__(self, config: URLFetchConfig | None = None) -> None:
        self.config = config or URLFetchConfig()
        self._session: Any | None = None
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

    async def __aenter__(self) -> "URLContentFetcher":
        try:
            import aiohttp
        except ImportError as exc:
            raise RuntimeError("Install aiohttp to use async URL fetching.") from exc

        timeout = aiohttp.ClientTimeout(total=self.config.timeout_s)
        headers = {"User-Agent": self.config.user_agent}
        self._session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._session is not None:
            await self._session.close()
        self._session = None

    async def fetch(self, url: str, goal: str = "") -> URLFetchResult:
        if self._session is None:
            raise RuntimeError("URLContentFetcher must be used as an async context manager.")

        url = str(url).strip()
        if not should_try_fetch_url(url):
            return URLFetchResult(url=url, ok=False, error="unsupported or empty URL")

        async with self._semaphore:
            visit_error = ""
            if self.config.visit_endpoint:
                visit_result = await self._fetch_with_visit_server(url, goal)
                if visit_result.ok or not self.config.visit_fallback_to_direct_fetch:
                    return visit_result
                visit_error = visit_result.error

            last_error: Exception | None = None
            for attempt in range(1, self.config.max_retries + 1):
                try:
                    result = await self._fetch_once(url)
                    result.visit_error = visit_error
                    return result
                except Exception as exc:
                    last_error = exc
                    if attempt >= self.config.max_retries:
                        break
                    await asyncio.sleep(self.config.retry_sleep_s * attempt)
            return URLFetchResult(url=url, ok=False, error=format_exception(last_error), visit_error=visit_error)

    async def _fetch_with_visit_server(self, url: str, goal: str) -> URLFetchResult:
        assert self._session is not None
        endpoint = normalize_visit_endpoint(self.config.visit_endpoint)
        last_error: Exception | None = None
        try:
            import aiohttp
        except ImportError as exc:
            return URLFetchResult(url=url, ok=False, error=str(exc), source="visit_server")

        for attempt in range(1, self.config.max_retries + 1):
            try:
                timeout = aiohttp.ClientTimeout(total=self.config.visit_timeout_s)
                async with self._session.post(
                    endpoint,
                    json={"url": url, "goal": goal},
                    timeout=timeout,
                ) as response:
                    raw_text = await response.text()
                    if response.status >= 400:
                        return URLFetchResult(
                            url=url,
                            ok=False,
                            status=response.status,
                            content_type=response.headers.get("content-type", ""),
                            final_url=url,
                            error=f"visit server HTTP {response.status}: {raw_text[:500]}",
                            source="visit_server",
                        )
                    try:
                        data = await response.json(content_type=None)
                    except Exception:
                        data = {"content": raw_text}
                    content = data.get("content", "") if isinstance(data, dict) else ""
                    content = clean_text(str(content))
                    return URLFetchResult(
                        url=url,
                        ok=bool(content),
                        status=response.status,
                        content_type=response.headers.get("content-type", ""),
                        final_url=url,
                        text=content[: self.config.max_extracted_chars],
                        error="" if content else "visit server returned empty content",
                        source="visit_server",
                    )
            except Exception as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    break
                await asyncio.sleep(self.config.retry_sleep_s * attempt)
        return URLFetchResult(url=url, ok=False, error=format_exception(last_error), source="visit_server")

    async def _fetch_once(self, url: str) -> URLFetchResult:
        assert self._session is not None
        async with self._session.get(url, allow_redirects=True) as response:
            body, truncated = await read_limited(response, self.config.max_bytes)
            content_type = response.headers.get("content-type", "")
            final_url = str(response.url)
            if response.status >= 400:
                return URLFetchResult(
                    url=url,
                    ok=False,
                    status=response.status,
                    content_type=content_type,
                    final_url=final_url,
                    error=f"HTTP {response.status}",
                    truncated_bytes=truncated,
                )

            text = extract_response_text(
                body=body,
                content_type=content_type,
                url=final_url,
                encoding=response.charset,
                max_chars=self.config.max_extracted_chars,
            )
            return URLFetchResult(
                url=url,
                ok=bool(text.strip()),
                status=response.status,
                content_type=content_type,
                final_url=final_url,
                text=text,
                error="" if text.strip() else "no extractable text",
                source="direct",
                truncated_bytes=truncated,
            )


async def read_limited(response: Any, max_bytes: int) -> tuple[bytes, bool]:
    chunks: list[bytes] = []
    total = 0
    truncated = False
    async for chunk in response.content.iter_chunked(65536):
        if not chunk:
            continue
        total += len(chunk)
        if total > max_bytes:
            keep = max_bytes - (total - len(chunk))
            if keep > 0:
                chunks.append(chunk[:keep])
            truncated = True
            break
        chunks.append(chunk)
    return b"".join(chunks), truncated


def should_try_fetch_url(url: str) -> bool:
    parsed = urlparse(str(url).strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def normalize_visit_endpoint(value: str) -> str:
    endpoint = str(value).strip().rstrip("/")
    if not endpoint:
        return ""
    if endpoint.endswith("/visit"):
        return endpoint
    return endpoint + "/visit"


def extract_response_text(
    body: bytes,
    content_type: str,
    url: str,
    encoding: str | None,
    max_chars: int,
) -> str:
    lowered_type = content_type.lower()
    lowered_url = url.lower().split("?", 1)[0]
    if "pdf" in lowered_type or lowered_url.endswith(".pdf"):
        text = extract_text_from_pdf(body)
    else:
        decoded = decode_body(body, encoding)
        if "html" in lowered_type or looks_like_html(decoded):
            text = extract_text_from_html(decoded)
        else:
            text = clean_text(decoded)
    return text[:max_chars]


def decode_body(body: bytes, encoding: str | None) -> str:
    encodings = [encoding, "utf-8", "gb18030", "latin-1"]
    for candidate in encodings:
        if not candidate:
            continue
        try:
            return body.decode(candidate, errors="replace")
        except LookupError:
            continue
    return body.decode("utf-8", errors="replace")


def looks_like_html(text: str) -> bool:
    sample = text[:2000].lower()
    return "<html" in sample or "<body" in sample or "<p" in sample or "<div" in sample


def extract_text_from_pdf(body: bytes) -> str:
    text = extract_text_from_pdf_with_pypdf(body)
    if text:
        return text
    return extract_text_from_pdf_with_pdfminer(body)


def extract_text_from_pdf_with_pypdf(body: bytes) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        return ""

    try:
        reader = PdfReader(BytesIO(body))
        texts: list[str] = []
        for page in reader.pages[:20]:
            page_text = page.extract_text() or ""
            if page_text.strip():
                texts.append(page_text)
        return clean_text("\n".join(texts))
    except Exception:
        return ""


def extract_text_from_pdf_with_pdfminer(body: bytes) -> str:
    try:
        from pdfminer.high_level import extract_text
    except Exception:
        return ""

    try:
        return clean_text(extract_text(BytesIO(body)) or "")
    except Exception:
        return ""


def format_exception(exc: Exception | None) -> str:
    if exc is None:
        return "unknown error"
    message = str(exc).strip()
    if message:
        return message
    return exc.__class__.__name__


class _HTMLTextExtractor(HTMLParser):
    _BLOCK_TAGS = {
        "article",
        "aside",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }
    _SKIP_TAGS = {"canvas", "form", "head", "iframe", "noscript", "script", "style", "svg"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth == 0 and tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if self._skip_depth == 0 and tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data:
            self.parts.append(data)


def extract_text_from_html(markup: str) -> str:
    parser = _HTMLTextExtractor()
    try:
        parser.feed(markup)
        parser.close()
        return clean_text("".join(parser.parts))
    except Exception:
        return clean_text(re.sub(r"<[^>]+>", " ", markup))


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = text.replace("\r", "\n")
    text = re.sub(r"[\t\f\v]+", " ", text)
    lines: list[str] = []
    for line in text.splitlines():
        line = re.sub(r" {2,}", " ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)
