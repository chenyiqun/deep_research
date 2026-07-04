from __future__ import annotations

import asyncio
import hashlib
import html
import json
import re
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
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
    cache_dir: str = ""
    cache_errors: bool = False


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
    cached: bool = False
    extraction_method: str = ""
    raw_text_chars: int = 0
    raw_content_chars: int = 0
    summary_provider: str = ""
    summary_model: str = ""
    summary_error: str = ""
    summary_chars: int = 0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["text_chars"] = len(self.text)
        if not data.get("raw_text_chars"):
            data["raw_text_chars"] = len(self.text)
        data.pop("text", None)
        return data


URL_FETCH_RESULT_FIELDS = set(URLFetchResult.__dataclass_fields__)


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
        headers = {
            "User-Agent": self.config.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf;q=0.8,*/*;q=0.7",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }
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
            cache_key = self._cache_key(url, goal)
            cached = self._read_cache(cache_key)
            if cached is not None:
                return cached

            visit_error = ""
            if self.config.visit_endpoint:
                visit_result = await self._fetch_with_visit_server(url, goal)
                if visit_result.ok or not self.config.visit_fallback_to_direct_fetch:
                    self._write_cache(cache_key, visit_result)
                    return visit_result
                visit_error = visit_result.error

            last_error: Exception | None = None
            for attempt in range(1, self.config.max_retries + 1):
                try:
                    result = await self._fetch_once(url)
                    result.visit_error = visit_error
                    self._write_cache(cache_key, result)
                    return result
                except Exception as exc:
                    last_error = exc
                    if attempt >= self.config.max_retries:
                        break
                    await asyncio.sleep(self.config.retry_sleep_s * attempt)
            result = URLFetchResult(url=url, ok=False, error=format_exception(last_error), visit_error=visit_error)
            self._write_cache(cache_key, result)
            return result

    def _cache_key(self, url: str, goal: str) -> str:
        if not self.config.cache_dir:
            return ""
        endpoint = normalize_visit_endpoint(self.config.visit_endpoint)
        if endpoint:
            raw_key = f"visit:{endpoint}:{url}:{goal}"
        else:
            raw_key = f"direct:{url}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def _cache_path(self, cache_key: str) -> Path | None:
        if not cache_key or not self.config.cache_dir:
            return None
        return Path(self.config.cache_dir) / f"{cache_key}.json"

    def _read_cache(self, cache_key: str) -> URLFetchResult | None:
        path = self._cache_path(cache_key)
        if path is None or not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            result = URLFetchResult(**{k: v for k, v in data.items() if k in URL_FETCH_RESULT_FIELDS})
            result.cached = True
            return result
        except Exception:
            return None

    def _write_cache(self, cache_key: str, result: URLFetchResult) -> None:
        if not result.ok and not self.config.cache_errors:
            return
        path = self._cache_path(cache_key)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = asdict(result)
            payload["cached"] = False
            tmp_path = path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp_path.replace(path)
        except Exception:
            return

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
                    extraction_method = ""
                    final_url = url
                    status = None
                    content_type = ""
                    raw_text_chars = len(content)
                    server_error = ""
                    summary_provider = ""
                    summary_model = ""
                    summary_error = ""
                    summary_chars = 0
                    raw_content_chars = 0
                    if isinstance(data, dict):
                        extraction_method = str(data.get("extraction_method", "") or data.get("method", ""))
                        final_url = str(data.get("final_url", "") or url)
                        status = safe_int(data.get("status"), None) if data.get("status") is not None else None
                        content_type = str(data.get("content_type") or "")
                        raw_text_chars = safe_int(data.get("raw_text_chars"), len(content))
                        server_error = str(data.get("error") or "")
                        summary_provider = str(data.get("summary_provider") or "")
                        summary_model = str(data.get("summary_model") or "")
                        summary_error = str(data.get("summary_error") or "")
                        summary_chars = safe_int(data.get("summary_chars"), 0)
                        raw_content_chars = safe_int(data.get("raw_content_chars"), 0)
                    return URLFetchResult(
                        url=url,
                        ok=bool(content),
                        status=status,
                        content_type=content_type,
                        final_url=final_url,
                        text=content[: self.config.max_extracted_chars],
                        error="" if content else (server_error or "visit server returned empty content"),
                        source="visit_server",
                        extraction_method=extraction_method or "visit_server",
                        raw_text_chars=raw_text_chars,
                        raw_content_chars=raw_content_chars,
                        summary_provider=summary_provider,
                        summary_model=summary_model,
                        summary_error=summary_error,
                        summary_chars=summary_chars,
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

            text, extraction_method, raw_text_chars = extract_response_text(
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
                extraction_method=extraction_method,
                raw_text_chars=raw_text_chars,
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
) -> tuple[str, str, int]:
    lowered_type = content_type.lower()
    lowered_url = url.lower().split("?", 1)[0]
    if "pdf" in lowered_type or lowered_url.endswith(".pdf"):
        text, method = extract_text_from_pdf(body)
    else:
        decoded = decode_body(body, encoding)
        if "html" in lowered_type or looks_like_html(decoded):
            text, method = extract_text_from_html(decoded)
        else:
            text = clean_text(decoded)
            method = "plain_text"
    raw_text_chars = len(text)
    return text[:max_chars], method, raw_text_chars


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


def extract_text_from_pdf(body: bytes) -> tuple[str, str]:
    text = extract_text_from_pdf_with_pymupdf(body)
    if text:
        return text, "pdf_pymupdf"
    text = extract_text_from_pdf_with_pypdf(body)
    if text:
        return text, "pdf_pypdf"
    text = extract_text_from_pdf_with_pdfminer(body)
    if text:
        return text, "pdf_pdfminer"
    return "", "pdf_none"


def extract_text_from_pdf_with_pymupdf(body: bytes) -> str:
    try:
        import fitz
    except Exception:
        return ""

    try:
        texts: list[str] = []
        with fitz.open(stream=body, filetype="pdf") as doc:
            for page in doc[:30]:
                page_text = page.get_text() or ""
                if page_text.strip():
                    texts.append(page_text)
        return clean_text("\n".join(texts))
    except Exception:
        return ""


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


def safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


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


def extract_text_from_html(markup: str) -> tuple[str, str]:
    text = extract_text_from_html_with_trafilatura(markup)
    if text:
        return text, "html_trafilatura"
    text = extract_text_from_html_with_bs4(markup)
    if text:
        return text, "html_bs4"
    parser = _HTMLTextExtractor()
    try:
        parser.feed(markup)
        parser.close()
        return clean_text("".join(parser.parts)), "html_parser"
    except Exception:
        return clean_text(re.sub(r"<[^>]+>", " ", markup)), "html_regex"


def extract_text_from_html_with_trafilatura(markup: str) -> str:
    try:
        import trafilatura
    except Exception:
        return ""

    try:
        extracted = trafilatura.extract(
            markup,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
            output_format="txt",
        )
        return clean_text(extracted or "")
    except Exception:
        return ""


def extract_text_from_html_with_bs4(markup: str) -> str:
    try:
        from bs4 import BeautifulSoup
    except Exception:
        return ""

    try:
        soup = BeautifulSoup(markup, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg", "canvas", "form", "iframe"]):
            tag.decompose()
        candidates = soup.find_all(["article", "main"])
        if not candidates:
            candidates = soup.find_all(["p", "h1", "h2", "h3", "li", "td", "th"])
        parts: list[str] = []
        for item in candidates:
            text = item.get_text("\n", strip=True)
            if text:
                parts.append(text)
        return clean_text("\n".join(parts))
    except Exception:
        return ""


def select_relevant_excerpt(text: str, goal: str, max_chars: int) -> str:
    """Return a goal-focused window instead of blindly taking the first chars."""
    text = clean_text(text)
    if not text or len(text) <= max_chars:
        return text

    chunks = split_text_chunks(text)
    if not chunks:
        return text[:max_chars]

    goal_tokens = tokenize_for_relevance(goal)
    if not goal_tokens:
        return text[:max_chars]

    scored: list[tuple[float, int, str]] = []
    for idx, chunk in enumerate(chunks):
        chunk_tokens = tokenize_for_relevance(chunk)
        overlap = len(goal_tokens & chunk_tokens)
        numeric_bonus = min(6, len(re.findall(r"\d+(?:[.,]\d+)?%?|20\d{2}|19\d{2}", chunk)))
        title_bonus = 1.5 if idx <= 2 else 0.0
        score = overlap + numeric_bonus * 0.35 + title_bonus
        scored.append((score, idx, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected_indices: set[int] = set()
    candidate_indices: list[int] = []
    for score, idx, _chunk in scored[: max(4, min(12, len(scored)))]:
        if score <= 0 and selected_indices:
            continue
        for candidate in (idx, idx - 1, idx + 1):
            if candidate < 0 or candidate >= len(chunks) or candidate in selected_indices:
                continue
            selected_indices.add(candidate)
            candidate_indices.append(candidate)
    ordered = [chunks[idx] for idx in candidate_indices]
    output: list[str] = []
    total = 0
    for chunk in ordered:
        addition = len(chunk) + 2
        if total + addition > max_chars:
            remaining = max_chars - total
            if remaining > 200:
                output.append(chunk[:remaining])
            break
        output.append(chunk)
        total += addition

    if not output and ordered:
        return ordered[0][:max_chars]
    if not output:
        return text[:max_chars]
    return "\n\n".join(output)


def split_text_chunks(text: str) -> list[str]:
    paragraphs = [line.strip() for line in text.splitlines() if line.strip()]
    chunks: list[str] = []
    buffer: list[str] = []
    buffer_len = 0
    for paragraph in paragraphs:
        if len(paragraph) >= 120:
            if buffer:
                chunks.append(" ".join(buffer))
                buffer = []
                buffer_len = 0
            chunks.append(paragraph)
            continue
        buffer.append(paragraph)
        buffer_len += len(paragraph)
        if buffer_len >= 240:
            chunks.append(" ".join(buffer))
            buffer = []
            buffer_len = 0
    if buffer:
        chunks.append(" ".join(buffer))
    return chunks


def tokenize_for_relevance(text: str) -> set[str]:
    lowered = text.lower()
    tokens = set(re.findall(r"[a-z0-9][a-z0-9_\-]{1,}|[\u4e00-\u9fff]{2,}", lowered))
    chinese_runs = re.findall(r"[\u4e00-\u9fff]{2,}", lowered)
    for run in chinese_runs:
        for size in (2, 3, 4):
            if len(run) < size:
                continue
            for idx in range(0, len(run) - size + 1):
                tokens.add(run[idx : idx + size])
    return {token for token in tokens if len(token) >= 2}


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
