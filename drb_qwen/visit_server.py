from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .async_llm_client import AsyncChatClient, AsyncChatConfig
from .json_utils import extract_json
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
    raw_content_chars: int = 0
    cached: bool = False
    summary_provider: str = ""
    summary_model: str = ""
    summary_error: str = ""
    summary_chars: int = 0
    error: str = ""


class VisitService:
    def __init__(
        self,
        fetch_config: URLFetchConfig,
        content_length: int = 12_000,
        min_content_chars: int = 500,
        enable_crawl4ai: bool = False,
        crawl4ai_timeout_s: int = 45,
        summary_provider: str = "none",
        summary_base_url: str = "http://127.0.0.1:8000/v1",
        summary_model: str = "qwen3-32b",
        summary_api_key: str = "EMPTY",
        summary_timeout_s: int = 300,
        summary_max_concurrent_requests: int = 4,
        summary_input_max_chars: int = 60_000,
        summary_chunk_chars: int = 20_000,
        summary_max_tokens: int = 2048,
        summary_merge_max_tokens: int = 3072,
    ) -> None:
        self.fetch_config = fetch_config
        self.content_length = content_length
        self.min_content_chars = min_content_chars
        self.enable_crawl4ai = enable_crawl4ai
        self.crawl4ai_timeout_s = crawl4ai_timeout_s
        self.summary_provider = normalize_summary_provider(summary_provider)
        self.summary_base_url = summary_base_url
        self.summary_model = summary_model
        self.summary_api_key = summary_api_key
        self.summary_timeout_s = summary_timeout_s
        self.summary_max_concurrent_requests = summary_max_concurrent_requests
        self.summary_input_max_chars = summary_input_max_chars
        self.summary_chunk_chars = summary_chunk_chars
        self.summary_max_tokens = summary_max_tokens
        self.summary_merge_max_tokens = summary_merge_max_tokens
        self.summary_cache_dir = Path(fetch_config.cache_dir) / "goal_summary" if fetch_config.cache_dir else None
        self.fetcher: URLContentFetcher | None = None
        self.summarizer: AsyncChatClient | None = None

    async def start(self) -> None:
        self.fetcher = await URLContentFetcher(self.fetch_config).__aenter__()
        if self.summary_provider == "local_vllm":
            self.summarizer = await AsyncChatClient(
                AsyncChatConfig(
                    base_url=self.summary_base_url,
                    model=self.summary_model,
                    api_key=self.summary_api_key,
                    timeout_s=self.summary_timeout_s,
                    max_concurrent_requests=self.summary_max_concurrent_requests,
                    strip_thinking=True,
                )
            ).__aenter__()

    async def close(self) -> None:
        if self.summarizer is not None:
            await self.summarizer.__aexit__(None, None, None)
        if self.fetcher is not None:
            await self.fetcher.__aexit__(None, None, None)
        self.summarizer = None
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

        raw_text = clean_text(best.text)
        text = ""
        summary_error = ""
        summary_chars = 0
        if raw_text and self.summary_provider != "none":
            summary_payload = await self._summarize_url_content(url=url, goal=goal, fetch_result=best, text=raw_text)
            if summary_payload.get("ok"):
                text = str(summary_payload.get("content") or "").strip()
                summary_chars = len(text)
            else:
                summary_error = str(summary_payload.get("error") or "")
        if not text and raw_text:
            text = select_relevant_excerpt(raw_text, goal=goal or url, max_chars=self.content_length)

        payload = asdict(best)
        extraction_method = best.extraction_method or best.source or "direct"
        if summary_chars:
            extraction_method = append_method(extraction_method, f"goal_summary_{self.summary_provider}")
        payload.update(
            {
                "content": text,
                "ok": bool(text),
                "extraction_method": extraction_method,
                "raw_text_chars": best.raw_text_chars or len(raw_text),
                "raw_content_chars": best.raw_text_chars or len(raw_text),
                "summary_provider": self.summary_provider if summary_chars else "",
                "summary_model": self.summary_model if summary_chars else "",
                "summary_error": summary_error,
                "summary_chars": summary_chars,
                "error": "" if text else (best.error or browser_error or "empty content"),
            }
        )
        payload.pop("text", None)
        return payload

    async def _summarize_url_content(
        self,
        url: str,
        goal: str,
        fetch_result: Any,
        text: str,
    ) -> dict[str, Any]:
        if self.summarizer is None:
            return {"ok": False, "error": "summary provider is not initialized"}
        selected_text = select_relevant_excerpt(
            text,
            goal=goal or url,
            max_chars=max(self.summary_input_max_chars, self.summary_chunk_chars),
        )
        cache_key = self._summary_cache_key(url=url, goal=goal, text=selected_text)
        cached = self._read_summary_cache(cache_key)
        if cached is not None:
            cached["cached_summary"] = True
            return cached

        chunks = split_text_chunks(selected_text, max_chars=self.summary_chunk_chars)
        if not chunks:
            return {"ok": False, "error": "empty text for summary"}
        try:
            if len(chunks) == 1:
                response = await self.summarizer.chat(
                    build_goal_summary_prompt(
                        url=url,
                        title=str(getattr(fetch_result, "final_url", "") or url),
                        goal=goal,
                        text=chunks[0],
                    ),
                    system_prompt=GOAL_SUMMARY_SYSTEM_PROMPT,
                    temperature=0.0,
                    max_tokens=self.summary_max_tokens,
                )
                summary = parse_summary_response(response)
            else:
                partials = []
                for idx, chunk in enumerate(chunks, start=1):
                    response = await self.summarizer.chat(
                        build_goal_summary_prompt(
                            url=url,
                            title=str(getattr(fetch_result, "final_url", "") or url),
                            goal=goal,
                            text=chunk,
                            chunk_idx=idx,
                            chunk_total=len(chunks),
                        ),
                        system_prompt=GOAL_SUMMARY_SYSTEM_PROMPT,
                        temperature=0.0,
                        max_tokens=self.summary_max_tokens,
                    )
                    partials.append(parse_summary_response(response))
                response = await self.summarizer.chat(
                    build_merge_summary_prompt(url=url, goal=goal, partial_summaries=partials),
                    system_prompt=GOAL_SUMMARY_SYSTEM_PROMPT,
                    temperature=0.0,
                    max_tokens=self.summary_merge_max_tokens,
                )
                summary = parse_summary_response(response)
            content = format_goal_summary(summary)
            payload = {"ok": bool(content.strip()), "content": content, "error": ""}
            self._write_summary_cache(cache_key, payload)
            return payload
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _summary_cache_key(self, url: str, goal: str, text: str) -> str:
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        raw_key = f"v1:{self.summary_provider}:{self.summary_model}:{url}:{goal}:{text_hash}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def _summary_cache_path(self, cache_key: str) -> Path | None:
        if self.summary_cache_dir is None:
            return None
        return self.summary_cache_dir / f"{cache_key}.json"

    def _read_summary_cache(self, cache_key: str) -> dict[str, Any] | None:
        path = self._summary_cache_path(cache_key)
        if path is None or not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            return None
        return None

    def _write_summary_cache(self, cache_key: str, payload: dict[str, Any]) -> None:
        path = self._summary_cache_path(cache_key)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp_path.replace(path)
        except Exception:
            return


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


GOAL_SUMMARY_SYSTEM_PROMPT = (
    "You are a goal-based visit summarizer in a deep research system. "
    "Extract only information supported by the provided page text. "
    "Do not invent facts, URLs, dates, numbers, or source names. Return JSON only."
)


def build_goal_summary_prompt(
    url: str,
    title: str,
    goal: str,
    text: str,
    chunk_idx: int = 1,
    chunk_total: int = 1,
) -> str:
    chunk_note = ""
    if chunk_total > 1:
        chunk_note = f"\nThis is chunk {chunk_idx} of {chunk_total}. Only summarize information present in this chunk."
    return f"""
请根据访问目标，对下面单个网页正文做 goal-based visit summary。只抽取与目标相关的信息。

要求：
1. 只使用 source_text 中明确支持的信息，不要编造。
2. 保留数字、日期、主体、口径、限制条件和不确定性。
3. 如果内容与目标无关，relevance 给低分，并说明 limitations。
4. evidence_quotes 应该是短摘录或近似原文要点，不要长篇复制。
5. source_url 必须原样使用输入 URL。
6. 只输出 JSON。{chunk_note}

<source_url>
{url}
</source_url>

<source_title_or_final_url>
{title}
</source_title_or_final_url>

<visit_goal>
{goal}
</visit_goal>

<source_text>
{text}
</source_text>

JSON schema:
{{
  "source_url": "{json_escape(url)}",
  "relevance": 0.0,
  "brief_summary": "与访问目标相关的简短总结",
  "key_facts": [
    {{
      "claim": "事实、数据或观点",
      "evidence": "支持该 claim 的短依据",
      "confidence": "high/medium/low"
    }}
  ],
  "useful_statistics": [
    {{
      "metric": "指标名",
      "value": "数值",
      "context": "时间、范围、口径或主体"
    }}
  ],
  "evidence_quotes": ["短摘录或接近原文的证据片段"],
  "limitations": ["这页内容的限制、不确定性或与目标不匹配之处"],
  "possible_followups": ["基于这页信息还需要继续搜索的问题"]
}}
"""


def build_merge_summary_prompt(url: str, goal: str, partial_summaries: list[dict[str, Any]]) -> str:
    summaries_text = json.dumps(partial_summaries, ensure_ascii=False, indent=2)
    return f"""
请把同一个 URL 的多个 chunk-level visit summary 合并成一个去重后的 goal-based visit summary。

要求：
1. 只合并 partial_summaries 中已有的信息，不新增外部事实。
2. 删除重复 claim，保留更具体、更有数字和日期的版本。
3. 对冲突、不确定或口径不同的信息，在 limitations 中说明。
4. source_url 必须原样使用输入 URL。
5. 只输出 JSON。

<source_url>
{url}
</source_url>

<visit_goal>
{goal}
</visit_goal>

<partial_summaries>
{summaries_text}
</partial_summaries>

JSON schema:
{{
  "source_url": "{json_escape(url)}",
  "relevance": 0.0,
  "brief_summary": "合并后的简短总结",
  "key_facts": [
    {{
      "claim": "事实、数据或观点",
      "evidence": "支持该 claim 的短依据",
      "confidence": "high/medium/low"
    }}
  ],
  "useful_statistics": [
    {{
      "metric": "指标名",
      "value": "数值",
      "context": "时间、范围、口径或主体"
    }}
  ],
  "evidence_quotes": ["短摘录或接近原文的证据片段"],
  "limitations": ["限制、不确定性或冲突"],
  "possible_followups": ["还需要继续搜索的问题"]
}}
"""


def parse_summary_response(response: str) -> dict[str, Any]:
    try:
        parsed = extract_json(response)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {
        "relevance": 0.0,
        "brief_summary": response.strip(),
        "key_facts": [],
        "useful_statistics": [],
        "evidence_quotes": [],
        "limitations": ["summary model did not return parseable JSON"],
        "possible_followups": [],
    }


def format_goal_summary(summary: dict[str, Any]) -> str:
    return (
        "Source acquisition: goal_based_visit_summary. "
        "This is a local-vLLM summary of fetched page text, not a raw search snippet.\n"
        + json.dumps(summary, ensure_ascii=False, indent=2)
    )


def split_text_chunks(text: str, max_chars: int, overlap_chars: int = 800) -> list[str]:
    text = text.strip()
    if not text:
        return []
    max_chars = max(max_chars, 2000)
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            split_at = max(text.rfind("\n", start, end), text.rfind("。", start, end), text.rfind(".", start, end))
            if split_at > start + max_chars // 2:
                end = split_at + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap_chars, start + 1)
    return [chunk for chunk in chunks if chunk]


def normalize_summary_provider(value: str) -> str:
    provider = str(value or "none").strip().lower()
    if provider in {"", "0", "false", "no", "none", "off"}:
        return "none"
    if provider in {"local", "local_vllm", "vllm", "qwen"}:
        return "local_vllm"
    raise ValueError(f"Unsupported summary provider: {value}")


def append_method(method: str, suffix: str) -> str:
    if not method:
        return suffix
    if suffix in method:
        return method
    return f"{method}+{suffix}"


def json_escape(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace('"', '\\"')


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
        summary_provider=args.summary_provider,
        summary_base_url=args.summary_base_url,
        summary_model=args.summary_model,
        summary_api_key=args.summary_api_key,
        summary_timeout_s=args.summary_timeout_s,
        summary_max_concurrent_requests=args.summary_max_concurrent_requests,
        summary_input_max_chars=args.summary_input_max_chars,
        summary_chunk_chars=args.summary_chunk_chars,
        summary_max_tokens=args.summary_max_tokens,
        summary_merge_max_tokens=args.summary_merge_max_tokens,
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
                    summary_provider=os.environ.get("VISIT_SUMMARY_PROVIDER", "none"),
                    summary_base_url=os.environ.get("VISIT_SUMMARY_BASE_URL", "http://127.0.0.1:8000/v1"),
                    summary_model=os.environ.get("VISIT_SUMMARY_MODEL", "qwen3-32b"),
                    summary_api_key=os.environ.get("VISIT_SUMMARY_API_KEY", "EMPTY"),
                    summary_timeout_s=int(os.environ.get("VISIT_SUMMARY_TIMEOUT_S", "300")),
                    summary_max_concurrent_requests=int(
                        os.environ.get("VISIT_SUMMARY_MAX_CONCURRENT_REQUESTS", "4")
                    ),
                    summary_input_max_chars=int(os.environ.get("VISIT_SUMMARY_INPUT_MAX_CHARS", "60000")),
                    summary_chunk_chars=int(os.environ.get("VISIT_SUMMARY_CHUNK_CHARS", "20000")),
                    summary_max_tokens=int(os.environ.get("VISIT_SUMMARY_MAX_TOKENS", "2048")),
                    summary_merge_max_tokens=int(os.environ.get("VISIT_SUMMARY_MERGE_MAX_TOKENS", "3072")),
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
    parser.add_argument("--summary-provider", default="none", choices=["none", "local_vllm"])
    parser.add_argument("--summary-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--summary-model", default="qwen3-32b")
    parser.add_argument("--summary-api-key", default="EMPTY")
    parser.add_argument("--summary-timeout-s", type=int, default=300)
    parser.add_argument("--summary-max-concurrent-requests", type=int, default=4)
    parser.add_argument("--summary-input-max-chars", type=int, default=60_000)
    parser.add_argument("--summary-chunk-chars", type=int, default=20_000)
    parser.add_argument("--summary-max-tokens", type=int, default=2048)
    parser.add_argument("--summary-merge-max-tokens", type=int, default=3072)
    args = parser.parse_args()

    configure_app(args)

    try:
        import uvicorn
    except Exception as exc:
        raise RuntimeError("Install uvicorn to run the visit server.") from exc
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
