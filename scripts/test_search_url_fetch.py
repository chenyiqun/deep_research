from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import urlparse

from drb_qwen.io_utils import load_jsonl, write_jsonl, write_text
from drb_qwen.url_fetcher import URLContentFetcher, URLFetchConfig, URLFetchResult, should_try_fetch_url
from drb_qwen.web_search import (
    DEFAULT_SEARCH_ENGINE,
    PROD_WEB_SEARCH_ENDPOINT,
    SUPPORTED_SEARCH_ENGINES,
    SearchResult,
    WebSearchClient,
    WebSearchConfig,
)


def main() -> None:
    args = build_parser().parse_args()
    if args.disable_log:
        asyncio.run(run_async(args))
        return

    log_file = Path(args.log_file) if args.log_file else default_log_file(args.summary_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_file.open("a", encoding="utf-8", buffering=1) as f:
        sys.stdout = TeeTextIO(original_stdout, f)
        sys.stderr = TeeTextIO(original_stderr, f)
        try:
            print(f"Log file: {log_file}")
            print(f"Started at: {datetime.now().isoformat(timespec='seconds')}")
            asyncio.run(run_async(args))
            print(f"Finished at: {datetime.now().isoformat(timespec='seconds')}")
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


async def run_async(args: argparse.Namespace) -> None:
    query_rows = load_query_rows(args)
    if not query_rows:
        raise ValueError("No queries provided. Use --query or --query-file.")

    access_key = args.web_search_api_key or os.environ.get("WEB_SEARCH_API_KEY", "")
    if not access_key:
        raise ValueError("WEB_SEARCH_API_KEY is required for normal web search.")

    output_file = Path(args.output_file)
    summary_file = Path(args.summary_file)
    search_results_file = Path(args.search_results_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    search_results_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Queries: {len(query_rows)}")
    print(f"Web search endpoint: {args.web_search_endpoint}")
    print(f"Search engine: {args.search_engine}")
    print(f"Search count: {args.search_count}")
    print(f"Search top-k: {args.search_top_k}")
    print(f"URL visit endpoint: {args.url_visit_endpoint or '<none>'}")
    print(f"Min usable extracted chars: {args.min_extracted_chars}")

    search_config = WebSearchConfig(
        endpoint=args.web_search_endpoint,
        access_key=access_key,
        search_engine=args.search_engine,
        count=args.search_count,
        search_domain_filter=args.search_domain_filter,
        search_recency_filter=args.search_recency_filter,
        content_size=args.search_content_size,
        timeout_s=args.web_search_timeout_s,
        max_concurrent_requests=args.max_concurrent_searches,
        max_retries=args.web_search_max_retries,
    )
    fetch_config = URLFetchConfig(
        timeout_s=args.url_fetch_timeout_s,
        visit_endpoint=args.url_visit_endpoint,
        visit_timeout_s=args.url_visit_timeout_s,
        visit_fallback_to_direct_fetch=not args.disable_url_visit_fallback,
        max_concurrent_requests=args.max_concurrent_url_fetches,
        max_retries=args.url_fetch_max_retries,
        max_bytes=args.url_fetch_max_bytes,
        max_extracted_chars=args.url_fetch_max_extracted_chars,
        cache_dir=args.url_fetch_cache_dir,
        cache_errors=args.url_fetch_cache_errors,
    )

    async with WebSearchClient(search_config) as search_client:
        search_rows, search_results = await run_searches(search_client, query_rows, args.search_top_k)

    write_jsonl(search_results_file, search_rows)
    unique_results = dedupe_results(search_results)
    if args.max_urls and args.max_urls > 0:
        unique_results = unique_results[: args.max_urls]

    print(f"Search results: {len(search_results)}")
    print(f"Unique fetchable URLs: {len(unique_results)}")

    async with URLContentFetcher(fetch_config) as fetcher:
        fetch_rows = await run_fetches(
            fetcher,
            unique_results,
            args.min_extracted_chars,
            print_each=not args.quiet_fetch_rows,
        )

    write_jsonl(output_file, fetch_rows)
    summary = summarize(fetch_rows, search_rows, query_rows, args.min_extracted_chars)
    write_text(summary_file, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    print("Done. Summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Search results file: {search_results_file}")
    print(f"Fetch results file: {output_file}")
    print(f"Summary file: {summary_file}")


async def run_searches(
    search_client: WebSearchClient,
    query_rows: list[dict[str, Any]],
    top_k: int,
) -> tuple[list[dict[str, Any]], list[SearchResult]]:
    async def search_one(row: dict[str, Any]) -> tuple[dict[str, Any], list[SearchResult], str]:
        query = str(row["query"])
        try:
            results = await search_client.search(query, top_k=top_k)
            return row, results, ""
        except Exception as exc:
            return row, [], str(exc)

    tasks = [asyncio.create_task(search_one(row)) for row in query_rows]
    search_rows: list[dict[str, Any]] = []
    all_results: list[SearchResult] = []
    for task in progress_as_completed(tasks, desc="Web search"):
        row, results, error = await task
        if error:
            search_rows.append(
                {
                    "query_id": row["query_id"],
                    "query": row["query"],
                    "error": error,
                    "num_results": 0,
                }
            )
            continue
        for rank, result in enumerate(results, start=1):
            item = result.to_dict()
            item.update(
                {
                    "query_id": row["query_id"],
                    "query": row["query"],
                    "rank": rank,
                    "snippet_chars": len(result.content),
                    "fetchable": should_try_fetch_url(result.link),
                }
            )
            search_rows.append(item)
            if should_try_fetch_url(result.link):
                all_results.append(result)
    return search_rows, all_results


async def run_fetches(
    fetcher: URLContentFetcher,
    results: list[SearchResult],
    min_extracted_chars: int,
    print_each: bool = True,
) -> list[dict[str, Any]]:
    async def fetch_one(index: int, result: SearchResult) -> dict[str, Any]:
        goal = build_fetch_goal(result)
        try:
            fetch_result = await fetcher.fetch(result.link, goal=goal)
        except Exception as exc:
            fetch_result = URLFetchResult(url=result.link, ok=False, error=str(exc))
        row = fetch_result.to_dict()
        text_chars = int(row.get("text_chars") or 0)
        usable = bool(fetch_result.ok and text_chars >= min_extracted_chars)
        row.update(
            {
                "index": index,
                "search_query": result.search_query,
                "source_title": result.title,
                "source_url": result.link,
                "media": result.media,
                "publish_date": result.publish_date,
                "snippet_chars": len(result.content),
                "text_chars": text_chars,
                "usable": usable,
                "domain": domain_of(result.link),
                "source_quality": source_quality_from_fetch(fetch_result, usable),
            }
        )
        return row

    tasks = [asyncio.create_task(fetch_one(index, result)) for index, result in enumerate(results, start=1)]
    rows: list[dict[str, Any]] = []
    for task in progress_as_completed(tasks, desc="URL fetch"):
        row = await task
        rows.append(row)
        if print_each:
            print(format_fetch_row(row, len(results)))
    rows.sort(key=lambda row: int(row.get("index") or 0))
    return rows


def load_query_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, query in enumerate(args.query or [], start=1):
        query = str(query).strip()
        if query:
            rows.append({"query_id": f"cli-{idx}", "query": query})

    if args.query_file:
        for row in load_jsonl(args.query_file):
            if args.only_lang and row.get("language") != args.only_lang:
                continue
            query = str(row.get(args.query_field, "")).strip()
            if not query:
                continue
            rows.append(
                {
                    "query_id": row.get("id", f"file-{len(rows) + 1}"),
                    "query": query,
                    "topic": row.get("topic", ""),
                    "language": row.get("language", ""),
                }
            )
            if args.query_limit and len(rows) >= args.query_limit:
                break

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        query = str(row["query"]).strip()
        norm = " ".join(query.lower().split())
        if norm in seen:
            continue
        seen.add(norm)
        deduped.append(row)
    return deduped


def dedupe_results(results: list[SearchResult]) -> list[SearchResult]:
    output: list[SearchResult] = []
    seen: set[str] = set()
    for result in results:
        key = result.link.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(result)
    return output


def summarize(
    fetch_rows: list[dict[str, Any]],
    search_rows: list[dict[str, Any]],
    query_rows: list[dict[str, Any]],
    min_extracted_chars: int,
) -> dict[str, Any]:
    total = len(fetch_rows)
    fetch_ok = sum(1 for row in fetch_rows if row.get("ok") is True)
    usable = sum(1 for row in fetch_rows if row.get("usable") is True)
    text_chars = [int(row.get("text_chars") or 0) for row in fetch_rows]
    raw_chars = [int(row.get("raw_text_chars") or 0) for row in fetch_rows]
    return {
        "num_queries": len(query_rows),
        "num_search_rows": len(search_rows),
        "num_fetchable_urls": total,
        "min_extracted_chars": min_extracted_chars,
        "fetch_ok": fetch_ok,
        "fetch_ok_rate": safe_rate(fetch_ok, total),
        "usable": usable,
        "usable_rate": safe_rate(usable, total),
        "avg_text_chars": int(sum(text_chars) / len(text_chars)) if text_chars else 0,
        "max_text_chars": max(text_chars) if text_chars else 0,
        "avg_raw_text_chars": int(sum(raw_chars) / len(raw_chars)) if raw_chars else 0,
        "max_raw_text_chars": max(raw_chars) if raw_chars else 0,
        "source_quality_counts": counter_dict(row.get("source_quality") for row in fetch_rows),
        "extraction_method_counts": counter_dict(row.get("extraction_method") for row in fetch_rows),
        "summary_provider_counts": counter_dict(row.get("summary_provider") for row in fetch_rows),
        "status_counts": counter_dict(row.get("status") for row in fetch_rows),
        "domain_counts": counter_dict((row.get("domain") for row in fetch_rows), limit=15),
        "top_errors": counter_dict((row.get("error") for row in fetch_rows if row.get("error")), limit=10),
    }


def source_quality_from_fetch(fetch_result: URLFetchResult, usable: bool) -> str:
    if usable:
        if "goal_summary" in (fetch_result.extraction_method or ""):
            return "goal_summary_visit"
        if fetch_result.source == "visit_server":
            return "full_text_visit"
        if "pdf" in (fetch_result.extraction_method or ""):
            return "full_text_pdf"
        return "full_text"
    if fetch_result.error or fetch_result.visit_error:
        return "fetch_failed"
    return "too_short"


def build_fetch_goal(result: SearchResult) -> str:
    return (
        "请抽取网页中最有助于回答本轮搜索 query 的正文内容。"
        "优先保留事实、数据、日期、主体、观点、证据和不确定性；忽略导航栏、广告、评论和版权声明。\n"
        f"搜索 query: {result.search_query}\n"
        f"搜索结果标题: {result.title}\n"
        f"搜索结果摘要: {result.content[:1000]}"
    )


def format_fetch_row(row: dict[str, Any], total: int) -> str:
    index = int(row.get("index") or 0)
    title = " ".join(str(row.get("source_title") or "").split())[:80]
    error = " ".join(str(row.get("error") or row.get("visit_error") or "").split())[:160]
    return (
        f"[fetch {index}/{total}] "
        f"ok={row.get('ok')} usable={row.get('usable')} "
        f"quality={row.get('source_quality') or '-'} "
        f"method={row.get('extraction_method') or '-'} "
        f"text_chars={row.get('text_chars') or 0} "
        f"raw_chars={row.get('raw_text_chars') or 0} "
        f"status={row.get('status') or '-'} "
        f"domain={row.get('domain') or '-'} "
        f"title={title!r} "
        f"error={error!r}"
    )


def progress_as_completed(tasks: list[asyncio.Task[Any]], desc: str) -> Any:
    try:
        from tqdm import tqdm

        return tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=desc)
    except Exception:
        return asyncio.as_completed(tasks)


def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def counter_dict(values: Any, limit: int | None = None) -> dict[str, int]:
    counts = Counter(str(value) for value in values if str(value or "").strip())
    items = counts.most_common(limit)
    return {key: count for key, count in items}


def safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test URL extraction success rate after normal web search."
    )
    parser.add_argument("--query", action="append", default=[], help="Search query. Can be passed multiple times.")
    parser.add_argument("--query-file", default="", help="JSONL file such as DeepResearch query.jsonl.")
    parser.add_argument("--query-field", default="prompt")
    parser.add_argument("--query-limit", type=int, default=0)
    parser.add_argument("--only-lang", default="")

    parser.add_argument("--output-file", default="outputs/search_url_fetch_test/url_fetch_results.jsonl")
    parser.add_argument("--summary-file", default="outputs/search_url_fetch_test/summary.json")
    parser.add_argument("--search-results-file", default="outputs/search_url_fetch_test/search_results.jsonl")
    parser.add_argument(
        "--log-file",
        default="",
        help="Write process logs here. Defaults to <summary-file-dir>/logs/search_url_fetch_<timestamp>.log.",
    )
    parser.add_argument("--disable-log", action="store_true", help="Do not tee stdout/stderr to a log file.")

    parser.add_argument("--web-search-endpoint", default=PROD_WEB_SEARCH_ENDPOINT)
    parser.add_argument("--web-search-api-key", default="")
    parser.add_argument(
        "--search-engine",
        default=DEFAULT_SEARCH_ENGINE,
        choices=SUPPORTED_SEARCH_ENGINES,
    )
    parser.add_argument("--search-count", type=int, default=15)
    parser.add_argument("--search-top-k", type=int, default=8)
    parser.add_argument("--search-domain-filter", default="")
    parser.add_argument("--search-recency-filter", default="noLimit")
    parser.add_argument("--search-content-size", default="high")
    parser.add_argument("--web-search-timeout-s", type=int, default=120)
    parser.add_argument("--web-search-max-retries", type=int, default=3)
    parser.add_argument("--max-concurrent-searches", type=int, default=4)

    parser.add_argument("--url-visit-endpoint", default="")
    parser.add_argument("--url-visit-timeout-s", type=int, default=120)
    parser.add_argument("--disable-url-visit-fallback", action="store_true")
    parser.add_argument("--url-fetch-timeout-s", type=int, default=60)
    parser.add_argument("--url-fetch-max-retries", type=int, default=2)
    parser.add_argument("--url-fetch-max-bytes", type=int, default=4_000_000)
    parser.add_argument("--url-fetch-max-extracted-chars", type=int, default=80_000)
    parser.add_argument("--url-fetch-cache-dir", default="")
    parser.add_argument("--url-fetch-cache-errors", action="store_true")
    parser.add_argument("--max-concurrent-url-fetches", type=int, default=16)
    parser.add_argument("--min-extracted-chars", type=int, default=500)
    parser.add_argument("--max-urls", type=int, default=0)
    parser.add_argument("--quiet-fetch-rows", action="store_true")
    return parser


class TeeTextIO:
    def __init__(self, primary: TextIO, secondary: TextIO) -> None:
        self.primary = primary
        self.secondary = secondary
        self.encoding = getattr(primary, "encoding", "utf-8")

    def write(self, text: str) -> int:
        self.primary.write(text)
        self.secondary.write(text)
        self.flush()
        return len(text)

    def flush(self) -> None:
        self.primary.flush()
        self.secondary.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.primary, "isatty", lambda: False)())


def default_log_file(summary_file: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(summary_file).parent / "logs" / f"search_url_fetch_{timestamp}.log"


if __name__ == "__main__":
    main()
