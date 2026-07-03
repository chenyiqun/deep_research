from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import Counter
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .async_llm_client import AsyncChatClient, AsyncChatConfig
from .deep_research_workflow import AsyncDeepResearchWorkflow, DeepResearchConfig
from .io_utils import existing_ids, filter_tasks, load_jsonl, prepare_output_file, write_jsonl, write_text
from .url_fetcher import URLContentFetcher, URLFetchConfig
from .web_search import PROD_WEB_SEARCH_ENDPOINT, WebSearchClient, WebSearchConfig


async def run_task(
    task: dict[str, Any],
    workflow: AsyncDeepResearchWorkflow,
    output_file: Path,
    output_lock: asyncio.Lock,
    trace_dir: Path | None,
    model_name: str,
    progress: tqdm,
) -> None:
    task_id = int(task["id"])
    task_log = ""
    try:
        result = await workflow.run(task)
        summary = summarize_research_result(result)
        trace_path = None
        if trace_dir is not None:
            trace_path = trace_dir / f"{task_id}.json"
            write_text(
                trace_path,
                json.dumps(
                    {
                        "id": task_id,
                        "prompt": task["prompt"],
                        "state": result["state"],
                        "trace": result["trace"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
            )
        row = {
            "id": task_id,
            "topic": task.get("topic"),
            "language": task.get("language"),
            "prompt": task["prompt"],
            "article": result["article"],
            "model": model_name,
        }
        if trace_path is not None:
            row["research_trace_file"] = str(trace_path)
        task_log = format_research_task_log(task_id, row, summary, trace_path)
    except Exception as exc:
        row = {
            "id": task_id,
            "topic": task.get("topic"),
            "language": task.get("language"),
            "prompt": task.get("prompt"),
            "article": "",
            "model": model_name,
            "error": str(exc),
        }
        task_log = f"[report task={task_id}] ERROR {str(exc)[:500]}"

    async with output_lock:
        write_jsonl(output_file, [row], append=True)
        if task_log:
            progress.write(task_log)
        progress.update(1)


def summarize_research_result(result: dict[str, Any]) -> dict[str, Any]:
    trace = result.get("trace", [])
    if not isinstance(trace, list):
        trace = []

    fetches: list[dict[str, Any]] = []
    search_queries = 0
    search_results = 0
    reader_notes = 0
    for round_item in trace:
        if not isinstance(round_item, dict):
            continue
        plan = round_item.get("plan", {})
        if isinstance(plan, dict):
            queries = plan.get("search_queries", [])
            if isinstance(queries, list):
                search_queries += len(queries)
        results_by_query = round_item.get("search_results", {})
        if isinstance(results_by_query, dict):
            search_results += sum(len(v) for v in results_by_query.values() if isinstance(v, list))
        round_fetches = round_item.get("source_fetches", [])
        if isinstance(round_fetches, list):
            fetches.extend(item for item in round_fetches if isinstance(item, dict))
        notes = round_item.get("reader_notes", [])
        if isinstance(notes, list):
            reader_notes += len(notes)

    reader_chars = [safe_int(item.get("reader_content_chars")) for item in fetches]
    reader_chars = [value for value in reader_chars if value >= 0]
    raw_text_chars = [safe_int(item.get("raw_text_chars")) for item in fetches]
    raw_text_chars = [value for value in raw_text_chars if value >= 0]
    errors = [
        normalize_log_value(item.get("error"))
        for item in fetches
        if normalize_log_value(item.get("error"))
    ]
    error_counts = Counter(errors)
    quality_counts = Counter(
        normalize_log_value(item.get("source_quality"))
        for item in fetches
        if normalize_log_value(item.get("source_quality"))
    )
    method_counts = Counter(
        normalize_log_value(item.get("extraction_method"))
        for item in fetches
        if normalize_log_value(item.get("extraction_method"))
    )
    state = result.get("state", {})
    if not isinstance(state, dict):
        state = {}
    return {
        "rounds": len(trace),
        "search_queries": search_queries,
        "search_results": search_results,
        "url_fetches": len(fetches),
        "fetch_ok": sum(1 for item in fetches if item.get("ok") is True),
        "full_text_sources": sum(1 for item in fetches if item.get("used_full_content") is True),
        "fetch_cached": sum(1 for item in fetches if item.get("cached") is True),
        "reader_notes": reader_notes,
        "reader_chars_avg": int(sum(reader_chars) / len(reader_chars)) if reader_chars else 0,
        "reader_chars_max": max(reader_chars) if reader_chars else 0,
        "raw_text_chars_avg": int(sum(raw_text_chars) / len(raw_text_chars)) if raw_text_chars else 0,
        "raw_text_chars_max": max(raw_text_chars) if raw_text_chars else 0,
        "fetch_errors": sum(error_counts.values()),
        "top_fetch_errors": ",".join(
            f"{name}:{count}" for name, count in error_counts.most_common(3)
        ),
        "source_quality": ",".join(
            f"{name}:{count}" for name, count in quality_counts.most_common(4)
        ),
        "extraction_methods": ",".join(
            f"{name}:{count}" for name, count in method_counts.most_common(4)
        ),
        "article_chars": len(str(result.get("article", ""))),
        "state_findings": len(state.get("findings", [])) if isinstance(state.get("findings"), list) else 0,
        "state_evidence": len(state.get("evidence", [])) if isinstance(state.get("evidence"), list) else 0,
    }


def format_research_task_log(
    task_id: int,
    row: dict[str, Any],
    summary: dict[str, Any],
    trace_path: Path | None,
) -> str:
    trace_text = str(trace_path) if trace_path is not None else "-"
    error_text = summary.get("top_fetch_errors") or "-"
    quality_text = summary.get("source_quality") or "-"
    method_text = summary.get("extraction_methods") or "-"
    return (
        f"[report task={task_id}] ok "
        f"article_chars={summary['article_chars']} "
        f"rounds={summary['rounds']} "
        f"search_queries={summary['search_queries']} "
        f"search_results={summary['search_results']} "
        f"url_fetches={summary['url_fetches']} "
        f"fetch_ok={summary['fetch_ok']} "
        f"full_text={summary['full_text_sources']} "
        f"cached={summary['fetch_cached']} "
        f"reader_notes={summary['reader_notes']} "
        f"reader_chars_avg={summary['reader_chars_avg']} "
        f"reader_chars_max={summary['reader_chars_max']} "
        f"raw_text_avg={summary['raw_text_chars_avg']} "
        f"raw_text_max={summary['raw_text_chars_max']} "
        f"fetch_errors={summary['fetch_errors']} "
        f"top_fetch_errors={error_text} "
        f"source_quality={quality_text} "
        f"methods={method_text} "
        f"state_findings={summary['state_findings']} "
        f"state_evidence={summary['state_evidence']} "
        f"trace={trace_text}"
    )


def safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return -1


def normalize_log_value(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text.split(":", 1)[0][:80]


async def run_async(args: argparse.Namespace) -> None:
    tasks = load_jsonl(args.query_file)
    skip_ids = existing_ids(args.output_file) if args.resume else set()
    tasks = filter_tasks(tasks, only_lang=args.only_lang, limit=args.limit, skip_ids=skip_ids)
    if not tasks:
        print("No tasks to process.")
        return

    output_file = prepare_output_file(args.output_file, resume=args.resume)
    trace_dir = Path(args.trace_dir) if args.trace_dir else None
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)

    web_search_api_key = args.web_search_api_key or os.environ.get(args.web_search_api_key_env, "")
    if not web_search_api_key:
        raise ValueError(
            f"Missing web search access key. Set {args.web_search_api_key_env} or pass --web-search-api-key."
        )

    llm_config = AsyncChatConfig(
        base_url=args.llm_base_url,
        model=args.llm_model,
        api_key=args.llm_api_key,
        timeout_s=args.llm_timeout_s,
        max_concurrent_requests=args.max_concurrent_llm_calls,
        max_retries=args.llm_max_retries,
    )
    search_config = WebSearchConfig(
        endpoint=args.web_search_endpoint,
        access_key=web_search_api_key,
        search_engine=args.search_engine,
        count=args.search_count,
        search_domain_filter=args.search_domain_filter,
        search_recency_filter=args.search_recency_filter,
        content_size=args.search_content_size,
        max_concurrent_requests=args.max_concurrent_searches,
        max_retries=args.search_max_retries,
    )
    workflow_config = DeepResearchConfig(
        max_rounds=args.max_rounds,
        min_rounds=args.min_rounds,
        max_search_queries_per_round=args.max_search_queries_per_round,
        search_top_k=args.search_top_k,
        search_count=args.search_count,
        fetch_full_content=not args.disable_url_fetch,
        min_fetched_content_chars=args.min_fetched_content_chars,
        max_concurrent_readers=args.max_concurrent_readers,
        planner_max_tokens=args.planner_max_tokens,
        reader_max_tokens=args.reader_max_tokens,
        summarizer_max_tokens=args.summarizer_max_tokens,
        state_updater_max_tokens=args.state_updater_max_tokens,
        report_max_tokens=args.report_max_tokens,
        source_content_max_chars=args.source_content_max_chars,
        state_prompt_max_chars=args.state_prompt_max_chars,
        evidence_prompt_max_chars=args.evidence_prompt_max_chars,
    )
    url_fetch_config = URLFetchConfig(
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

    print(f"Async vLLM base URL: {args.llm_base_url}")
    print(f"Async vLLM model: {args.llm_model}")
    print(f"Web search endpoint: {args.web_search_endpoint}")
    print(f"Tasks: {len(tasks)}")
    print(f"Max concurrent tasks: {args.max_concurrent_tasks}")
    print(f"Max concurrent LLM calls: {args.max_concurrent_llm_calls}")
    print(f"Max rounds: {args.max_rounds}")
    print(f"Search count: {args.search_count}")
    print(f"Search top-k: {args.search_top_k}")
    print(f"Search queries per round: {args.max_search_queries_per_round}")
    print(f"Search domain filter: {args.search_domain_filter or '<none>'}")
    print(f"URL fetch enabled: {not args.disable_url_fetch}")
    if not args.disable_url_fetch:
        if args.url_visit_endpoint:
            print(f"URL visit endpoint: {args.url_visit_endpoint}")
        print(f"Max concurrent URL fetches: {args.max_concurrent_url_fetches}")
        print(f"URL fetch timeout: {args.url_fetch_timeout_s}s")
        print(f"URL fetch max bytes: {args.url_fetch_max_bytes}")
        print(f"URL fetch max extracted chars: {args.url_fetch_max_extracted_chars}")
        print(f"URL fetch cache dir: {args.url_fetch_cache_dir or '<disabled>'}")
        print(f"URL fetch cache errors: {args.url_fetch_cache_errors}")
        print(f"Min fetched content chars: {args.min_fetched_content_chars}")
        print(f"Source content max chars for reader: {args.source_content_max_chars}")
    print(f"Max concurrent readers: {args.max_concurrent_readers}")
    print(f"Trace dir: {trace_dir if trace_dir is not None else '<none>'}")

    task_semaphore = asyncio.Semaphore(args.max_concurrent_tasks)
    output_lock = asyncio.Lock()

    async with AsyncExitStack() as stack:
        llm = await stack.enter_async_context(AsyncChatClient(llm_config))
        search_client = await stack.enter_async_context(WebSearchClient(search_config))
        content_fetcher = None
        if not args.disable_url_fetch:
            content_fetcher = await stack.enter_async_context(URLContentFetcher(url_fetch_config))
        workflow = AsyncDeepResearchWorkflow(
            llm=llm,
            search_client=search_client,
            content_fetcher=content_fetcher,
            config=workflow_config,
        )
        progress = tqdm(total=len(tasks), desc="Async deep research")

        async def guarded_run(task: dict[str, Any]) -> None:
            async with task_semaphore:
                await run_task(
                    task=task,
                    workflow=workflow,
                    output_file=output_file,
                    output_lock=output_lock,
                    trace_dir=trace_dir,
                    model_name=args.llm_model,
                    progress=progress,
                )

        await asyncio.gather(*(guarded_run(task) for task in tasks))
        progress.close()

    print(f"Saved async deep research reports to {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate DeepResearch reports with an async multi-agent workflow."
    )
    parser.add_argument("--query-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--trace-dir", default=None)
    parser.add_argument("--only-lang", choices=["zh", "en"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--llm-model", default="qwen3-32b")
    parser.add_argument("--llm-api-key", default="EMPTY")
    parser.add_argument("--llm-timeout-s", type=int, default=600)
    parser.add_argument("--llm-max-retries", type=int, default=3)
    parser.add_argument("--max-concurrent-llm-calls", type=int, default=16)

    parser.add_argument("--web-search-endpoint", default=PROD_WEB_SEARCH_ENDPOINT)
    parser.add_argument("--web-search-api-key-env", default="WEB_SEARCH_API_KEY")
    parser.add_argument("--web-search-api-key", default=None)
    parser.add_argument("--search-engine", default="search_prime")
    parser.add_argument("--search-count", type=int, default=15)
    parser.add_argument("--search-top-k", type=int, default=5)
    parser.add_argument("--search-domain-filter", default="")
    parser.add_argument("--search-recency-filter", default="noLimit")
    parser.add_argument("--search-content-size", default="high")
    parser.add_argument("--search-max-retries", type=int, default=3)
    parser.add_argument("--max-concurrent-searches", type=int, default=8)
    parser.add_argument("--disable-url-fetch", action="store_true")
    parser.add_argument("--url-visit-endpoint", default="")
    parser.add_argument("--url-visit-timeout-s", type=int, default=60)
    parser.add_argument("--disable-url-visit-fallback", action="store_true")
    parser.add_argument("--max-concurrent-url-fetches", type=int, default=16)
    parser.add_argument("--url-fetch-timeout-s", type=int, default=30)
    parser.add_argument("--url-fetch-max-retries", type=int, default=2)
    parser.add_argument("--url-fetch-max-bytes", type=int, default=2_000_000)
    parser.add_argument("--url-fetch-max-extracted-chars", type=int, default=50_000)
    parser.add_argument("--url-fetch-cache-dir", default="")
    parser.add_argument("--url-fetch-cache-errors", action="store_true")
    parser.add_argument("--min-fetched-content-chars", type=int, default=500)

    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--min-rounds", type=int, default=1)
    parser.add_argument("--max-search-queries-per-round", type=int, default=3)
    parser.add_argument("--max-concurrent-tasks", type=int, default=4)
    parser.add_argument("--max-concurrent-readers", type=int, default=12)
    parser.add_argument("--planner-max-tokens", type=int, default=2048)
    parser.add_argument("--reader-max-tokens", type=int, default=2048)
    parser.add_argument("--summarizer-max-tokens", type=int, default=3072)
    parser.add_argument("--state-updater-max-tokens", type=int, default=4096)
    parser.add_argument("--report-max-tokens", type=int, default=8192)
    parser.add_argument("--source-content-max-chars", type=int, default=12000)
    parser.add_argument("--state-prompt-max-chars", type=int, default=24000)
    parser.add_argument("--evidence-prompt-max-chars", type=int, default=36000)
    args = parser.parse_args()
    asyncio.run(run_async(args))


if __name__ == "__main__":
    main()
