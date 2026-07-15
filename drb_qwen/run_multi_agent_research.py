from __future__ import annotations

import argparse
import asyncio
from contextlib import AsyncExitStack
import json
import os
from pathlib import Path
from typing import Any

from .async_llm_client import AsyncChatClient, AsyncChatConfig
from .deep_research_workflow import AsyncDeepResearchWorkflow, DeepResearchConfig
from .io_utils import write_text
from .url_fetcher import URLContentFetcher, URLFetchConfig
from .web_search import (
    DEFAULT_SEARCH_ENGINE,
    PROD_WEB_SEARCH_ENDPOINT,
    SUPPORTED_SEARCH_ENGINES,
    URL_FETCH_MODES,
    WebSearchClient,
    WebSearchConfig,
    should_fetch_result_pages,
)


async def run_async(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_state_dir = args.run_state_dir or str(output_dir / "run_state")
    web_search_api_key = args.web_search_api_key or os.environ.get(args.web_search_api_key_env, "")
    if not web_search_api_key:
        raise ValueError(
            f"Missing web search access key. Set {args.web_search_api_key_env} or pass --web-search-api-key."
        )

    task = {
        "id": args.task_id,
        "prompt": args.prompt,
        "language": args.language,
        "topic": args.topic,
    }
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
        search_recency_filter=args.search_recency_filter,
        content_size=args.search_content_size,
        max_concurrent_requests=args.max_concurrent_searches,
        max_retries=args.search_max_retries,
    )
    url_fetch_mode = "never" if args.disable_url_fetch else args.url_fetch_mode
    fetch_full_content = should_fetch_result_pages(search_config.search_engine, url_fetch_mode)
    fetch_config = URLFetchConfig(
        timeout_s=args.url_fetch_timeout_s,
        visit_endpoint=args.url_visit_endpoint,
        visit_timeout_s=args.url_visit_timeout_s,
        visit_fallback_to_direct_fetch=not args.disable_url_visit_fallback,
        max_concurrent_requests=args.max_concurrent_url_fetches,
        max_retries=args.url_fetch_max_retries,
        max_bytes=args.url_fetch_max_bytes,
        max_extracted_chars=args.url_fetch_max_extracted_chars,
        cache_dir=args.url_fetch_cache_dir or str(output_dir / "url_cache"),
    )
    workflow_config = DeepResearchConfig(
        max_rounds=args.max_rounds,
        max_initial_tasks=args.max_initial_tasks,
        max_researchers=args.max_researchers,
        max_subtasks=args.max_subtasks,
        max_new_tasks_per_round=args.max_new_tasks_per_round,
        max_react_steps=args.max_react_steps,
        max_search_queries_per_round=args.max_queries_per_step,
        max_tool_calls_per_subtask=args.max_tool_calls_per_subtask,
        max_total_tool_calls=args.max_total_tool_calls,
        max_total_searches=args.max_total_searches,
        max_total_tokens=args.max_total_tokens,
        max_run_seconds=args.max_run_seconds,
        search_top_k=args.search_top_k,
        fetch_full_content=fetch_full_content,
        min_fetched_content_chars=args.min_fetched_content_chars,
        max_concurrent_readers=args.max_concurrent_readers,
        min_total_claims=args.min_total_claims,
        min_coverage_ratio=args.min_coverage_ratio,
        citation_audit_enabled=not args.disable_citation_audit,
        max_audit_rounds=args.max_audit_rounds,
        max_repair_tasks=args.max_repair_tasks,
        planner_max_tokens=args.planner_max_tokens,
        reader_max_tokens=args.reader_max_tokens,
        summarizer_max_tokens=args.researcher_max_tokens,
        state_updater_max_tokens=args.replan_max_tokens,
        report_max_tokens=args.report_max_tokens,
        auditor_max_tokens=args.auditor_max_tokens,
        source_content_max_chars=args.source_content_max_chars,
        run_state_dir=run_state_dir,
        resume_runs=args.resume,
        max_model_len=args.max_model_len,
        context_safety_tokens=args.context_safety_tokens,
        tokenizer_path=args.tokenizer_path,
        inference_max_concurrent_requests=args.max_concurrent_llm_calls,
        inference_control_concurrency=args.max_concurrent_control_calls,
        inference_long_output_concurrency=args.max_concurrent_long_calls,
        inference_max_concurrent_per_run=args.max_concurrent_llm_calls_per_run,
        inference_max_inflight_tokens=args.max_inflight_llm_tokens,
        inference_structured_outputs=not args.disable_structured_outputs,
        inference_forward_priority=args.forward_vllm_priority,
        inference_disable_thinking_for_json=not args.enable_agent_thinking,
    )

    async with AsyncExitStack() as stack:
        llm = await stack.enter_async_context(AsyncChatClient(llm_config))
        search_client = await stack.enter_async_context(WebSearchClient(search_config))
        fetcher = None
        if fetch_full_content:
            fetcher = await stack.enter_async_context(URLContentFetcher(fetch_config))
        workflow = AsyncDeepResearchWorkflow(
            llm=llm,
            search_client=search_client,
            content_fetcher=fetcher,
            config=workflow_config,
        )
        result = await workflow.run(task, run_id=args.run_id or None, resume=args.resume)

    write_text(output_dir / "report.md", result["article"] + "\n")
    write_text(
        output_dir / "result.json",
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
    )
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "phase": result["state"]["phase"],
                "subtasks": len(result["state"]["tasks"]),
                "claims": len(result["state"]["claims"]),
                "evidence": len(result["state"]["evidence"]),
                "audit_passed": bool((result.get("audit") or {}).get("passed")),
                "search_engine": search_config.search_engine,
                "url_fetch_mode": url_fetch_mode,
                "url_fetch_enabled": fetch_full_content,
                "report_file": str(output_dir / "report.md"),
                "result_file": str(output_dir / "result.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one event-driven multi-agent deep-research task.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--language", choices=["zh", "en"], default="zh")
    parser.add_argument("--topic", default="")
    parser.add_argument("--task-id", default="single")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-state-dir", default="")
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
    parser.add_argument(
        "--search-engine",
        default=DEFAULT_SEARCH_ENGINE,
        choices=SUPPORTED_SEARCH_ENGINES,
    )
    parser.add_argument("--search-count", type=int, default=15)
    parser.add_argument("--search-top-k", type=int, default=5)
    parser.add_argument("--search-recency-filter", default="noLimit")
    parser.add_argument("--search-content-size", default="high")
    parser.add_argument("--search-max-retries", type=int, default=3)
    parser.add_argument("--max-concurrent-searches", type=int, default=8)

    parser.add_argument(
        "--url-fetch-mode",
        default="auto",
        choices=URL_FETCH_MODES,
        help="auto follows the search-engine profile; search_live skips page fetching by default.",
    )
    parser.add_argument(
        "--disable-url-fetch",
        action="store_true",
        help="Deprecated alias for --url-fetch-mode never.",
    )
    parser.add_argument("--url-visit-endpoint", default="")
    parser.add_argument("--url-visit-timeout-s", type=int, default=60)
    parser.add_argument("--disable-url-visit-fallback", action="store_true")
    parser.add_argument("--max-concurrent-url-fetches", type=int, default=16)
    parser.add_argument("--url-fetch-timeout-s", type=int, default=30)
    parser.add_argument("--url-fetch-max-retries", type=int, default=2)
    parser.add_argument("--url-fetch-max-bytes", type=int, default=2_000_000)
    parser.add_argument("--url-fetch-max-extracted-chars", type=int, default=50_000)
    parser.add_argument("--url-fetch-cache-dir", default="")
    parser.add_argument("--min-fetched-content-chars", type=int, default=500)

    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--max-initial-tasks", type=int, default=4)
    parser.add_argument("--max-researchers", type=int, default=4)
    parser.add_argument("--max-subtasks", type=int, default=16)
    parser.add_argument("--max-new-tasks-per-round", type=int, default=3)
    parser.add_argument("--max-react-steps", type=int, default=3)
    parser.add_argument("--max-queries-per-step", type=int, default=2)
    parser.add_argument("--max-tool-calls-per-subtask", type=int, default=18)
    parser.add_argument("--max-total-tool-calls", type=int, default=160)
    parser.add_argument("--max-total-searches", type=int, default=30)
    parser.add_argument("--max-total-tokens", type=int, default=1_000_000)
    parser.add_argument("--max-run-seconds", type=int, default=3600)
    parser.add_argument("--max-concurrent-readers", type=int, default=12)
    parser.add_argument("--min-total-claims", type=int, default=3)
    parser.add_argument("--min-coverage-ratio", type=float, default=0.6)
    parser.add_argument("--disable-citation-audit", action="store_true")
    parser.add_argument("--max-audit-rounds", type=int, default=2)
    parser.add_argument("--max-repair-tasks", type=int, default=3)

    parser.add_argument("--planner-max-tokens", type=int, default=3072)
    parser.add_argument("--researcher-max-tokens", type=int, default=1200)
    parser.add_argument("--reader-max-tokens", type=int, default=1536)
    parser.add_argument("--replan-max-tokens", type=int, default=3072)
    parser.add_argument("--report-max-tokens", type=int, default=8192)
    parser.add_argument("--auditor-max-tokens", type=int, default=3072)
    parser.add_argument("--source-content-max-chars", type=int, default=12000)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--context-safety-tokens", type=int, default=512)
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--max-concurrent-control-calls", type=int, default=8)
    parser.add_argument("--max-concurrent-long-calls", type=int, default=2)
    parser.add_argument("--max-concurrent-llm-calls-per-run", type=int, default=12)
    parser.add_argument("--max-inflight-llm-tokens", type=int, default=262_144)
    parser.add_argument("--disable-structured-outputs", action="store_true")
    parser.add_argument("--forward-vllm-priority", action="store_true")
    parser.add_argument("--enable-agent-thinking", action="store_true")
    return parser


def main() -> None:
    asyncio.run(run_async(build_parser().parse_args()))


if __name__ == "__main__":
    main()
