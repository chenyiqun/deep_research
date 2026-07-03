from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .async_llm_client import AsyncChatClient, AsyncChatConfig
from .deep_research_workflow import AsyncDeepResearchWorkflow, DeepResearchConfig
from .io_utils import existing_ids, filter_tasks, load_jsonl, prepare_output_file, write_jsonl, write_text
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
    try:
        result = await workflow.run(task)
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

    async with output_lock:
        write_jsonl(output_file, [row], append=True)
        progress.update(1)


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

    print(f"Async vLLM base URL: {args.llm_base_url}")
    print(f"Async vLLM model: {args.llm_model}")
    print(f"Web search endpoint: {args.web_search_endpoint}")
    print(f"Tasks: {len(tasks)}")
    print(f"Max concurrent tasks: {args.max_concurrent_tasks}")
    print(f"Max concurrent LLM calls: {args.max_concurrent_llm_calls}")
    print(f"Max rounds: {args.max_rounds}")
    print(f"Search top-k: {args.search_top_k}")

    task_semaphore = asyncio.Semaphore(args.max_concurrent_tasks)
    output_lock = asyncio.Lock()

    async with AsyncChatClient(llm_config) as llm, WebSearchClient(search_config) as search_client:
        workflow = AsyncDeepResearchWorkflow(llm=llm, search_client=search_client, config=workflow_config)
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
