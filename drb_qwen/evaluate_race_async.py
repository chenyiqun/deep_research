from __future__ import annotations

import argparse
import asyncio
import json
import re
from typing import Any

from tqdm import tqdm

from .async_llm_client import AsyncChatClient, AsyncChatConfig
from .evaluate_race import build_item_prompt, score_item
from .io_utils import (
    existing_ids,
    filter_tasks,
    index_by_prompt,
    load_jsonl,
    prepare_output_file,
    write_jsonl,
    write_text,
)
from .scoring import summarize_race


async def judge_task(
    task: dict[str, Any],
    llm: AsyncChatClient,
    target_by_prompt: dict[str, dict[str, Any]],
    reference_by_prompt: dict[str, dict[str, Any]],
    criteria_by_prompt: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    judge_prompt, error = build_item_prompt(
        task,
        target_by_prompt=target_by_prompt,
        reference_by_prompt=reference_by_prompt,
        criteria_by_prompt=criteria_by_prompt,
    )
    if error:
        return {"id": int(task["id"]), "prompt": task["prompt"], "error": error}

    try:
        response = await call_judge_with_context_retry(
            llm=llm,
            judge_prompt=judge_prompt or "",
            args=args,
        )
    except Exception as exc:
        return {
            "id": int(task["id"]),
            "prompt": task["prompt"],
            "error": str(exc),
            "judge_prompt_chars": len(judge_prompt or ""),
        }
    try:
        return score_item(
            task,
            response,
            criteria_by_prompt,
            save_judge_output=args.save_judge_output,
        )
    except Exception as exc:
        return {
            "id": int(task["id"]),
            "prompt": task["prompt"],
            "error": str(exc),
            "judge_response_text": response[:4000],
        }


async def call_judge_with_context_retry(
    llm: AsyncChatClient,
    judge_prompt: str,
    args: argparse.Namespace,
) -> str:
    max_tokens = args.max_tokens
    last_error: Exception | None = None
    for _ in range(args.context_retry_attempts + 1):
        try:
            return await llm.chat(
                judge_prompt,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=max_tokens,
            )
        except RuntimeError as exc:
            last_error = exc
            retry_tokens = infer_retry_max_tokens(
                str(exc),
                current_max_tokens=max_tokens,
                min_retry_max_tokens=args.min_retry_max_tokens,
                safety_tokens=args.context_safety_tokens,
            )
            if retry_tokens is None:
                raise
            max_tokens = retry_tokens
    raise RuntimeError(f"judge request failed after context retries: {last_error}") from last_error


def infer_retry_max_tokens(
    error_text: str,
    current_max_tokens: int,
    min_retry_max_tokens: int,
    safety_tokens: int,
) -> int | None:
    context_match = re.search(r"maximum context length is (\d+) tokens", error_text)
    input_match = re.search(r"prompt contains at least (\d+) input tokens", error_text)
    if not context_match or not input_match:
        return None
    context_tokens = int(context_match.group(1))
    input_tokens = int(input_match.group(1))
    retry_tokens = context_tokens - input_tokens - safety_tokens
    if retry_tokens < min_retry_max_tokens:
        retry_tokens = max(1, retry_tokens)
    if retry_tokens <= 0 or retry_tokens >= current_max_tokens:
        return None
    return retry_tokens


async def run_async(args: argparse.Namespace) -> None:
    tasks = load_jsonl(args.query_file)
    skip_ids = existing_ids(args.output_file) if args.resume else set()
    tasks = filter_tasks(tasks, only_lang=args.only_lang, limit=args.limit, skip_ids=skip_ids)
    target_by_prompt = index_by_prompt(load_jsonl(args.target_file))
    reference_by_prompt = index_by_prompt(load_jsonl(args.reference_file))
    criteria_by_prompt = index_by_prompt(load_jsonl(args.criteria_file))

    if not tasks:
        print("No tasks to evaluate.")
        return

    output_path = prepare_output_file(args.output_file, resume=args.resume)
    output_lock = asyncio.Lock()
    task_semaphore = asyncio.Semaphore(args.max_concurrent_tasks)

    llm_config = AsyncChatConfig(
        base_url=args.llm_base_url,
        model=args.judge_model,
        api_key=args.llm_api_key,
        timeout_s=args.llm_timeout_s,
        max_concurrent_requests=args.max_concurrent_llm_calls,
        max_retries=args.llm_max_retries,
    )

    print(f"Async judge vLLM base URL: {args.llm_base_url}")
    print(f"Async judge model: {args.judge_model}")
    print(f"Tasks: {len(tasks)}")
    print(f"Max concurrent judge tasks: {args.max_concurrent_tasks}")
    print(f"Max concurrent LLM calls: {args.max_concurrent_llm_calls}")
    print(f"Judge max tokens: {args.max_tokens}")
    print(f"Context retry attempts: {args.context_retry_attempts}")

    async with AsyncChatClient(llm_config) as llm:
        progress = tqdm(total=len(tasks), desc="Async RACE judging")

        async def guarded_judge(task: dict[str, Any]) -> None:
            async with task_semaphore:
                row = await judge_task(
                    task,
                    llm=llm,
                    target_by_prompt=target_by_prompt,
                    reference_by_prompt=reference_by_prompt,
                    criteria_by_prompt=criteria_by_prompt,
                    args=args,
                )
                async with output_lock:
                    write_jsonl(output_path, [row], append=True)
                    progress.update(1)

        await asyncio.gather(*(guarded_judge(task) for task in tasks))
        progress.close()

    all_rows = load_jsonl(output_path)
    summary = summarize_race(all_rows)
    write_text(args.summary_file, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run async RACE-style LLM-as-judge scoring through vLLM serve."
    )
    parser.add_argument("--query-file", required=True)
    parser.add_argument("--criteria-file", required=True)
    parser.add_argument("--target-file", required=True)
    parser.add_argument("--reference-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--summary-file", required=True)
    parser.add_argument("--judge-model", default="qwen3-32b")
    parser.add_argument("--only-lang", choices=["zh", "en"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-judge-output", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--context-retry-attempts", type=int, default=2)
    parser.add_argument("--context-safety-tokens", type=int, default=256)
    parser.add_argument("--min-retry-max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)

    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--llm-api-key", default="EMPTY")
    parser.add_argument("--llm-timeout-s", type=int, default=600)
    parser.add_argument("--llm-max-retries", type=int, default=3)
    parser.add_argument("--max-concurrent-llm-calls", type=int, default=16)
    parser.add_argument("--max-concurrent-tasks", type=int, default=8)
    args = parser.parse_args()
    asyncio.run(run_async(args))


if __name__ == "__main__":
    main()
