from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .io_utils import existing_ids, filter_tasks, index_by_prompt, load_jsonl, write_jsonl, write_text
from .json_utils import extract_json
from .prompts import build_race_judge_prompt, format_criteria_for_judge
from .scoring import calculate_weighted_scores, normalize_pair_scores, summarize_race
from .vllm_chat import GenerationConfig, VLLMChatModel


DEFAULT_QWEN3_8B_PATH = "/mnt/tidal-alsh01/usr/chenyiqun/base_models/Qwen/Qwen3-8B"


def build_item_prompt(
    task: dict[str, Any],
    target_by_prompt: dict[str, dict[str, Any]],
    reference_by_prompt: dict[str, dict[str, Any]],
    criteria_by_prompt: dict[str, dict[str, Any]],
) -> tuple[str | None, str | None]:
    prompt = task["prompt"]
    target = target_by_prompt.get(prompt)
    reference = reference_by_prompt.get(prompt)
    criteria = criteria_by_prompt.get(prompt)
    if target is None:
        return None, "target article not found"
    if reference is None:
        return None, "reference article not found"
    if criteria is None:
        return None, "criteria not found"

    criteria_list = format_criteria_for_judge(criteria)
    judge_prompt = build_race_judge_prompt(
        task_prompt=prompt,
        article_1=target.get("article", ""),
        article_2=reference.get("article", ""),
        criteria_list=criteria_list,
        language=task.get("language", "en"),
    )
    return judge_prompt, None


def score_item(
    task: dict[str, Any],
    judge_response: str,
    criteria_by_prompt: dict[str, dict[str, Any]],
    save_judge_output: bool = False,
) -> dict[str, Any]:
    prompt = task["prompt"]
    criteria = criteria_by_prompt[prompt]
    parsed = extract_json(judge_response)
    if not isinstance(parsed, dict):
        raise ValueError("judge output JSON must be an object")
    weighted = calculate_weighted_scores(parsed, criteria)
    normalized = normalize_pair_scores(weighted)
    row = {
        "id": int(task["id"]),
        "prompt": prompt,
        "language": task.get("language"),
        "topic": task.get("topic"),
        **normalized,
        "target_total_raw": weighted["target"]["total"],
        "reference_total_raw": weighted["reference"]["total"],
    }
    if save_judge_output:
        row["judge_output"] = parsed
        row["judge_response_text"] = judge_response
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RACE-style LLM-as-judge scoring with Qwen via vLLM.")
    parser.add_argument("--query-file", required=True)
    parser.add_argument("--criteria-file", required=True)
    parser.add_argument("--target-file", required=True, help="Generated reports JSONL.")
    parser.add_argument("--reference-file", required=True, help="Reference reports JSONL.")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--summary-file", required=True)
    parser.add_argument("--judge-model", default=DEFAULT_QWEN3_8B_PATH)
    parser.add_argument("--only-lang", choices=["zh", "en"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-judge-output", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Qwen3 thinking mode. Default is off for cleaner parseable JSON.",
    )
    args = parser.parse_args()

    tasks = load_jsonl(args.query_file)
    skip_ids = existing_ids(args.output_file) if args.resume else set()
    tasks = filter_tasks(tasks, only_lang=args.only_lang, limit=args.limit, skip_ids=skip_ids)
    target_by_prompt = index_by_prompt(load_jsonl(args.target_file))
    reference_by_prompt = index_by_prompt(load_jsonl(args.reference_file))
    criteria_by_prompt = index_by_prompt(load_jsonl(args.criteria_file))

    if not tasks:
        print("No tasks to evaluate.")
        return

    print(f"Loading judge model: {args.judge_model}")
    model = VLLMChatModel(
        model_name=args.judge_model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        enable_thinking=args.enable_thinking,
    )
    gen_config = GenerationConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        strip_thinking=True,
    )

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    produced: list[dict[str, Any]] = []

    for start in tqdm(range(0, len(tasks), args.batch_size), desc="RACE judging"):
        batch = tasks[start : start + args.batch_size]
        prompts: list[str] = []
        prompt_tasks: list[dict[str, Any]] = []
        pre_errors: list[dict[str, Any]] = []

        for task in batch:
            judge_prompt, error = build_item_prompt(
                task,
                target_by_prompt,
                reference_by_prompt,
                criteria_by_prompt,
            )
            if error:
                pre_errors.append({"id": int(task["id"]), "prompt": task["prompt"], "error": error})
            else:
                prompts.append(judge_prompt or "")
                prompt_tasks.append(task)

        if pre_errors:
            write_jsonl(output_path, pre_errors, append=True)
            produced.extend(pre_errors)

        if not prompts:
            continue

        responses = model.generate_batch(prompts, config=gen_config)
        rows: list[dict[str, Any]] = []
        for task, response in zip(prompt_tasks, responses):
            try:
                rows.append(
                    score_item(
                        task,
                        response,
                        criteria_by_prompt,
                        save_judge_output=args.save_judge_output,
                    )
                )
            except Exception as exc:  # Keep the run moving; inspect failed rows later.
                rows.append(
                    {
                        "id": int(task["id"]),
                        "prompt": task["prompt"],
                        "error": str(exc),
                        "judge_response_text": response[:4000],
                    }
                )
        write_jsonl(output_path, rows, append=True)
        produced.extend(rows)

    all_rows = load_jsonl(output_path)
    summary = summarize_race(all_rows)
    write_text(args.summary_file, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
