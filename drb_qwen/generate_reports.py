from __future__ import annotations

import argparse
from typing import Any

from tqdm import tqdm

from .io_utils import existing_ids, filter_tasks, load_jsonl, prepare_output_file, write_jsonl
from .prompts import REPORT_SYSTEM_PROMPT, build_report_prompt
from .vllm_chat import GenerationConfig, VLLMChatModel


DEFAULT_QWEN3_8B_PATH = "/mnt/tidal-alsh01/usr/chenyiqun/base_models/Qwen/Qwen3-8B"


def batched(items: list[Any], batch_size: int) -> list[list[Any]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DeepResearch-style reports with Qwen via vLLM.")
    parser.add_argument("--query-file", required=True, help="Path to query.jsonl.")
    parser.add_argument("--output-file", required=True, help="Output JSONL report file.")
    parser.add_argument("--model", default=DEFAULT_QWEN3_8B_PATH, help="HF model id or local path.")
    parser.add_argument("--only-lang", choices=["zh", "en"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Skip IDs already present in output file.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--gpu-devices",
        default=None,
        help="Comma-separated single-node GPU IDs, e.g. 0,1,2,3,4,5,6,7. Sets CUDA_VISIBLE_DEVICES before vLLM loads.",
    )
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Qwen3 thinking mode. Default is off so reports do not include <think> blocks.",
    )
    args = parser.parse_args()

    tasks = load_jsonl(args.query_file)
    skip_ids = existing_ids(args.output_file) if args.resume else set()
    tasks = filter_tasks(tasks, only_lang=args.only_lang, limit=args.limit, skip_ids=skip_ids)

    if not tasks:
        print("No tasks to process.")
        return

    print(f"Loading generation model: {args.model}")
    if args.gpu_devices:
        print(f"CUDA_VISIBLE_DEVICES: {args.gpu_devices}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    model = VLLMChatModel(
        model_name=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        enable_thinking=args.enable_thinking,
        gpu_devices=args.gpu_devices,
    )
    gen_config = GenerationConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        strip_thinking=True,
    )

    output_path = prepare_output_file(args.output_file, resume=args.resume)

    for batch in tqdm(batched(tasks, args.batch_size), desc="Generating reports"):
        prompts = [build_report_prompt(task) for task in batch]
        articles = model.generate_batch(
            prompts,
            system_prompt=REPORT_SYSTEM_PROMPT,
            config=gen_config,
        )
        rows = []
        for task, article in zip(batch, articles):
            rows.append(
                {
                    "id": int(task["id"]),
                    "topic": task.get("topic"),
                    "language": task.get("language"),
                    "prompt": task["prompt"],
                    "article": article,
                    "model": args.model,
                }
            )
        write_jsonl(output_path, rows, append=True)

    print(f"Saved reports to {output_path}")


if __name__ == "__main__":
    main()
