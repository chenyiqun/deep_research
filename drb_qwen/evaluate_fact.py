from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from tqdm import tqdm

from .io_utils import existing_ids, filter_tasks, index_by_prompt, load_jsonl, write_jsonl, write_text
from .json_utils import ensure_list, extract_json
from .prompts import build_fact_extract_prompt, build_fact_validate_prompt
from .scoring import summarize_fact
from .vllm_chat import GenerationConfig, VLLMChatModel


URL_RE = re.compile(r"https?://[^\s\])}>\"']+")


def clean_url(url: str) -> str:
    return url.strip().rstrip(".,;:)]}")


def fallback_extract_urls(article: str) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for match in URL_RE.finditer(article):
        url = clean_url(match.group(0))
        left = max(0, match.start() - 350)
        right = min(len(article), match.end() + 120)
        statement = article[left:right].replace("\n", " ").strip()
        items.append({"statement": statement, "url": url})
    return items


def sanitize_extracted(value: Any, article: str, max_citations: int) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for item in ensure_list(value):
        if not isinstance(item, dict):
            continue
        statement = str(item.get("statement", "")).strip()
        url = clean_url(str(item.get("url", "")).strip())
        if not statement or not url.startswith(("http://", "https://")):
            continue
        items.append({"statement": statement, "url": url})

    if not items:
        items = fallback_extract_urls(article)

    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        key = (item["url"], item["statement"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= max_citations:
            break
    return deduped


def fetch_url_text(url: str, timeout_s: int = 20, max_chars: int = 20000) -> str:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("Install requests to use FACT page fetching.") from exc

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return "scrape failed: unsupported URL scheme"

    try:
        response = requests.get(
            url,
            timeout=timeout_s,
            headers={"User-Agent": "drb-qwen-pipeline/0.1"},
        )
        response.raise_for_status()
        text = response.text
    except Exception as exc:
        return f"scrape failed: {exc}"

    text = re.sub(r"(?is)<script.*?</script>", " ", text)
    text = re.sub(r"(?is)<style.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def parse_validation_response(value: Any, citations: list[dict[str, str]]) -> list[dict[str, str]]:
    parsed = ensure_list(value)
    results: list[dict[str, str]] = []
    seen_idxs: set[int] = set()
    for item in parsed:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("idx")) - 1
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= len(citations):
            continue
        result = str(item.get("result", "unknown")).strip().lower()
        if result not in {"supported", "unsupported", "unknown"}:
            result = "unknown"
        results.append({**citations[idx], "result": result})
        seen_idxs.add(idx)

    for idx, citation in enumerate(citations):
        if idx not in seen_idxs:
            results.append({**citation, "result": "unknown"})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a simplified FACT citation evaluation with Qwen via vLLM.")
    parser.add_argument("--query-file", required=True)
    parser.add_argument("--reports-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--summary-file", required=True)
    parser.add_argument("--judge-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--only-lang", choices=["zh", "en"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-citations-per-task", type=int, default=80)
    parser.add_argument("--max-reference-chars", type=int, default=20000)
    parser.add_argument("--no-fetch-pages", action="store_true", help="Use URL snippets only; faster but less faithful.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    tasks = load_jsonl(args.query_file)
    reports_by_prompt = index_by_prompt(load_jsonl(args.reports_file))
    skip_ids = existing_ids(args.output_file) if args.resume else set()
    tasks = filter_tasks(tasks, only_lang=args.only_lang, limit=args.limit, skip_ids=skip_ids)

    if not tasks:
        print("No tasks to evaluate.")
        return

    print(f"Loading FACT judge/extractor model: {args.judge_model}")
    model = VLLMChatModel(
        model_name=args.judge_model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )
    gen_config = GenerationConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    produced: list[dict[str, Any]] = []

    for task in tqdm(tasks, desc="FACT evaluation"):
        report = reports_by_prompt.get(task["prompt"])
        if report is None:
            row = {"id": int(task["id"]), "prompt": task["prompt"], "error": "report not found"}
            write_jsonl(output_path, [row], append=True)
            produced.append(row)
            continue

        article = report.get("article", "")
        extract_prompt = build_fact_extract_prompt(article, task.get("language", "en"))
        extract_response = model.generate_batch([extract_prompt], config=gen_config)[0]
        try:
            extracted = extract_json(extract_response)
        except Exception:
            extracted = []
        citations = sanitize_extracted(extracted, article, args.max_citations_per_task)

        validated: list[dict[str, str]] = []
        for citation in citations:
            reference = citation["url"]
            if not args.no_fetch_pages:
                reference = fetch_url_text(citation["url"], max_chars=args.max_reference_chars)
            validate_prompt = build_fact_validate_prompt(
                reference=reference,
                statements=[citation["statement"]],
                language=task.get("language", "en"),
            )
            response = model.generate_batch([validate_prompt], config=gen_config)[0]
            try:
                parsed = extract_json(response)
                validated.extend(parse_validation_response(parsed, [citation]))
            except Exception:
                validated.append({**citation, "result": "unknown"})

        row = {
            "id": int(task["id"]),
            "prompt": task["prompt"],
            "language": task.get("language"),
            "topic": task.get("topic"),
            "num_extracted_citations": len(citations),
            "validated_citations": validated,
        }
        write_jsonl(output_path, [row], append=True)
        produced.append(row)

    all_rows = load_jsonl(output_path)
    summary = summarize_fact([row for row in all_rows if not row.get("error")])
    write_text(args.summary_file, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
