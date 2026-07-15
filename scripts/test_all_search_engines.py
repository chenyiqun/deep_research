from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from drb_qwen.io_utils import write_jsonl, write_text
from drb_qwen.web_search import (
    PROD_WEB_SEARCH_ENDPOINT,
    SUPPORTED_SEARCH_ENGINES,
    get_search_engine_profile,
    parse_search_results,
)


async def probe_api_engine(
    *,
    session: Any,
    endpoint: str,
    access_key: str,
    logical_engine: str,
    api_engine: str,
    query: str,
    count: int,
    output_dir: Path,
) -> dict[str, Any]:
    profile = get_search_engine_profile(logical_engine)
    request_payload = {
        "search_engine": api_engine,
        "search_query": query,
        "count": count,
    }
    headers = {
        "X-EdithAI-Access-Key": access_key,
        "Content-Type": "application/json",
    }
    started = time.monotonic()
    status: int | None = None
    raw_body = ""
    response_json: Any = None
    transport_error = ""
    try:
        async with session.post(endpoint, json=request_payload, headers=headers) as response:
            status = response.status
            raw_body = await response.text()
            try:
                response_json = json.loads(raw_body)
            except json.JSONDecodeError:
                response_json = None
    except Exception as exc:
        transport_error = f"{exc.__class__.__name__}: {exc}"

    elapsed_ms = int((time.monotonic() - started) * 1000)
    normalized_results = []
    if isinstance(response_json, (dict, list)):
        normalized_results = [
            result.to_dict()
            for result in parse_search_results(
                response_json,
                query,
                search_engine=logical_engine,
                api_search_engine=api_engine,
            )
        ]

    http_ok = status is not None and 200 <= status < 300
    record = {
        "logical_engine": logical_engine,
        "provider": profile.provider,
        "api_engine": api_engine,
        "native_content": profile.native_content,
        "request_payload": request_payload,
        "http_status": status,
        "http_ok": http_ok,
        "has_results": bool(normalized_results),
        "usable": bool(http_ok and normalized_results),
        "num_results": len(normalized_results),
        "elapsed_ms": elapsed_ms,
        "transport_error": transport_error,
        "raw_body": raw_body,
        "response_json": response_json,
        "normalized_results": normalized_results,
    }
    raw_path = output_dir / "raw" / f"{logical_engine}__{api_engine}.json"
    write_text(raw_path, json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    record["raw_file"] = str(raw_path)
    return record


async def run_async(args: argparse.Namespace) -> int:
    access_key = args.web_search_api_key or os.environ.get(args.web_search_api_key_env, "")
    if not access_key:
        raise ValueError(
            f"Missing search key. Set {args.web_search_api_key_env} or pass --web-search-api-key."
        )

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import aiohttp
    except ImportError as exc:
        raise RuntimeError("Install aiohttp to run the search-engine probe.") from exc

    timeout = aiohttp.ClientTimeout(total=args.timeout_s + 5)
    attempts: list[dict[str, Any]] = []
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for logical_engine in SUPPORTED_SEARCH_ENGINES:
            profile = get_search_engine_profile(logical_engine)
            for api_engine in profile.api_engines:
                print(f"Testing {logical_engine} ({profile.provider}) -> {api_engine} ...", flush=True)
                record = await probe_api_engine(
                    session=session,
                    endpoint=args.web_search_endpoint,
                    access_key=access_key,
                    logical_engine=logical_engine,
                    api_engine=api_engine,
                    query=args.query,
                    count=args.search_count,
                    output_dir=output_dir,
                )
                attempts.append(record)
                print(format_attempt(record), flush=True)

    engines: list[dict[str, Any]] = []
    for logical_engine in SUPPORTED_SEARCH_ENGINES:
        profile = get_search_engine_profile(logical_engine)
        engine_attempts = [item for item in attempts if item["logical_engine"] == logical_engine]
        usable_attempt = next((item for item in engine_attempts if item["usable"]), None)
        engines.append(
            {
                "logical_engine": logical_engine,
                "provider": profile.provider,
                "native_content": profile.native_content,
                "api_engine_candidates": list(profile.api_engines),
                "usable": usable_attempt is not None,
                "selected_api_engine": usable_attempt["api_engine"] if usable_attempt else "",
                "num_results": usable_attempt["num_results"] if usable_attempt else 0,
                "attempts": [summarize_attempt(item) for item in engine_attempts],
            }
        )

    summary = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "query": args.query,
        "endpoint": args.web_search_endpoint,
        "search_count": args.search_count,
        "usable_engines": [item["logical_engine"] for item in engines if item["usable"]],
        "num_usable_engines": sum(1 for item in engines if item["usable"]),
        "engines": engines,
    }
    summary_path = output_dir / "summary.json"
    attempts_path = output_dir / "attempts.jsonl"
    write_text(summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    write_jsonl(attempts_path, attempts)

    print("\n" + "=" * 88)
    print("Search-engine probe summary")
    for engine in engines:
        status = "USABLE" if engine["usable"] else "UNAVAILABLE"
        selected = engine["selected_api_engine"] or "-"
        print(
            f"{engine['logical_engine']:<18} {engine['provider']:<8} {status:<11} "
            f"api_engine={selected:<18} results={engine['num_results']}"
        )
    print(f"\nSummary: {summary_path}")
    print(f"All attempts: {attempts_path}")
    print(f"Raw per-request files: {output_dir / 'raw'}")
    return 0


def summarize_attempt(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "api_engine": record["api_engine"],
        "http_status": record["http_status"],
        "http_ok": record["http_ok"],
        "has_results": record["has_results"],
        "usable": record["usable"],
        "num_results": record["num_results"],
        "elapsed_ms": record["elapsed_ms"],
        "transport_error": record["transport_error"],
        "raw_file": record["raw_file"],
        "error": extract_error(record),
    }


def extract_error(record: dict[str, Any]) -> str:
    if record.get("transport_error"):
        return str(record["transport_error"])
    response_json = record.get("response_json")
    if isinstance(response_json, dict):
        error = response_json.get("error")
        if isinstance(error, dict):
            code = str(error.get("code") or "")
            message = str(error.get("message") or "")
            return f"{code}: {message}".strip(": ")
        if error:
            return str(error)
    if not record.get("http_ok"):
        return str(record.get("raw_body") or "")[:500]
    if not record.get("has_results"):
        return "request succeeded but returned no parsed results"
    return ""


def format_attempt(record: dict[str, Any]) -> str:
    return (
        f"  status={record['http_status']} usable={record['usable']} "
        f"results={record['num_results']} elapsed_ms={record['elapsed_ms']} "
        f"error={extract_error(record) or '-'}"
    )


def default_output_dir() -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / "search_engine_probe" / run_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe all supported logical/API search-engine names and save raw responses."
    )
    parser.add_argument("--query", required=True)
    parser.add_argument("--web-search-endpoint", default=PROD_WEB_SEARCH_ENDPOINT)
    parser.add_argument("--web-search-api-key-env", default="WEB_SEARCH_API_KEY")
    parser.add_argument("--web-search-api-key", default="")
    parser.add_argument("--search-count", type=int, default=10)
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--output-dir", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.search_count <= 0 or args.timeout_s <= 0:
        raise SystemExit("ERROR: --search-count and --timeout-s must be positive")
    try:
        exit_code = asyncio.run(run_async(args))
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
