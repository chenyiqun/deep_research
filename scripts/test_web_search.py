from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
from typing import Any

# Allow `python scripts/test_web_search.py ...` from the repository root
# without requiring a package installation or an explicit PYTHONPATH.
REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from drb_qwen.io_utils import write_text
from drb_qwen.web_search import (
    DEFAULT_SEARCH_ENGINE,
    PROD_WEB_SEARCH_ENDPOINT,
    SUPPORTED_SEARCH_ENGINES,
    WebSearchClient,
    WebSearchConfig,
    get_search_engine_profile,
)


async def search_one_engine(
    args: argparse.Namespace,
    engine: str,
    access_key: str,
) -> dict[str, Any]:
    profile = get_search_engine_profile(engine)
    config = WebSearchConfig(
        endpoint=args.web_search_endpoint,
        access_key=access_key,
        search_engine=engine,
        count=args.search_count,
        search_domain_filter=args.search_domain_filter,
        search_recency_filter=args.search_recency_filter,
        content_size=args.search_content_size,
        timeout_s=args.timeout_s,
        max_concurrent_requests=1,
        max_retries=args.max_retries,
    )
    try:
        async with WebSearchClient(config) as client:
            results = await client.search(args.query, top_k=args.search_top_k)
        return {
            "ok": bool(results),
            "error": "" if results else "search returned no results",
            "query": args.query,
            "search_engine": profile.key,
            "provider": profile.provider,
            "api_engine_candidates": list(profile.api_engines),
            "native_content": profile.native_content,
            "fetch_pages_by_default": profile.fetch_pages_by_default,
            "num_results": len(results),
            "results": [result.to_dict() for result in results],
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "query": args.query,
            "search_engine": profile.key,
            "provider": profile.provider,
            "api_engine_candidates": list(profile.api_engines),
            "native_content": profile.native_content,
            "fetch_pages_by_default": profile.fetch_pages_by_default,
            "num_results": 0,
            "results": [],
        }


async def run_async(args: argparse.Namespace) -> int:
    access_key = args.web_search_api_key or os.environ.get(args.web_search_api_key_env, "")
    if not access_key:
        raise ValueError(
            f"Missing search key. Set {args.web_search_api_key_env} or pass --web-search-api-key."
        )

    engines = list(SUPPORTED_SEARCH_ENGINES) if args.all_engines else [args.search_engine]
    payloads = await asyncio.gather(
        *(search_one_engine(args, engine, access_key) for engine in engines)
    )

    output = {
        "query": args.query,
        "endpoint": args.web_search_endpoint,
        "tested_engines": engines,
        "all_ok": all(payload["ok"] for payload in payloads),
        "engines": payloads,
    }
    if args.output_file:
        write_text(
            Path(args.output_file),
            json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        )

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print_human_readable(output, max_content_chars=args.max_content_chars)
        if args.output_file:
            print(f"\nFull JSON saved to: {args.output_file}")
    return 0 if output["all_ok"] else 1


def print_human_readable(output: dict[str, Any], max_content_chars: int) -> None:
    print(f"Query: {output['query']}")
    print(f"Endpoint: {output['endpoint']}")
    for payload in output["engines"]:
        print("\n" + "=" * 80)
        print(
            f"Engine: {payload['search_engine']} ({payload['provider']}) | "
            f"native_content={payload['native_content']} | "
            f"default_page_fetch={payload['fetch_pages_by_default']}"
        )
        print(f"API engine candidates: {', '.join(payload['api_engine_candidates'])}")
        print(f"Status: {'OK' if payload['ok'] else 'FAILED'} | results={payload['num_results']}")
        if payload["error"]:
            print(f"Error: {payload['error']}")
        for index, result in enumerate(payload["results"], start=1):
            content = str(result.get("content") or "")
            preview = content[:max_content_chars]
            if len(content) > max_content_chars:
                preview += "\n...[truncated]"
            print(f"\n[{index}] {result.get('title') or '<untitled>'}")
            print(f"URL: {result.get('link') or '<missing>'}")
            print(f"Published: {result.get('publish_date') or '<unknown>'}")
            print(
                f"Content: chars={len(content)} kind={result.get('content_kind') or '-'} "
                f"quality={result.get('source_quality') or '-'} "
                f"method={result.get('extraction_method') or '-'} "
                f"api_engine={result.get('api_search_engine') or '-'}"
            )
            print(preview or "<empty content>")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test the configured web-search endpoint without vLLM or URL page fetching."
    )
    parser.add_argument("--query", required=True, help="Search query to send.")
    parser.add_argument(
        "--search-engine",
        default=DEFAULT_SEARCH_ENGINE,
        choices=SUPPORTED_SEARCH_ENGINES,
    )
    parser.add_argument(
        "--all-engines",
        action="store_true",
        help="Test all six supported engines concurrently instead of only --search-engine.",
    )
    parser.add_argument("--web-search-endpoint", default=PROD_WEB_SEARCH_ENDPOINT)
    parser.add_argument("--web-search-api-key-env", default="WEB_SEARCH_API_KEY")
    parser.add_argument("--web-search-api-key", default="")
    parser.add_argument("--search-count", type=int, default=10)
    parser.add_argument("--search-top-k", type=int, default=5)
    parser.add_argument("--search-domain-filter", default="")
    parser.add_argument("--search-recency-filter", default="noLimit")
    parser.add_argument("--search-content-size", default="high")
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-content-chars", type=int, default=1200)
    parser.add_argument("--output-file", default="")
    parser.add_argument("--json", action="store_true", help="Print the complete JSON response summary.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.search_top_k <= 0 or args.search_count <= 0 or args.max_content_chars <= 0:
        raise ValueError("search-count, search-top-k, and max-content-chars must be positive")
    try:
        exit_code = asyncio.run(run_async(args))
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
