#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from drb_qwen.io_utils import load_jsonl
from drb_qwen.scoring import DIMENSIONS, summarize_race


def valid_by_id(path: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        task_id = str(row.get("id", "")).strip()
        if task_id and not row.get("error"):
            output[task_id] = row
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two RACE judge JSONL files on their common valid task IDs."
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output-file", default="")
    args = parser.parse_args()

    baseline = valid_by_id(args.baseline)
    candidate = valid_by_id(args.candidate)
    shared_ids = sorted(set(baseline) & set(candidate), key=lambda value: (not value.isdigit(), value))
    baseline_summary = summarize_race([baseline[task_id] for task_id in shared_ids])
    candidate_summary = summarize_race([candidate[task_id] for task_id in shared_ids])
    keys = [*DIMENSIONS, "overall_score"]
    result = {
        "baseline_file": str(Path(args.baseline)),
        "candidate_file": str(Path(args.candidate)),
        "baseline_valid_ids": len(baseline),
        "candidate_valid_ids": len(candidate),
        "shared_valid_ids": len(shared_ids),
        "shared_ids": shared_ids,
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "delta": {
            key: candidate_summary.get(key, 0.0) - baseline_summary.get(key, 0.0)
            for key in keys
        },
    }
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output_file:
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_file).write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
