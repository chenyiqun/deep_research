from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]], append: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def prepare_output_file(path: str | Path, resume: bool = False) -> Path:
    """Create the output directory and restart the file unless resuming."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not resume:
        path.unlink()
    return path


def index_by_prompt(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["prompt"]): row for row in rows if "prompt" in row}


def existing_ids(path: str | Path) -> set[int]:
    path = Path(path)
    if not path.exists():
        return set()
    ids: set[int] = set()
    for row in load_jsonl(path):
        if "id" in row:
            try:
                ids.add(int(row["id"]))
            except (TypeError, ValueError):
                continue
    return ids


def filter_tasks(
    tasks: list[dict[str, Any]],
    only_lang: str | None = None,
    limit: int | None = None,
    skip_ids: set[int] | None = None,
) -> list[dict[str, Any]]:
    skip_ids = skip_ids or set()
    filtered: list[dict[str, Any]] = []
    for task in tasks:
        if only_lang and task.get("language") != only_lang:
            continue
        try:
            task_id = int(task["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if task_id in skip_ids:
            continue
        filtered.append(task)
        if limit is not None and len(filtered) >= limit:
            break
    return filtered
