from __future__ import annotations

import json
import re
from typing import Any


FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def strip_json_fence(text: str) -> str:
    text = text.strip()
    match = FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text


def _balanced_json_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    starts = [i for i, ch in enumerate(text) if ch in "[{"]
    for start in starts:
        stack: list[str] = []
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch in "[{":
                stack.append(ch)
            elif ch in "]}":
                if not stack:
                    break
                opener = stack.pop()
                if (opener, ch) not in {("[", "]"), ("{", "}")}:
                    break
                if not stack:
                    candidates.append(text[start : idx + 1])
                    break
    return candidates


def extract_json(text: str) -> Any:
    """Parse JSON from a model response, accepting fenced or prefixed JSON."""
    cleaned = strip_json_fence(text)
    for candidate in [cleaned, *_balanced_json_candidates(cleaned)]:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("Could not extract valid JSON from model response")


def ensure_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

