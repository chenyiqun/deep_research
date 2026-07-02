from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any

from .json_utils import safe_float
from .prompts import DIMENSIONS


def _criteria_weight_map(criteria_data: dict[str, Any]) -> dict[str, dict[str, float]]:
    weight_map: dict[str, dict[str, float]] = {}
    for dim in DIMENSIONS:
        items = criteria_data.get("criterions", {}).get(dim, [])
        weight_map[dim] = {
            str(item.get("criterion", "")).strip(): safe_float(item.get("weight"), 0.0)
            for item in items
            if item.get("criterion")
        }
    return weight_map


def _match_weight(criterion: str, dim_weight_map: dict[str, float]) -> float:
    criterion = criterion.strip()
    if criterion in dim_weight_map:
        return dim_weight_map[criterion]

    lower = criterion.lower()
    for key, weight in dim_weight_map.items():
        if key.lower() == lower:
            return weight

    for key, weight in dim_weight_map.items():
        key_lower = key.lower()
        if lower in key_lower or key_lower in lower:
            return weight

    if dim_weight_map:
        return sum(dim_weight_map.values()) / len(dim_weight_map)
    return 0.0


def calculate_weighted_scores(
    judge_output: dict[str, Any],
    criteria_data: dict[str, Any],
) -> dict[str, Any]:
    """Calculate weighted target/reference scores from judge JSON and rubric weights."""
    dimension_weights = {
        dim: safe_float(criteria_data.get("dimension_weight", {}).get(dim), 0.0)
        for dim in DIMENSIONS
    }
    criterion_weights = _criteria_weight_map(criteria_data)

    result = {
        "target": {"dims": {}, "total": 0.0},
        "reference": {"dims": {}, "total": 0.0},
    }

    for dim in DIMENSIONS:
        target_sum = 0.0
        reference_sum = 0.0
        total_weight = 0.0
        scores = judge_output.get(dim, [])
        if not isinstance(scores, list):
            scores = []

        for item in scores:
            if not isinstance(item, dict):
                continue
            criterion = str(item.get("criterion", "")).strip()
            weight = _match_weight(criterion, criterion_weights.get(dim, {}))
            if weight <= 0:
                continue
            target_score = safe_float(item.get("article_1_score"), None)  # type: ignore[arg-type]
            reference_score = safe_float(item.get("article_2_score"), None)  # type: ignore[arg-type]
            if target_score is None or reference_score is None:
                continue
            target_sum += target_score * weight
            reference_sum += reference_score * weight
            total_weight += weight

        target_avg = target_sum / total_weight if total_weight else 0.0
        reference_avg = reference_sum / total_weight if total_weight else 0.0

        result["target"]["dims"][f"{dim}_weighted_avg"] = target_avg
        result["reference"]["dims"][f"{dim}_weighted_avg"] = reference_avg
        result["target"]["total"] += target_avg * dimension_weights[dim]
        result["reference"]["total"] += reference_avg * dimension_weights[dim]

    return result


def normalize_pair_scores(weighted_scores: dict[str, Any]) -> dict[str, float]:
    """Return normalized RACE-style scores in 0..1."""
    normalized: dict[str, float] = {}
    for dim in DIMENSIONS:
        key = f"{dim}_weighted_avg"
        target = safe_float(weighted_scores["target"]["dims"].get(key), 0.0)
        reference = safe_float(weighted_scores["reference"]["dims"].get(key), 0.0)
        denom = target + reference
        normalized[dim] = target / denom if denom > 0 else 0.0

    target_total = safe_float(weighted_scores["target"].get("total"), 0.0)
    reference_total = safe_float(weighted_scores["reference"].get("total"), 0.0)
    denom = target_total + reference_total
    normalized["overall_score"] = target_total / denom if denom > 0 else 0.0
    return normalized


def summarize_race(results: list[dict[str, Any]]) -> dict[str, float]:
    valid = [row for row in results if not row.get("error")]
    summary = {"n": float(len(valid))}
    for key in [*DIMENSIONS, "overall_score"]:
        values = [safe_float(row.get(key), 0.0) for row in valid]
        summary[key] = mean(values) if values else 0.0
        summary[f"{key}_percent"] = summary[key] * 100.0
    return summary


def summarize_fact(rows: list[dict[str, Any]]) -> dict[str, float]:
    total_citations = 0
    total_supported = 0
    per_task_citations: list[int] = []
    per_task_supported: list[int] = []

    for row in rows:
        citations = 0
        supported = 0
        for citation in row.get("validated_citations", []):
            result = citation.get("result")
            if result == "unknown":
                continue
            citations += 1
            if result == "supported":
                supported += 1
        per_task_citations.append(citations)
        per_task_supported.append(supported)
        total_citations += citations
        total_supported += supported

    return {
        "n": float(len(rows)),
        "total_citations_per_task": mean(per_task_citations) if per_task_citations else 0.0,
        "effective_citations_per_task": mean(per_task_supported) if per_task_supported else 0.0,
        "citation_accuracy": total_supported / total_citations if total_citations else 0.0,
        "citation_accuracy_percent": (total_supported / total_citations * 100.0)
        if total_citations
        else 0.0,
    }

