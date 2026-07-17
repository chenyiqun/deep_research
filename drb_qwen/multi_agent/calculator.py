from __future__ import annotations

import math
import re
from typing import Any

from .schemas import CalculationRecord, EvidenceRecord, stable_id


ALLOWED_OPERATIONS = {
    "ratio",
    "percentage",
    "difference",
    "percent_change",
    "sum",
    "cagr",
    "rank",
}
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?%?")


def execute_calculation_proposals(
    proposals: Any,
    *,
    evidence: dict[str, EvidenceRecord],
    subtask_id: str,
) -> tuple[list[CalculationRecord], list[str]]:
    """Validate evidence-linked numeric inputs and execute a small safe operation set."""

    if not isinstance(proposals, list):
        return [], []
    records: list[CalculationRecord] = []
    errors: list[str] = []
    for index, proposal in enumerate(proposals[:8], 1):
        if not isinstance(proposal, dict):
            errors.append(f"calculation {index}: proposal is not an object")
            continue
        operation = str(proposal.get("operation", "")).strip().lower()
        if operation not in ALLOWED_OPERATIONS:
            errors.append(f"calculation {index}: unsupported operation {operation or '<empty>'}")
            continue
        raw_inputs = proposal.get("inputs", [])
        if not isinstance(raw_inputs, list):
            errors.append(f"calculation {index}: inputs must be a list")
            continue
        validated_inputs: list[dict[str, Any]] = []
        for raw_input in raw_inputs[:16]:
            if not isinstance(raw_input, dict):
                continue
            evidence_id = str(raw_input.get("evidence_id", "")).strip()
            evidence_item = evidence.get(evidence_id)
            if evidence_item is None:
                errors.append(f"calculation {index}: unknown evidence_id {evidence_id or '<empty>'}")
                validated_inputs = []
                break
            value = parse_number(raw_input.get("value"))
            if value is None or not number_occurs_in_text(value, evidence_item.excerpt):
                errors.append(
                    f"calculation {index}: value {raw_input.get('value')!r} is not grounded in {evidence_id}"
                )
                validated_inputs = []
                break
            validated_inputs.append(
                {
                    "label": str(raw_input.get("label", "value"))[:200],
                    "value": value,
                    "evidence_id": evidence_id,
                }
            )
        if not validated_inputs:
            continue
        try:
            unit = str(
                proposal.get(
                    "unit",
                    "%"
                    if operation in {"percentage", "percent_change", "cagr"}
                    else ("rank" if operation == "rank" else ""),
                )
            )[:100]
            if operation == "ratio" and unit.strip().casefold() in {"%", "percent", "percentage"}:
                raise ValueError("use operation=percentage when the desired unit is percent")
            result, formula = calculate(
                operation,
                [float(item["value"]) for item in validated_inputs],
                periods=parse_number(proposal.get("periods")),
                direction=str(proposal.get("direction", "higher")),
            )
        except ValueError as exc:
            errors.append(f"calculation {index}: {exc}")
            continue
        if not math.isfinite(result):
            errors.append(f"calculation {index}: non-finite result")
            continue
        evidence_ids = list(dict.fromkeys(item["evidence_id"] for item in validated_inputs))
        scope = str(proposal.get("scope", "")).strip()[:1000]
        record = CalculationRecord(
            id=stable_id(
                "calc",
                subtask_id,
                operation,
                validated_inputs,
                proposal.get("periods"),
                proposal.get("direction"),
                unit,
                scope,
            ),
            subtask_id=subtask_id,
            operation=operation,
            inputs=validated_inputs,
            formula=formula,
            result=round(result, 8),
            unit=unit,
            period=str(proposal.get("period", proposal.get("periods", "")))[:200],
            scope=scope,
            assumptions=[
                str(value)[:500]
                for value in proposal.get("assumptions", [])
                if str(value).strip()
            ][:10]
            if isinstance(proposal.get("assumptions", []), list)
            else [],
            description=str(proposal.get("description", ""))[:1000],
            evidence_ids=evidence_ids,
            confidence="high" if scope and operation != "cagr" else "medium",
        )
        records.append(record)
    return records, errors


def calculate(
    operation: str,
    values: list[float],
    *,
    periods: float | None,
    direction: str = "higher",
) -> tuple[float, str]:
    if operation == "sum":
        if not values:
            raise ValueError("sum requires at least one input")
        return sum(values), " + ".join(str(value) for value in values)
    if operation == "rank":
        if len(values) < 2:
            raise ValueError("rank requires a target value followed by at least one comparison value")
        target = values[0]
        descending = str(direction).strip().lower() not in {"lower", "ascending", "asc"}
        better = sum(value > target for value in values[1:]) if descending else sum(
            value < target for value in values[1:]
        )
        order = "descending" if descending else "ascending"
        return float(1 + better), f"1 + count(values better than {target}, order={order})"
    if len(values) != 2:
        raise ValueError(f"{operation} requires exactly two inputs")
    first, second = values
    if operation == "ratio":
        if second == 0:
            raise ValueError("ratio denominator is zero")
        return first / second, f"{first} / {second}"
    if operation == "percentage":
        if second == 0:
            raise ValueError("percentage denominator is zero")
        return first / second * 100.0, f"({first} / {second}) * 100"
    if operation == "difference":
        return first - second, f"{first} - {second}"
    if operation == "percent_change":
        if first == 0:
            raise ValueError("percent_change baseline is zero")
        return (second - first) / abs(first) * 100.0, f"(({second} - {first}) / abs({first})) * 100"
    if operation == "cagr":
        if first <= 0 or second < 0 or periods is None or periods <= 0:
            raise ValueError("cagr requires positive start, non-negative end, and positive periods")
        return ((second / first) ** (1.0 / periods) - 1.0) * 100.0, (
            f"(({second} / {first}) ** (1 / {periods}) - 1) * 100"
        )
    raise ValueError(f"unsupported operation {operation}")


def parse_number(value: Any) -> float | None:
    text = ("" if value is None else str(value)).strip().replace(",", "")
    if text.endswith("%"):
        text = text[:-1]
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def number_occurs_in_text(value: float, text: str) -> bool:
    for token in NUMBER_RE.findall(str(text or "")):
        parsed = parse_number(token)
        if parsed is not None and math.isclose(parsed, value, rel_tol=1e-9, abs_tol=1e-9):
            return True
    return False
