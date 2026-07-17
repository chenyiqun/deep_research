from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schemas import (
    ClaimRecord,
    GlobalResearchState,
    ResearchExecutionBundle,
    TaskStatus,
    normalize_text,
    texts_semantically_equivalent,
    utc_now,
)
from .security import validate_external_url


CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}


@dataclass
class ResearchGateDecision:
    passed: bool
    partial: bool
    reason: str
    coverage_ratio: float


def merge_research_bundle(state: GlobalResearchState, bundle: ResearchExecutionBundle) -> dict[str, Any]:
    result = bundle.result
    task = state.tasks.get(result.subtask_id)
    if task is None:
        raise ValueError(f"AgentResult references unknown task {result.subtask_id}")

    rejected_sources = 0
    rejected_evidence = 0
    rejected_claims = 0
    rejected_calculations = 0
    added_sources = 0
    added_evidence = 0
    added_claims = 0
    accepted_source_ids: set[str] = set()
    for source in bundle.sources:
        url_allowed, _ = validate_external_url(source.url)
        if not source.id or not url_allowed:
            rejected_sources += 1
            continue
        accepted_source_ids.add(source.id)
        if source.id not in state.sources:
            state.sources[source.id] = source
            added_sources += 1

    accepted_evidence_ids: set[str] = set()
    for evidence in bundle.evidence:
        if (
            not evidence.id
            or evidence.source_id not in accepted_source_ids
            or evidence.subtask_id != task.id
            or evidence.relation not in {"supports", "refutes", "qualifies"}
            or not evidence.claim_text.strip()
            or not evidence.excerpt.strip()
        ):
            rejected_evidence += 1
            continue
        accepted_evidence_ids.add(evidence.id)
        if evidence.id not in state.evidence:
            state.evidence[evidence.id] = evidence
            added_evidence += 1

    accepted_claim_ids: set[str] = set()
    claim_id_map: dict[str, str] = {}
    for claim in bundle.claims:
        valid_evidence_ids = [
            evidence_id for evidence_id in claim.evidence_ids if evidence_id in accepted_evidence_ids
        ]
        if not claim.id or claim.subtask_id != task.id or not claim.text.strip() or not valid_evidence_ids:
            rejected_claims += 1
            continue
        claim.evidence_ids = valid_evidence_ids
        existing = state.claims.get(claim.id)
        if existing is None:
            existing = next(
                (
                    value
                    for value in state.claims.values()
                    if value.subtask_id == claim.subtask_id
                    and (
                        texts_semantically_equivalent(value.text, claim.text, threshold=0.9)
                        or same_claim_dimensions(value, claim)
                    )
                ),
                None,
            )
        if existing is None:
            state.claims[claim.id] = claim
            canonical_id = claim.id
            added_claims += 1
        else:
            merge_claim(existing, claim)
            canonical_id = existing.id
        accepted_claim_ids.add(canonical_id)
        claim_id_map[claim.id] = canonical_id
    refresh_claim_statuses(state)

    accepted_calculation_ids: set[str] = set()
    for calculation in bundle.calculations:
        if (
            not calculation.id
            or calculation.subtask_id != task.id
            or not calculation.evidence_ids
            or any(evidence_id not in accepted_evidence_ids for evidence_id in calculation.evidence_ids)
        ):
            rejected_calculations += 1
            continue
        state.calculations[calculation.id] = calculation
        accepted_calculation_ids.add(calculation.id)

    result.source_ids = [source_id for source_id in result.source_ids if source_id in accepted_source_ids]
    result.evidence_ids = [
        evidence_id for evidence_id in result.evidence_ids if evidence_id in accepted_evidence_ids
    ]
    result.claim_ids = list(
        dict.fromkeys(
            claim_id_map[claim_id]
            for claim_id in result.claim_ids
            if claim_id in claim_id_map and claim_id_map[claim_id] in accepted_claim_ids
        )
    )
    result.calculation_ids = [
        calculation_id
        for calculation_id in result.calculation_ids
        if calculation_id in accepted_calculation_ids
    ]
    # REFINE_TASK starts a fresh local execution but keeps the task's accepted
    # evidence history. Rebuild AgentResult as the cumulative task dossier so
    # a later verification pass does not lose a source requirement already met.
    historical_claim_ids = [
        claim.id for claim in state.claims.values() if claim.subtask_id == task.id
    ]
    append_unique_strings(result.claim_ids, historical_claim_ids)
    historical_evidence_ids = [
        evidence_id
        for claim_id in result.claim_ids
        if claim_id in state.claims
        for evidence_id in state.claims[claim_id].evidence_ids
        if evidence_id in state.evidence
    ]
    append_unique_strings(result.evidence_ids, historical_evidence_ids)
    historical_source_ids = [
        state.evidence[evidence_id].source_id
        for evidence_id in result.evidence_ids
        if evidence_id in state.evidence
        and state.evidence[evidence_id].source_id in state.sources
    ]
    append_unique_strings(result.source_ids, historical_source_ids)
    append_unique_strings(
        result.calculation_ids,
        [
            calculation.id
            for calculation in state.calculations.values()
            if calculation.subtask_id == task.id
        ],
    )
    if result.status == TaskStatus.COMPLETED and not result.claim_ids:
        result.status = TaskStatus.PARTIAL
        append_unique_strings(
            result.unresolved_gaps,
            ["Subtask produced no valid source-grounded claims after integrity checks."],
        )

    missing_source_types = source_requirement_gaps(state, task, result.source_ids)
    if missing_source_types:
        append_semantically_unique_strings(result.unresolved_gaps, missing_source_types, max_items=40)
        if result.status == TaskStatus.COMPLETED:
            result.status = TaskStatus.PARTIAL

    task.status = result.status
    task.result_summary = result.answer_summary
    task.error = result.error
    task.attempts += 1
    task.updated_at = utc_now()
    state.agent_results[result.subtask_id] = result
    append_unique_strings(state.query_ledger, bundle.local_state.queries)
    state.query_ledger = state.query_ledger[-256:]
    task_queries = state.query_ledger_by_task.setdefault(result.subtask_id, [])
    append_unique_strings(task_queries, bundle.local_state.queries)
    state.query_ledger_by_task[result.subtask_id] = task_queries[-64:]
    if result.resolved_gaps:
        state.gaps = [
            gap
            for gap in state.gaps
            if not any(texts_semantically_equivalent(gap, resolved) for resolved in result.resolved_gaps)
        ]
    append_semantically_unique_strings(state.gaps, result.unresolved_gaps, max_items=100)
    append_unique_dicts(state.conflicts, result.conflicts)
    state.budget.add(result.usage)
    if result.status == TaskStatus.COMPLETED:
        state.budget.completed_subtasks += 1
    elif result.status == TaskStatus.FAILED:
        state.budget.failed_subtasks += 1

    update_coverage(state, task.id)
    state.bump_version()
    return {
        "subtask_id": task.id,
        "status": task.status.value,
        "new_sources": added_sources,
        "new_evidence": added_evidence,
        "new_claims": added_claims,
        "rejected_sources": rejected_sources,
        "rejected_evidence": rejected_evidence,
        "rejected_claims": rejected_claims,
        "new_calculations": len(accepted_calculation_ids),
        "rejected_calculations": rejected_calculations,
        "gaps": result.unresolved_gaps,
        "conflicts": result.conflicts,
    }


def merge_claim(existing: ClaimRecord, incoming: ClaimRecord) -> None:
    append_unique_strings(existing.evidence_ids, incoming.evidence_ids)
    append_unique_strings(existing.qualifiers, incoming.qualifiers)
    append_unique_strings(existing.required_source_types, incoming.required_source_types)
    for key, value in incoming.dimensions.items():
        existing.dimensions.setdefault(key, value)
    if CONFIDENCE_ORDER.get(incoming.confidence, 0) > CONFIDENCE_ORDER.get(existing.confidence, 0):
        existing.confidence = incoming.confidence
    if existing.status == "provisional" and incoming.status != "provisional":
        existing.status = incoming.status
    existing.updated_at = utc_now()


def refresh_claim_statuses(state: GlobalResearchState) -> None:
    for claim in state.claims.values():
        supporting_groups: set[str] = set()
        qualifying_groups: set[str] = set()
        refuting_groups: set[str] = set()
        for evidence_id in claim.evidence_ids:
            evidence = state.evidence.get(evidence_id)
            source = state.sources.get(evidence.source_id) if evidence else None
            if not source or not source.independence_group or evidence is None:
                continue
            if evidence.relation == "refutes":
                refuting_groups.add(source.independence_group)
            elif evidence.relation == "qualifies":
                qualifying_groups.add(source.independence_group)
            else:
                supporting_groups.add(source.independence_group)
        positive_groups = supporting_groups | qualifying_groups
        if positive_groups and refuting_groups:
            claim.status = "contested"
        elif len(positive_groups) >= 2:
            claim.status = "corroborated"
        elif supporting_groups:
            claim.status = "single_source"
        elif qualifying_groups:
            claim.status = "qualified"
        elif refuting_groups:
            claim.status = "disputed"
        claim.missing_source_types = claim_source_requirement_gaps(state, claim)
        if claim.missing_source_types and claim.status in {
            "corroborated",
            "single_source",
            "qualified",
            "provisional",
        }:
            claim.status = "source_incomplete"


def update_coverage(state: GlobalResearchState, subtask_id: str) -> None:
    task = state.tasks[subtask_id]
    result = state.agent_results.get(subtask_id)
    for target in task.coverage_targets:
        target_tasks = [
            candidate
            for candidate in state.tasks.values()
            if target in candidate.coverage_targets
        ]
        target_has_claims = any(
            state.agent_results.get(candidate.id)
            and state.agent_results[candidate.id].claim_ids
            for candidate in target_tasks
        )
        target_claim_ids = list(
            dict.fromkeys(
                claim_id
                for candidate in target_tasks
                for claim_id in (
                    state.agent_results[candidate.id].claim_ids
                    if candidate.id in state.agent_results
                    else []
                )
                if claim_id in state.claims
            )
        )
        target_claims = [state.claims[claim_id] for claim_id in target_claim_ids]
        target_quality_score = task_evidence_quality_score(state, target_claims)
        if any(task_has_covered_evidence(state, candidate) for candidate in target_tasks):
            state.coverage[target] = "covered"
        elif target_has_claims:
            state.coverage[target] = "partial"
        else:
            state.coverage[target] = "missing"
        detail = state.coverage_details.setdefault(
            target,
            {
                "status": "missing",
                "task_ids": [],
                "completed_task_ids": [],
                "claim_ids": [],
                "source_types": [],
                "high_authority_source_ids": [],
                "missing_source_requirements": [],
                "missing_claim_source_requirements": [],
                "quality_score": 0.0,
            },
        )
        # Resumed runs may contain an older/partial coverage_details shape.
        # Populate every collection before merging new bundle data.
        for key in (
            "task_ids",
            "completed_task_ids",
            "claim_ids",
            "source_types",
            "high_authority_source_ids",
            "missing_source_requirements",
            "missing_claim_source_requirements",
        ):
            if not isinstance(detail.get(key), list):
                detail[key] = []
        append_unique_strings(detail["task_ids"], [task.id])
        if task.status == TaskStatus.COMPLETED:
            append_unique_strings(detail["completed_task_ids"], [task.id])
        if result:
            append_unique_strings(detail["claim_ids"], result.claim_ids)
            source_types: list[str] = []
            high_authority_ids: list[str] = []
            for source_id in result.source_ids:
                source = state.sources.get(source_id)
                if not source:
                    continue
                source_types.append(source.source_type)
                if source.authority_score >= 0.8:
                    high_authority_ids.append(source.id)
            append_unique_strings(detail["source_types"], source_types)
            append_unique_strings(detail["high_authority_source_ids"], high_authority_ids)
            detail["missing_source_requirements"] = sorted(
                {
                    requirement
                    for candidate in target_tasks
                    for requirement in source_requirement_gaps(
                        state,
                        candidate,
                        state.agent_results[candidate.id].source_ids
                        if candidate.id in state.agent_results
                        else [],
                        descriptions=False,
                    )
                }
            )
            missing_claim_requirements = sorted(
                {
                    requirement
                    for claim in target_claims
                    for requirement in claim.missing_source_types
                }
            )
            detail["missing_claim_source_requirements"] = missing_claim_requirements
            detail["quality_score"] = target_quality_score
        detail["status"] = state.coverage[target]
        detail["claim_count"] = len(detail["claim_ids"])


def initialize_coverage(state: GlobalResearchState) -> None:
    targets: list[str] = []
    if state.brief:
        targets.extend(state.brief.coverage_targets)
    for task in state.tasks.values():
        targets.extend(task.coverage_targets)
    for target in targets:
        state.coverage.setdefault(target, "missing")
        state.coverage_details.setdefault(
            target,
            {
                "status": state.coverage[target],
                "task_ids": [],
                "completed_task_ids": [],
                "claim_ids": [],
                "source_types": [],
                "high_authority_source_ids": [],
                "missing_source_requirements": [],
                "missing_claim_source_requirements": [],
                "claim_count": 0,
                "quality_score": 0.0,
            },
        )
    for task in state.tasks.values():
        for target in task.coverage_targets:
            detail = state.coverage_details.setdefault(target, {"status": state.coverage.get(target, "missing")})
            detail.setdefault("task_ids", [])
            append_unique_strings(detail["task_ids"], [task.id])


def evaluate_research_gate(
    state: GlobalResearchState,
    *,
    min_total_claims: int,
    min_coverage_ratio: float,
    budget_exhausted: bool,
) -> ResearchGateDecision:
    targets = list(state.coverage.items())
    target_weights: list[float] = []
    for target, status in targets:
        if status == "covered":
            target_weights.append(1.0)
        elif status == "partial":
            detail = state.coverage_details.get(target, {})
            score = float(detail.get("quality_score", 0.0) or 0.0)
            target_weights.append(max(0.35, min(0.85, score)))
        else:
            target_weights.append(0.0)
    ratio = sum(target_weights) / len(target_weights) if target_weights else (1.0 if state.claims else 0.0)
    enough_claims = len(state.claims) >= max(1, min_total_claims)
    if enough_claims and ratio >= min_coverage_ratio:
        return ResearchGateDecision(True, False, "claim and coverage thresholds reached", ratio)
    if budget_exhausted:
        return ResearchGateDecision(False, True, "budget or planning limit exhausted", ratio)
    if not enough_claims:
        return ResearchGateDecision(False, False, f"only {len(state.claims)} claims available", ratio)
    return ResearchGateDecision(False, False, f"coverage ratio {ratio:.2f} below threshold", ratio)


def append_unique_strings(target: list[str], values: list[str]) -> None:
    seen = {normalize_text(value) for value in target}
    for value in values:
        key = normalize_text(value)
        if key and key not in seen:
            target.append(value)
            seen.add(key)


def append_semantically_unique_strings(
    target: list[str],
    values: list[str],
    *,
    max_items: int,
) -> None:
    for value in values:
        text = str(value or "").strip()
        if not text or any(texts_semantically_equivalent(text, existing) for existing in target):
            continue
        target.append(text)
        if len(target) >= max(1, int(max_items)):
            break


def source_requirement_gaps(
    state: GlobalResearchState,
    task: Any,
    source_ids: list[str],
    *,
    descriptions: bool = True,
) -> list[str]:
    if not task.required_source_types:
        return []
    sources = [state.sources[source_id] for source_id in source_ids if source_id in state.sources]
    source_types = {source.source_type for source in sources}
    independence_groups = {source.independence_group for source in sources if source.independence_group}
    missing: list[str] = []
    for requirement in task.required_source_types:
        normalized = normalize_text(requirement)
        satisfied = False
        if normalized == "primary":
            satisfied = bool(source_types & {"official", "primary", "institutional"})
        elif normalized == "independent":
            satisfied = "independent_media" in source_types
        elif normalized == "corroborated":
            satisfied = len(independence_groups) >= 2
        else:
            satisfied = normalized in source_types
        if not satisfied:
            missing.append(
                f"Missing required source type for {task.objective}: {requirement}"
                if descriptions
                else requirement
            )
    return missing


def claim_source_requirement_gaps(
    state: GlobalResearchState,
    claim: ClaimRecord,
) -> list[str]:
    sources = []
    for evidence_id in claim.evidence_ids:
        evidence = state.evidence.get(evidence_id)
        source = state.sources.get(evidence.source_id) if evidence else None
        if source is not None:
            sources.append(source)
    source_types = {source.source_type for source in sources}
    groups = {source.independence_group for source in sources if source.independence_group}
    missing: list[str] = []
    for requirement in claim.required_source_types:
        normalized = normalize_text(requirement)
        if normalized == "primary":
            satisfied = bool(source_types & {"official", "primary", "institutional"})
        elif normalized == "independent":
            satisfied = "independent_media" in source_types
        elif normalized == "corroborated":
            satisfied = len(groups) >= 2
        else:
            satisfied = normalized in source_types
        if not satisfied:
            missing.append(requirement)
    return missing


def same_claim_dimensions(first: ClaimRecord, second: ClaimRecord) -> bool:
    left = {key: normalize_text(value) for key, value in first.dimensions.items() if normalize_text(value)}
    right = {key: normalize_text(value) for key, value in second.dimensions.items() if normalize_text(value)}
    shared = set(left) & set(right)
    if len(shared) < 2 or not ({"entity", "metric"} <= shared or {"metric", "period"} <= shared):
        return False
    discriminators = {"period", "geography", "unit", "denominator", "accounting_scope"}
    if any(key in left.keys() ^ right.keys() for key in discriminators):
        return False
    return all(left[key] == right[key] for key in shared)


def task_evidence_quality_score(
    state: GlobalResearchState,
    claims: list[ClaimRecord],
) -> float:
    scores: list[float] = []
    for claim in claims:
        sources = []
        for evidence_id in claim.evidence_ids:
            evidence = state.evidence.get(evidence_id)
            source = state.sources.get(evidence.source_id) if evidence else None
            if source is not None:
                sources.append(source)
        max_authority = max((source.authority_score for source in sources), default=0.0)
        groups = {source.independence_group for source in sources if source.independence_group}
        status_score = {
            "corroborated": 0.95,
            "single_source": 0.72,
            "qualified": 0.68,
            "provisional": 0.6,
            "source_incomplete": 0.5,
            "contested": 0.45,
            "disputed": 0.25,
        }.get(claim.status, 0.5)
        score = 0.55 * status_score + 0.35 * max_authority + 0.1 * min(1.0, len(groups) / 2)
        score += {"high": 0.05, "medium": 0.0, "low": -0.1}.get(claim.confidence, 0.0)
        if claim.missing_source_types:
            score = min(score, 0.55)
        scores.append(score)
    if not scores:
        return 0.0
    top = sorted(scores, reverse=True)[:3]
    return round(min(1.0, sum(top) / len(top) + 0.05 * (len(top) - 1)), 4)


def task_has_covered_evidence(
    state: GlobalResearchState,
    task: Any,
) -> bool:
    result = state.agent_results.get(task.id)
    if result is None or not result.claim_ids:
        return False
    claims = [state.claims[claim_id] for claim_id in result.claim_ids if claim_id in state.claims]
    decisive = any(
        claim.status in {"corroborated", "single_source", "qualified", "provisional"}
        and not claim.missing_source_types
        for claim in claims
    )
    if not decisive:
        return False
    if source_requirement_gaps(state, task, result.source_ids, descriptions=False):
        return False
    quality = task_evidence_quality_score(state, claims)
    return quality >= 0.72 or (task.status == TaskStatus.COMPLETED and quality >= 0.55)


def append_unique_dicts(target: list[dict[str, Any]], values: list[dict[str, Any]]) -> None:
    seen = {normalize_text(item) for item in target}
    for value in values:
        key = normalize_text(value)
        if key and key not in seen:
            target.append(value)
            seen.add(key)
