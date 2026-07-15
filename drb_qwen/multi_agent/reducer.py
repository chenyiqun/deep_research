from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schemas import (
    ClaimRecord,
    GlobalResearchState,
    ResearchExecutionBundle,
    TaskStatus,
    normalize_text,
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
    for claim in bundle.claims:
        valid_evidence_ids = [
            evidence_id for evidence_id in claim.evidence_ids if evidence_id in accepted_evidence_ids
        ]
        if not claim.id or claim.subtask_id != task.id or not claim.text.strip() or not valid_evidence_ids:
            rejected_claims += 1
            continue
        claim.evidence_ids = valid_evidence_ids
        accepted_claim_ids.add(claim.id)
        existing = state.claims.get(claim.id)
        if existing is None:
            state.claims[claim.id] = claim
            added_claims += 1
        else:
            merge_claim(existing, claim)
    refresh_claim_statuses(state)

    result.source_ids = [source_id for source_id in result.source_ids if source_id in accepted_source_ids]
    result.evidence_ids = [
        evidence_id for evidence_id in result.evidence_ids if evidence_id in accepted_evidence_ids
    ]
    result.claim_ids = [claim_id for claim_id in result.claim_ids if claim_id in accepted_claim_ids]
    if result.status == TaskStatus.COMPLETED and not result.claim_ids:
        result.status = TaskStatus.PARTIAL
        append_unique_strings(
            result.unresolved_gaps,
            ["Subtask produced no valid source-grounded claims after integrity checks."],
        )

    task.status = result.status
    task.result_summary = result.answer_summary
    task.error = result.error
    task.attempts += 1
    task.updated_at = utc_now()
    state.agent_results[result.subtask_id] = result
    append_unique_strings(state.gaps, result.unresolved_gaps)
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
        "gaps": result.unresolved_gaps,
        "conflicts": result.conflicts,
    }


def merge_claim(existing: ClaimRecord, incoming: ClaimRecord) -> None:
    append_unique_strings(existing.evidence_ids, incoming.evidence_ids)
    append_unique_strings(existing.qualifiers, incoming.qualifiers)
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


def update_coverage(state: GlobalResearchState, subtask_id: str) -> None:
    task = state.tasks[subtask_id]
    result = state.agent_results.get(subtask_id)
    has_claims = bool(result and result.claim_ids)
    for target in task.coverage_targets:
        current = state.coverage.get(target, "missing")
        if task.status == TaskStatus.COMPLETED and has_claims:
            state.coverage[target] = "covered"
        elif task.status in {TaskStatus.COMPLETED, TaskStatus.PARTIAL} and current != "covered":
            state.coverage[target] = "partial"
        elif current not in {"covered", "partial"}:
            state.coverage[target] = "missing"


def initialize_coverage(state: GlobalResearchState) -> None:
    targets: list[str] = []
    if state.brief:
        targets.extend(state.brief.coverage_targets)
    for task in state.tasks.values():
        targets.extend(task.coverage_targets)
    for target in targets:
        state.coverage.setdefault(target, "missing")


def evaluate_research_gate(
    state: GlobalResearchState,
    *,
    min_total_claims: int,
    min_coverage_ratio: float,
    budget_exhausted: bool,
) -> ResearchGateDecision:
    targets = list(state.coverage.values())
    covered = sum(1 for value in targets if value == "covered")
    partial_targets = sum(1 for value in targets if value == "partial")
    ratio = (covered + 0.5 * partial_targets) / len(targets) if targets else (1.0 if state.claims else 0.0)
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


def append_unique_dicts(target: list[dict[str, Any]], values: list[dict[str, Any]]) -> None:
    seen = {normalize_text(item) for item in target}
    for value in values:
        key = normalize_text(value)
        if key and key not in seen:
            target.append(value)
            seen.add(key)
