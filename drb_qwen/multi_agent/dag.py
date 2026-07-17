from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .schemas import (
    GlobalResearchState,
    SATISFIED_DEPENDENCY_STATUSES,
    SubTask,
    TaskStatus,
    TaskType,
    normalize_text,
    safe_int,
    stable_id,
    string_list,
    texts_semantically_equivalent,
    utc_now,
)


TASK_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,79}$")
ALLOWED_OPERATIONS = {
    "ADD_TASK",
    "REFINE_TASK",
    "CANCEL_TASK",
    "ADD_DEPENDENCY",
    "SET_PRIORITY",
}


class DecisionValidationError(ValueError):
    pass


@dataclass
class PatchResult:
    added_task_ids: list[str]
    refined_task_ids: list[str]
    cancelled_task_ids: list[str]
    changed_task_ids: list[str]
    warnings: list[str]


def coerce_subtask(
    value: dict[str, Any],
    *,
    index: int,
    max_steps: int,
    max_tool_calls: int,
    created_by: str,
) -> SubTask:
    objective = str(value.get("objective", "")).strip()
    if not objective:
        raise DecisionValidationError("SubTask objective is required")
    proposed_id = str(value.get("id", value.get("task_id", ""))).strip()
    task_id = proposed_id if TASK_ID_RE.fullmatch(proposed_id) else stable_id("st", objective, index)
    task_type_text = str(value.get("task_type", value.get("type", "research"))).lower()
    try:
        task_type = TaskType(task_type_text)
    except ValueError:
        task_type = TaskType.RESEARCH
    return SubTask(
        id=task_id,
        objective=objective,
        task_type=task_type,
        rationale=str(value.get("rationale", "")).strip(),
        coverage_targets=string_list(value.get("coverage_targets"), 12),
        depends_on=string_list(value.get("depends_on"), 20),
        priority=max(0, min(100, safe_int(value.get("priority"), 50))),
        max_steps=max(1, min(max_steps, safe_int(value.get("max_steps"), max_steps))),
        max_tool_calls=max(1, min(max_tool_calls, safe_int(value.get("max_tool_calls"), max_tool_calls))),
        # Search caps and profiles are assigned by deterministic runtime policy;
        # planner output cannot expand its own budget.
        max_search_calls=0,
        research_profile="standard",
        required_source_types=string_list(value.get("required_source_types"), 10),
        created_by=created_by,
    )


def add_initial_tasks(
    state: GlobalResearchState,
    task_values: list[dict[str, Any]],
    *,
    max_subtasks: int,
    max_steps: int,
    max_tool_calls: int,
) -> list[str]:
    added: list[str] = []
    for index, value in enumerate(task_values[:max_subtasks], 1):
        if not isinstance(value, dict):
            continue
        task = coerce_subtask(
            value,
            index=index,
            max_steps=max_steps,
            max_tool_calls=max_tool_calls,
            created_by="main:initial",
        )
        task.id = unique_task_id(state.tasks, task.id, task.objective)
        state.tasks[task.id] = task
        added.append(task.id)
    validate_dependencies_exist(state.tasks)
    validate_acyclic(state.tasks)
    return added


def apply_decision_patch(
    state: GlobalResearchState,
    decision: dict[str, Any],
    *,
    max_subtasks: int,
    max_steps: int,
    max_tool_calls: int,
    max_new_tasks: int | None = None,
    max_task_attempts: int = 2,
) -> PatchResult:
    base_version = decision.get("base_state_version")
    if base_version is not None and safe_int(base_version, -1) != state.state_version:
        raise DecisionValidationError(
            f"stale DecisionPatch: base_state_version={base_version}, current={state.state_version}"
        )
    operations = decision.get("operations", [])
    if not isinstance(operations, list):
        raise DecisionValidationError("DecisionPatch.operations must be a list")

    candidate = {task_id: SubTask.from_dict(task.to_dict()) for task_id, task in state.tasks.items()}
    result = PatchResult([], [], [], [], [])
    new_task_limit = max_subtasks if max_new_tasks is None else max(0, int(max_new_tasks))
    for op_index, operation in enumerate(operations, 1):
        if not isinstance(operation, dict):
            result.warnings.append(f"ignored non-object operation at index {op_index}")
            continue
        op = str(operation.get("op", "")).upper()
        if op not in ALLOWED_OPERATIONS:
            result.warnings.append(f"ignored unsupported operation {op or '<empty>'}")
            continue
        if op == "ADD_TASK":
            if len(result.added_task_ids) >= new_task_limit:
                result.warnings.append("max_new_tasks reached; ADD_TASK ignored")
                continue
            if len(candidate) >= max_subtasks:
                result.warnings.append("max_subtasks reached; ADD_TASK ignored")
                continue
            raw_task = operation.get("task") if isinstance(operation.get("task"), dict) else operation
            task = coerce_subtask(
                raw_task,
                index=len(candidate) + 1,
                max_steps=max_steps,
                max_tool_calls=max_tool_calls,
                created_by=f"main:round:{state.main_round}",
            )
            duplicate = find_duplicate_task(candidate, task)
            if duplicate:
                existing = candidate[duplicate]
                if (
                    existing.status in {TaskStatus.PARTIAL, TaskStatus.FAILED}
                    and existing.attempts < max(1, max_task_attempts)
                ):
                    _refine_task(
                        existing,
                        raw_task,
                        max_steps=max_steps,
                        max_tool_calls=max_tool_calls,
                    )
                    result.refined_task_ids.append(existing.id)
                    result.changed_task_ids.append(existing.id)
                    result.warnings.append(
                        f"duplicate ADD_TASK converted to REFINE_TASK; existing task={duplicate}"
                    )
                else:
                    result.warnings.append(f"duplicate objective ignored; existing task={duplicate}")
                continue
            task.id = unique_task_id(candidate, task.id, task.objective)
            candidate[task.id] = task
            result.added_task_ids.append(task.id)
            continue
        if op == "REFINE_TASK":
            task_id = str(operation.get("task_id", ""))
            task = candidate.get(task_id)
            if task is None:
                result.warnings.append(f"cannot refine missing task {task_id}")
                continue
            if task.status not in {TaskStatus.PARTIAL, TaskStatus.FAILED}:
                result.warnings.append(
                    f"cannot refine task {task_id} with status={task.status.value}"
                )
                continue
            if task.attempts >= max(1, max_task_attempts):
                result.warnings.append(
                    f"cannot refine task {task_id}; max attempts reached ({task.attempts})"
                )
                continue
            raw_task = operation.get("task") if isinstance(operation.get("task"), dict) else operation
            _refine_task(
                task,
                raw_task,
                max_steps=max_steps,
                max_tool_calls=max_tool_calls,
            )
            result.refined_task_ids.append(task_id)
            result.changed_task_ids.append(task_id)
            continue
        if op == "CANCEL_TASK":
            task_id = str(operation.get("task_id", ""))
            task = candidate.get(task_id)
            if task is None or task.status in SATISFIED_DEPENDENCY_STATUSES:
                result.warnings.append(f"cannot cancel missing/completed task {task_id}")
                continue
            task.status = TaskStatus.CANCELLED
            task.updated_at = utc_now()
            task.error = str(operation.get("reason", operation.get("reason_code", "cancelled by main")))
            result.cancelled_task_ids.append(task_id)
            continue
        if op == "ADD_DEPENDENCY":
            dependency_id = str(operation.get("from", operation.get("depends_on", "")))
            task_id = str(operation.get("to", operation.get("task_id", "")))
            if dependency_id not in candidate or task_id not in candidate:
                result.warnings.append(f"dependency references unknown task: {dependency_id} -> {task_id}")
                continue
            if dependency_id == task_id:
                raise DecisionValidationError(f"self dependency is not allowed: {task_id}")
            task = candidate[task_id]
            if dependency_id not in task.depends_on:
                task.depends_on.append(dependency_id)
                result.changed_task_ids.append(task_id)
            continue
        if op == "SET_PRIORITY":
            task_id = str(operation.get("task_id", ""))
            if task_id not in candidate:
                result.warnings.append(f"priority references unknown task: {task_id}")
                continue
            candidate[task_id].priority = max(0, min(100, safe_int(operation.get("priority"), 50)))
            result.changed_task_ids.append(task_id)

    validate_dependencies_exist(candidate)
    validate_acyclic(candidate)
    state.tasks = candidate
    return result


def _refine_task(
    task: SubTask,
    value: dict[str, Any],
    *,
    max_steps: int,
    max_tool_calls: int,
) -> None:
    """Reopen a terminal incomplete task while preserving its identity and evidence history."""

    objective = str(value.get("objective", "")).strip()
    if objective:
        task.objective = objective
    rationale = str(value.get("rationale", "")).strip()
    if rationale:
        task.rationale = rationale
    for target in string_list(value.get("coverage_targets"), 12):
        if normalize_text(target) not in {
            normalize_text(existing) for existing in task.coverage_targets
        }:
            task.coverage_targets.append(target)
    if "required_source_types" in value:
        task.required_source_types = string_list(value.get("required_source_types"), 10)
    task.priority = max(0, min(100, safe_int(value.get("priority"), task.priority)))
    task.max_steps = max(
        task.max_steps,
        max(1, min(max_steps, safe_int(value.get("max_steps"), task.max_steps))),
    )
    task.max_tool_calls = max(
        task.max_tool_calls,
        max(1, min(max_tool_calls, safe_int(value.get("max_tool_calls"), task.max_tool_calls))),
    )
    task.status = TaskStatus.PENDING
    task.result_summary = ""
    task.error = ""
    task.updated_at = utc_now()


COMPLEX_PROFILE_PATTERNS = {
    "comparison": ("对比", "比较", "横向", "排名", "compare", "comparison", "ranking"),
    "time_series": ("历年", "历史", "走势", "逐年", "cagr", "time series", "historical"),
    "quantitative": ("量化", "占比", "幅度", "金额", "规模", "quantify", "percentage", "impact"),
}


def infer_research_profile(task: SubTask) -> str:
    if task.task_type == TaskType.REPAIR:
        return "repair"
    if task.task_type == TaskType.VERIFY:
        return "verify"
    haystack = normalize_text(" ".join([task.objective, *task.coverage_targets]))
    matched = [
        profile
        for profile, patterns in COMPLEX_PROFILE_PATTERNS.items()
        if any(pattern in haystack for pattern in patterns)
    ]
    if len(matched) >= 2 or len(_task_concepts(task)) >= 2:
        return "complex"
    if matched:
        return matched[0]
    return "standard"


def apply_adaptive_task_budgets(
    state: GlobalResearchState,
    *,
    base_steps: int,
    base_tool_calls: int,
    complex_max_steps: int,
    complex_max_tool_calls: int,
    complex_max_search_calls: int,
    max_queries_per_step: int,
) -> list[dict[str, Any]]:
    """Raise only genuinely complex task budgets; global budgets remain hard limits."""

    changes: list[dict[str, Any]] = []
    for task in state.tasks.values():
        if task.status not in {TaskStatus.PENDING, TaskStatus.PARTIAL, TaskStatus.FAILED}:
            continue
        profile = infer_research_profile(task)
        before = (task.max_steps, task.max_tool_calls, task.max_search_calls, task.research_profile)
        if profile in {"comparison", "time_series", "quantitative", "complex", "repair"}:
            task.max_steps = max(task.max_steps, complex_max_steps)
            task.max_tool_calls = max(task.max_tool_calls, complex_max_tool_calls)
            task.max_search_calls = max(task.max_search_calls, complex_max_search_calls)
        elif profile == "verify":
            task.max_steps = max(task.max_steps, min(complex_max_steps, base_steps + 1))
            task.max_tool_calls = max(
                task.max_tool_calls,
                min(complex_max_tool_calls, base_tool_calls + max_queries_per_step * 3),
            )
            task.max_search_calls = max(
                task.max_search_calls,
                min(complex_max_search_calls, task.max_steps * max_queries_per_step),
            )
        else:
            task.max_steps = max(task.max_steps, base_steps)
            task.max_tool_calls = max(task.max_tool_calls, base_tool_calls)
            task.max_search_calls = max(
                task.max_search_calls,
                min(complex_max_search_calls, task.max_steps * max_queries_per_step),
            )
        task.max_steps = min(task.max_steps, complex_max_steps)
        task.max_tool_calls = min(task.max_tool_calls, complex_max_tool_calls)
        task.max_search_calls = min(task.max_search_calls, complex_max_search_calls)
        task.research_profile = profile
        after = (task.max_steps, task.max_tool_calls, task.max_search_calls, task.research_profile)
        if after != before:
            task.updated_at = utc_now()
            changes.append(
                {
                    "task_id": task.id,
                    "profile": profile,
                    "max_steps": task.max_steps,
                    "max_tool_calls": task.max_tool_calls,
                    "max_search_calls": task.max_search_calls,
                }
            )
    return changes


def split_overloaded_tasks(
    state: GlobalResearchState,
    *,
    max_targets_per_task: int,
    max_subtasks: int,
) -> list[str]:
    """Split broad pending research tasks and fan dependent edges out to every shard."""

    chunk_size = max(1, int(max_targets_per_task))
    added: list[str] = []
    dependency_expansions: dict[str, list[str]] = {}
    for task in list(state.tasks.values()):
        targets = list(task.coverage_targets)
        if (
            task.status != TaskStatus.PENDING
            or task.attempts > 0
            or len(targets) <= chunk_size
        ):
            continue
        chunks = [targets[index : index + chunk_size] for index in range(0, len(targets), chunk_size)]
        available = max(0, max_subtasks - len(state.tasks))
        max_chunks = 1 + available
        if len(chunks) > max_chunks:
            chunks = chunks[: max_chunks - 1] + [
                [target for chunk in chunks[max_chunks - 1 :] for target in chunk]
            ]
        if len(chunks) <= 1:
            continue
        original_id = task.id
        original_objective = task.objective
        task.coverage_targets = chunks[0]
        task.objective = scoped_objective(original_objective, chunks[0])
        shard_ids = [original_id]
        for shard_index, chunk in enumerate(chunks[1:], 2):
            shard = SubTask.from_dict(task.to_dict())
            shard.id = unique_task_id(state.tasks, f"{original_id}_{shard_index}", shard.objective)
            shard.objective = scoped_objective(original_objective, chunk)
            shard.coverage_targets = chunk
            shard.created_by = f"{task.created_by}:split"
            shard.created_at = utc_now()
            shard.updated_at = shard.created_at
            state.tasks[shard.id] = shard
            shard_ids.append(shard.id)
            added.append(shard.id)
        dependency_expansions[original_id] = shard_ids

    for task in state.tasks.values():
        expanded: list[str] = []
        for dependency in task.depends_on:
            expanded.extend(dependency_expansions.get(dependency, [dependency]))
        task.depends_on = list(dict.fromkeys(dep for dep in expanded if dep != task.id))
    validate_dependencies_exist(state.tasks)
    validate_acyclic(state.tasks)
    return added


def scoped_objective(objective: str, targets: list[str]) -> str:
    scope = "；".join(targets)
    return f"{objective} [Focused scope / 本子任务仅覆盖：{scope}]"


def ready_tasks(state: GlobalResearchState, limit: int) -> list[SubTask]:
    ready: list[SubTask] = []
    for task in state.tasks.values():
        if task.status != TaskStatus.PENDING:
            continue
        dependencies = [state.tasks.get(dep) for dep in task.depends_on]
        if dependencies and not all(dep and dep.status in SATISFIED_DEPENDENCY_STATUSES for dep in dependencies):
            continue
        ready.append(task)
    ready.sort(key=lambda task: (-task.priority, task.created_at, task.id))
    return ready[: max(0, limit)]


def blocked_tasks(state: GlobalResearchState) -> list[SubTask]:
    output: list[SubTask] = []
    for task in state.tasks.values():
        if task.status != TaskStatus.PENDING:
            continue
        deps = [state.tasks.get(dep) for dep in task.depends_on]
        if any(dep and dep.status in {TaskStatus.FAILED, TaskStatus.CANCELLED} for dep in deps):
            output.append(task)
    return output


def reset_interrupted_tasks(state: GlobalResearchState) -> list[str]:
    reset: list[str] = []
    for task in state.tasks.values():
        if task.status == TaskStatus.RUNNING:
            task.status = TaskStatus.PENDING
            task.error = "recovered after interrupted run"
            task.updated_at = utc_now()
            reset.append(task.id)
    return reset


def all_tasks_terminal(state: GlobalResearchState) -> bool:
    return bool(state.tasks) and all(task.status in {
        TaskStatus.COMPLETED,
        TaskStatus.PARTIAL,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    } for task in state.tasks.values())


TASK_CONCEPT_PATTERNS = {
    "list": ("名单", "名录", "list", "ranking members"),
    "basic_info": ("基本信息", "基础信息", "basic information", "profile"),
    "finance": ("融资", "financing", "funding", "capital raising"),
    "dividend": ("分红", "股息", "dividend", "distribution"),
    "credit": ("信誉", "信用评级", "credibility", "credit rating"),
    "growth": ("增长", "增幅", "growth", "cagr"),
    "history": ("历史数据", "历史走势", "historical data", "price history"),
    "support": ("支撑", "support level", "support levels"),
    "resistance": ("压力位", "阻力位", "resistance", "resistance level"),
    "impact": ("影响幅度", "具体影响", "量化", "quantify", "impact magnitude"),
    "composition": ("收入构成", "完整构成", "composition", "breakdown"),
}


def _task_concepts(task: SubTask) -> set[str]:
    haystack = normalize_text(" ".join([task.objective, *task.coverage_targets]))
    return {
        concept
        for concept, patterns in TASK_CONCEPT_PATTERNS.items()
        if any(pattern in haystack for pattern in patterns)
    }


def find_duplicate_task(tasks: dict[str, SubTask], incoming: SubTask) -> str:
    incoming_targets = {normalize_text(value) for value in incoming.coverage_targets if normalize_text(value)}
    incoming_concepts = _task_concepts(incoming)
    for task_id, task in tasks.items():
        if task.status == TaskStatus.CANCELLED:
            continue
        if texts_semantically_equivalent(task.objective, incoming.objective):
            if incoming.task_type == TaskType.RESEARCH or task.task_type == incoming.task_type:
                return task_id
        # Research tasks must not be recreated merely to revisit an already
        # assigned coverage slot.  Verification/repair tasks are intentionally
        # allowed to overlap an earlier research task.
        if task.task_type == incoming.task_type == TaskType.RESEARCH:
            existing_targets = {
                normalize_text(value) for value in task.coverage_targets if normalize_text(value)
            }
            if incoming_targets & existing_targets:
                return task_id
            existing_concepts = _task_concepts(task)
            if len(incoming_concepts) >= 2 and len(existing_concepts) >= 2:
                overlap = len(incoming_concepts & existing_concepts) / min(
                    len(incoming_concepts),
                    len(existing_concepts),
                )
                if overlap >= 0.8:
                    return task_id
    return ""


def unique_task_id(tasks: dict[str, SubTask], proposed: str, objective: str) -> str:
    if proposed not in tasks:
        return proposed
    for suffix in range(2, 1000):
        candidate = f"{proposed}_{suffix}"
        if candidate not in tasks:
            return candidate
    return stable_id("st", objective, len(tasks))


def validate_dependencies_exist(tasks: dict[str, SubTask]) -> None:
    for task in tasks.values():
        missing = [dep for dep in task.depends_on if dep not in tasks]
        if missing:
            raise DecisionValidationError(f"task {task.id} has missing dependencies: {missing}")


def validate_acyclic(tasks: dict[str, SubTask]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> None:
        if task_id in visited:
            return
        if task_id in visiting:
            raise DecisionValidationError(f"task DAG contains a cycle at {task_id}")
        visiting.add(task_id)
        for dependency in tasks[task_id].depends_on:
            visit(dependency)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in tasks:
        visit(task_id)
