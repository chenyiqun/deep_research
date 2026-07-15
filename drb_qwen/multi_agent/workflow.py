from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import time
from typing import Any

from .agents import (
    CitationAuditor,
    MainAgent,
    ReportWriter,
    ResearcherAgent,
    fallback_initial_plan,
)
from .dag import (
    DecisionValidationError,
    add_initial_tasks,
    all_tasks_terminal,
    apply_decision_patch,
    blocked_tasks,
    ready_tasks,
    reset_interrupted_tasks,
)
from .context import ResearcherContextBuilder, TokenCounter
from .inference import AgentInferenceConfig, AgentInferenceGateway
from .reducer import evaluate_research_gate, initialize_coverage, merge_research_bundle
from .schemas import (
    AgentResult,
    GlobalResearchState,
    LocalResearchState,
    ResearchExecutionBundle,
    RunPhase,
    SubTask,
    TaskStatus,
    stable_id,
    normalize_text,
    utc_now,
)
from .store import RunStore
from .tools import ResearchTools


@dataclass
class DeepResearchConfig:
    # Backward-compatible names used by the existing CLI.
    max_rounds: int = 4
    min_rounds: int = 1
    max_search_queries_per_round: int = 2
    search_top_k: int = 5
    search_count: int = 10
    fetch_full_content: bool = True
    min_fetched_content_chars: int = 500
    max_concurrent_readers: int = 12
    planner_max_tokens: int = 3072
    reader_max_tokens: int = 1536
    summarizer_max_tokens: int = 1200
    state_updater_max_tokens: int = 3072
    report_max_tokens: int = 8192
    source_content_max_chars: int = 12000
    state_prompt_max_chars: int = 30000
    evidence_prompt_max_chars: int = 52000
    temperature_planner: float = 0.2
    temperature_reader: float = 0.0
    temperature_summarizer: float = 0.1
    temperature_report: float = 0.2

    # New event-driven multi-agent controls.
    max_initial_tasks: int = 4
    max_researchers: int = 4
    max_subtasks: int = 16
    max_new_tasks_per_round: int = 3
    max_react_steps: int = 3
    max_tool_calls_per_subtask: int = 18
    max_total_tool_calls: int = 160
    max_total_searches: int = 30
    max_total_tokens: int = 1_000_000
    max_run_seconds: int = 3600
    min_total_claims: int = 3
    min_coverage_ratio: float = 0.6
    citation_audit_enabled: bool = True
    auditor_max_tokens: int = 3072
    max_audit_rounds: int = 2
    max_repair_tasks: int = 3
    run_state_dir: str = ""
    resume_runs: bool = False

    # One inference request is one Agent turn. State is external and tools run
    # only after the request has released its inference admission permit.
    max_model_len: int = 32768
    context_safety_tokens: int = 512
    tokenizer_path: str = ""
    inference_max_concurrent_requests: int = 16
    inference_control_concurrency: int = 8
    inference_long_output_concurrency: int = 2
    inference_max_concurrent_per_run: int = 12
    inference_max_inflight_tokens: int = 262_144
    inference_structured_outputs: bool = True
    inference_forward_priority: bool = False
    inference_disable_thinking_for_json: bool = True

    def __post_init__(self) -> None:
        positive_fields = (
            "max_rounds",
            "min_rounds",
            "max_search_queries_per_round",
            "search_top_k",
            "search_count",
            "min_fetched_content_chars",
            "max_concurrent_readers",
            "planner_max_tokens",
            "reader_max_tokens",
            "summarizer_max_tokens",
            "state_updater_max_tokens",
            "report_max_tokens",
            "source_content_max_chars",
            "max_initial_tasks",
            "max_researchers",
            "max_subtasks",
            "max_new_tasks_per_round",
            "max_react_steps",
            "max_tool_calls_per_subtask",
            "max_total_tool_calls",
            "max_total_searches",
            "max_total_tokens",
            "max_run_seconds",
            "min_total_claims",
            "auditor_max_tokens",
            "max_audit_rounds",
            "max_repair_tasks",
            "max_model_len",
            "context_safety_tokens",
            "inference_max_concurrent_requests",
            "inference_control_concurrency",
            "inference_long_output_concurrency",
            "inference_max_concurrent_per_run",
            "inference_max_inflight_tokens",
        )
        invalid = [name for name in positive_fields if int(getattr(self, name)) <= 0]
        if invalid:
            raise ValueError(f"DeepResearchConfig fields must be positive: {', '.join(invalid)}")
        if not 0.0 <= self.min_coverage_ratio <= 1.0:
            raise ValueError("min_coverage_ratio must be between 0 and 1")
        if self.max_initial_tasks > self.max_subtasks:
            raise ValueError("max_initial_tasks cannot exceed max_subtasks")
        if self.min_rounds > self.max_rounds:
            raise ValueError("min_rounds cannot exceed max_rounds")
        if self.summarizer_max_tokens + self.context_safety_tokens >= self.max_model_len:
            raise ValueError("Researcher output and safety budgets must leave room for model input")


class AsyncDeepResearchWorkflow:
    """Adaptive centralized Main + bounded Researcher deep-research runtime."""

    def __init__(
        self,
        llm: Any,
        search_client: Any,
        content_fetcher: Any | None = None,
        config: DeepResearchConfig | None = None,
    ) -> None:
        self.search_client = search_client
        self.content_fetcher = content_fetcher
        self.config = config or DeepResearchConfig()
        token_counter = getattr(llm, "token_counter", None) or TokenCounter(self.config.tokenizer_path)
        if hasattr(llm, "infer_with_usage"):
            self.llm = llm
        else:
            self.llm = AgentInferenceGateway(
                llm,
                config=AgentInferenceConfig(
                    max_concurrent_requests=self.config.inference_max_concurrent_requests,
                    control_concurrency=self.config.inference_control_concurrency,
                    reader_concurrency=self.config.max_concurrent_readers,
                    long_output_concurrency=self.config.inference_long_output_concurrency,
                    max_concurrent_per_run=self.config.inference_max_concurrent_per_run,
                    max_inflight_tokens=self.config.inference_max_inflight_tokens,
                    structured_outputs=self.config.inference_structured_outputs,
                    forward_priority=self.config.inference_forward_priority,
                    disable_thinking_for_json=self.config.inference_disable_thinking_for_json,
                ),
                token_counter=token_counter,
            )
        self.store = RunStore(self.config.run_state_dir)
        self.main = MainAgent(
            llm=self.llm,
            max_initial_tasks=self.config.max_initial_tasks,
            max_new_tasks_per_round=self.config.max_new_tasks_per_round,
            max_react_steps=self.config.max_react_steps,
            max_tool_calls_per_subtask=self.config.max_tool_calls_per_subtask,
            planner_max_tokens=self.config.planner_max_tokens,
            replan_max_tokens=self.config.state_updater_max_tokens,
            temperature=self.config.temperature_planner,
        )
        tools = ResearchTools(
            llm=self.llm,
            search_client=search_client,
            content_fetcher=content_fetcher,
            store=self.store,
            search_top_k=self.config.search_top_k,
            fetch_full_content=self.config.fetch_full_content,
            min_fetched_content_chars=self.config.min_fetched_content_chars,
            source_content_max_chars=self.config.source_content_max_chars,
            reader_max_tokens=self.config.reader_max_tokens,
            reader_temperature=self.config.temperature_reader,
            max_concurrent_readers=self.config.max_concurrent_readers,
        )
        self.researcher = ResearcherAgent(
            llm=self.llm,
            tools=tools,
            store=self.store,
            context_builder=ResearcherContextBuilder(
                token_counter=token_counter,
                max_input_tokens=(
                    self.config.max_model_len
                    - self.config.summarizer_max_tokens
                    - self.config.context_safety_tokens
                ),
            ),
            max_queries_per_step=self.config.max_search_queries_per_round,
            researcher_max_tokens=self.config.summarizer_max_tokens,
            temperature=self.config.temperature_summarizer,
        )
        self.writer = ReportWriter(
            llm=self.llm,
            max_tokens=self.config.report_max_tokens,
            temperature=self.config.temperature_report,
        )
        self.auditor = CitationAuditor(
            llm=self.llm,
            max_tokens=self.config.auditor_max_tokens,
            max_repair_tasks=self.config.max_repair_tasks,
        )

    async def run(
        self,
        task: dict[str, Any],
        *,
        run_id: str | None = None,
        resume: bool | None = None,
    ) -> dict[str, Any]:
        resolved_run_id = run_id or derive_run_id(task)
        should_resume = self.config.resume_runs if resume is None else resume
        state = self.store.load_global(resolved_run_id) if should_resume else None
        trace: list[dict[str, Any]] = self.store.load_events(resolved_run_id) if state is not None else []
        started = time.monotonic()

        if state is not None and not same_research_task(state.task, task):
            raise ValueError(
                f"run_id {resolved_run_id!r} belongs to a different research task; "
                "use a new run_id or resume with the original task"
            )

        if state is None:
            if not should_resume:
                self.store.clear_run(resolved_run_id)
            state = GlobalResearchState(run_id=resolved_run_id, task=dict(task))
            await self._initialize_run(state, trace)
        else:
            if state.phase in {RunPhase.COMPLETED, RunPhase.PARTIAL, RunPhase.CANCELLED}:
                return self._result(state, trace)
            if state.phase == RunPhase.FAILED:
                if state.brief is None or not state.tasks:
                    self.store.clear_run(resolved_run_id)
                    trace = []
                    state = GlobalResearchState(run_id=resolved_run_id, task=dict(task))
                    await self._initialize_run(state, trace)
                else:
                    state.phase = RunPhase.RESEARCHING
                    state.stop_reason = ""
            reset_ids = reset_interrupted_tasks(state)
            if reset_ids:
                state.bump_version()
            self._record(trace, state, "run_resumed", {"reset_task_ids": reset_ids})
            self.store.save_global(state)
        try:
            while not state.terminal:
                if self._sync_cancelled(state):
                    break
                if time.monotonic() - started >= self.config.max_run_seconds:
                    state.stop_reason = "run deadline reached"
                    state.phase = RunPhase.WRITING

                if state.phase in {RunPhase.SCOPED, RunPhase.RESEARCHING}:
                    await self._research_phase(state, trace)
                    continue

                if state.phase == RunPhase.WRITING:
                    state.article, evidence_packet, writer_usage = await self.writer.write(state)
                    if self._sync_cancelled(state):
                        break
                    state.budget.writer_calls += 1
                    state.budget.add(writer_usage)
                    draft_artifact_id = f"draft_{state.audit_round + 1}"
                    self.store.save_artifact(
                        state.run_id,
                        draft_artifact_id,
                        state.article,
                        {"kind": "report_draft", "audit_round": state.audit_round},
                    )
                    state.phase = RunPhase.AUDITING if self.config.citation_audit_enabled else self._release_phase(state)
                    if not self.config.citation_audit_enabled:
                        self.store.save_artifact(
                            state.run_id,
                            "final_report",
                            state.article,
                            {"kind": "final_report", "phase": state.phase.value},
                        )
                    state.bump_version()
                    self._record(
                        trace,
                        state,
                        "report_written",
                        {
                            "article_chars": len(state.article),
                            "evidence_packet_items": len(evidence_packet),
                            "artifact_id": draft_artifact_id,
                            "next_phase": state.phase.value,
                        },
                    )
                    self.store.save_global(state)
                    continue

                if state.phase == RunPhase.AUDITING:
                    await self._audit_phase(state, trace)
                    continue

                unsupported_phase = state.phase.value
                state.phase = RunPhase.FAILED
                state.stop_reason = f"unsupported workflow phase: {unsupported_phase}"
                state.bump_version()
        except Exception as exc:
            state.phase = RunPhase.FAILED
            state.stop_reason = str(exc)
            state.bump_version()
            self._record(trace, state, "run_failed", {"error": str(exc)})
            self.store.save_global(state)
            raise

        self._record(
            trace,
            state,
            "run_finished",
            {"phase": state.phase.value, "stop_reason": state.stop_reason},
        )
        self.store.save_global(state)
        return self._result(state, trace)

    def cancel(self, run_id: str, reason: str = "cancelled by caller") -> bool:
        state = self.store.load_global(run_id)
        if state is None or state.terminal:
            return False
        state.phase = RunPhase.CANCELLED
        state.stop_reason = reason
        state.bump_version()
        self.store.append_event(run_id, "run_cancelled", {"reason": reason})
        self.store.save_global(state)
        return True

    async def _initialize_run(self, state: GlobalResearchState, trace: list[dict[str, Any]]) -> None:
        self._record(trace, state, "run_created", {"task_id": state.task.get("id")})
        plan = await self.main.initial_plan(state.task, run_id=state.run_id)
        state.budget.main_calls += 1
        state.budget.add(plan.usage or {})
        state.main_round = 1
        state.brief = plan.brief
        state.phase = RunPhase.SCOPED
        try:
            added = add_initial_tasks(
                state,
                plan.tasks,
                max_subtasks=self.config.max_subtasks,
                max_steps=self.config.max_react_steps,
                max_tool_calls=self.config.max_tool_calls_per_subtask,
            )
        except DecisionValidationError as exc:
            fallback = fallback_initial_plan(state.task, plan.raw_response, self.config.max_initial_tasks)
            state.brief = fallback.brief
            state.tasks = {}
            added = add_initial_tasks(
                state,
                fallback.tasks,
                max_subtasks=self.config.max_subtasks,
                max_steps=self.config.max_react_steps,
                max_tool_calls=self.config.max_tool_calls_per_subtask,
            )
            plan.used_fallback = True
            self._record(trace, state, "main_plan_rejected", {"error": str(exc)})
        initialize_coverage(state)
        state.phase = RunPhase.RESEARCHING
        state.bump_version()
        self._record(
            trace,
            state,
            "main_initial_plan",
            {
                "main_round": state.main_round,
                "task_ids": added,
                "used_fallback": plan.used_fallback,
                "brief": state.brief.to_dict() if state.brief else {},
            },
        )
        self.store.save_global(state)

    async def _research_phase(self, state: GlobalResearchState, trace: list[dict[str, Any]]) -> None:
        state.phase = RunPhase.RESEARCHING
        recovered_task_ids = self._merge_cached_bundles(state, trace)
        if recovered_task_ids:
            self._record(
                trace,
                state,
                "cached_bundles_recovered",
                {"task_ids": recovered_task_ids},
            )
            self.store.save_global(state)
        budget_exhausted = self._budget_exhausted(state)
        if budget_exhausted:
            state.stop_reason = "research budget exhausted; producing a qualified partial report"
            state.phase = RunPhase.WRITING
            state.bump_version()
            self._record(trace, state, "research_budget_exhausted", {"budget": state.budget.to_dict()})
            self.store.save_global(state)
            return

        remaining_tool_budget = max(0, self.config.max_total_tool_calls - state.budget.tool_calls)
        remaining_search_budget = max(0, self.config.max_total_searches - state.budget.search_calls)
        minimum_task_budget = 3 if self.config.fetch_full_content and self.content_fetcher is not None else 2
        max_wave_by_budget = remaining_tool_budget // minimum_task_budget
        ready = ready_tasks(
            state,
            min(self.config.max_researchers, max_wave_by_budget, remaining_search_budget),
        )
        if ready:
            for task in ready:
                task.status = TaskStatus.RUNNING
                task.updated_at = utc_now()
            state.bump_version()
            self._record(
                trace,
                state,
                "scheduler_wave_started",
                {"task_ids": [task.id for task in ready], "wave_size": len(ready)},
            )
            self.store.save_global(state)

            execution_tasks: list[SubTask] = []
            search_allocations: list[int] = []
            allocatable = remaining_tool_budget
            allocatable_searches = remaining_search_budget
            for index, task in enumerate(ready):
                tasks_after = len(ready) - index - 1
                allocation = min(
                    task.max_tool_calls,
                    allocatable - tasks_after * minimum_task_budget,
                )
                allocation = max(1, allocation)
                execution_tasks.append(replace(task, max_tool_calls=allocation))
                allocatable -= allocation

                task_search_ceiling = task.max_steps * self.config.max_search_queries_per_round
                search_allocation = min(
                    task_search_ceiling,
                    allocatable_searches - tasks_after,
                )
                search_allocation = max(1, search_allocation)
                search_allocations.append(search_allocation)
                allocatable_searches -= search_allocation

            executions = await asyncio.gather(
                *(
                    self.researcher.execute(state, task, search_call_budget=search_budget)
                    for task, search_budget in zip(execution_tasks, search_allocations)
                ),
                return_exceptions=True,
            )
            if self._sync_cancelled(state):
                return
            for task, execution in zip(ready, executions):
                bundle = execution if isinstance(execution, ResearchExecutionBundle) else failed_bundle(state, task.id, execution)
                for event in bundle.events:
                    self._record(trace, state, str(event.get("type", "researcher_event")), event)
                merge_summary = merge_research_bundle(state, bundle)
                self._record(trace, state, "subtask_merged", merge_summary)
                self.store.save_global(state)
            self._record(
                trace,
                state,
                "scheduler_wave_completed",
                {"task_ids": [task.id for task in ready], "budget": state.budget.to_dict()},
            )
            self.store.save_global(state)
            await self._strategic_boundary(state, trace)
            return

        blocked = blocked_tasks(state)
        if blocked:
            for task in blocked:
                task.status = TaskStatus.CANCELLED
                task.error = "dependency failed or was cancelled"
                task.updated_at = utc_now()
                state.gaps.append(f"Blocked subtask: {task.objective}")
            state.bump_version()
            self._record(trace, state, "blocked_tasks_cancelled", {"task_ids": [task.id for task in blocked]})
            self.store.save_global(state)

        await self._strategic_boundary(state, trace)

    def _merge_cached_bundles(
        self,
        state: GlobalResearchState,
        trace: list[dict[str, Any]],
    ) -> list[str]:
        """Recover completed Researcher output that was durable before its global merge."""

        recovered: list[str] = []
        for task in state.tasks.values():
            if task.status != TaskStatus.PENDING or task.id in state.agent_results:
                continue
            bundle = self.store.load_bundle(state.run_id, task.id)
            if bundle is None:
                continue
            for event in bundle.events:
                self._record(trace, state, str(event.get("type", "researcher_event")), event)
            merge_summary = merge_research_bundle(state, bundle)
            self._record(trace, state, "subtask_merged_from_cache", merge_summary)
            self.store.save_global(state)
            recovered.append(task.id)
        return recovered

    async def _strategic_boundary(self, state: GlobalResearchState, trace: list[dict[str, Any]]) -> None:
        budget_exhausted = self._budget_exhausted(state)
        gate = evaluate_research_gate(
            state,
            min_total_claims=self.config.min_total_claims,
            min_coverage_ratio=self.config.min_coverage_ratio,
            budget_exhausted=budget_exhausted,
        )
        action = "continue"
        patch_result: dict[str, Any] = {}
        if state.main_round < self.config.max_rounds:
            decision, _, used_fallback = await self._call_main_replan(state)
            if self._sync_cancelled(state):
                return
            action = str(decision.get("action", "continue"))
            try:
                compiled = apply_decision_patch(
                    state,
                    decision,
                    max_subtasks=self.config.max_subtasks,
                    max_steps=self.config.max_react_steps,
                    max_tool_calls=self.config.max_tool_calls_per_subtask,
                    max_new_tasks=self.config.max_new_tasks_per_round,
                )
                patch_result = {
                    "added_task_ids": compiled.added_task_ids,
                    "cancelled_task_ids": compiled.cancelled_task_ids,
                    "changed_task_ids": compiled.changed_task_ids,
                    "warnings": compiled.warnings,
                }
            except DecisionValidationError as exc:
                action = "write" if state.claims else "partial"
                patch_result = {"error": str(exc)}
            initialize_coverage(state)
            state.bump_version()
            self._record(
                trace,
                state,
                "main_replan",
                {
                    "main_round": state.main_round,
                    "action": action,
                    "reason": decision.get("reason", ""),
                    "used_fallback": used_fallback,
                    "patch": patch_result,
                    "research_gate": gate.__dict__,
                },
            )
            self.store.save_global(state)
        elif all_tasks_terminal(state):
            action = "write" if state.claims else "partial"

        completed_waves = sum(1 for event in trace if event.get("type") == "scheduler_wave_completed")
        has_unfinished_work = any(
            task.status in {TaskStatus.PENDING, TaskStatus.RUNNING} for task in state.tasks.values()
        )
        if completed_waves < self.config.min_rounds and has_unfinished_work and action != "continue":
            action = "continue"
            self._record(
                trace,
                state,
                "minimum_rounds_enforced",
                {"completed_waves": completed_waves, "minimum_waves": self.config.min_rounds},
            )

        if action == "continue":
            if ready_tasks(state, 1):
                return
            if any(task.status == TaskStatus.PENDING for task in state.tasks.values()):
                return
            if gate.passed:
                state.phase = RunPhase.WRITING
            elif state.main_round >= self.config.max_rounds or all_tasks_terminal(state):
                state.phase = RunPhase.WRITING
                state.stop_reason = gate.reason
        elif action == "write":
            state.phase = RunPhase.WRITING
            if not gate.passed:
                state.stop_reason = f"qualified partial report: {gate.reason}"
        else:
            state.phase = RunPhase.WRITING
            state.stop_reason = str(state.stop_reason or "Main requested partial completion")
        state.bump_version()
        self.store.save_global(state)

    async def _audit_phase(self, state: GlobalResearchState, trace: list[dict[str, Any]]) -> None:
        evidence_packet = self._evidence_packet_for_audit(state)
        state.audit_round += 1
        audit, _, auditor_usage = await self.auditor.audit(state, evidence_packet)
        if self._sync_cancelled(state):
            return
        state.audit = audit
        state.budget.auditor_calls += 1
        state.budget.add(auditor_usage)
        state.bump_version()
        self._record(
            trace,
            state,
            "citation_audit",
            {
                "audit_round": state.audit_round,
                "passed": audit.passed,
                "issues": audit.issues,
                "repair_tasks": audit.repair_tasks,
            },
        )
        if audit.passed:
            state.phase = self._release_phase(state)
            self.store.save_artifact(
                state.run_id,
                "final_report",
                state.article,
                {"kind": "final_report", "phase": state.phase.value},
            )
            state.bump_version()
            self.store.save_global(state)
            return

        if state.audit_round >= self.config.max_audit_rounds or not audit.repair_tasks:
            state.phase = RunPhase.PARTIAL
            state.stop_reason = "citation audit did not pass within the repair budget"
            state.bump_version()
            self.store.save_global(state)
            return

        decision, _, used_fallback = await self._call_main_replan(state, allow_over_round_limit=True)
        if self._sync_cancelled(state):
            return
        if not decision.get("operations"):
            decision = audit_repair_patch(state)
            used_fallback = True
        try:
            compiled = apply_decision_patch(
                state,
                decision,
                max_subtasks=self.config.max_subtasks,
                max_steps=self.config.max_react_steps,
                max_tool_calls=self.config.max_tool_calls_per_subtask,
                max_new_tasks=self.config.max_repair_tasks,
            )
            added = compiled.added_task_ids
            warnings = compiled.warnings
        except DecisionValidationError as exc:
            fallback = audit_repair_patch(state)
            compiled = apply_decision_patch(
                state,
                fallback,
                max_subtasks=self.config.max_subtasks,
                max_steps=self.config.max_react_steps,
                max_tool_calls=self.config.max_tool_calls_per_subtask,
                max_new_tasks=self.config.max_repair_tasks,
            )
            added = compiled.added_task_ids
            warnings = [str(exc), *compiled.warnings]
            used_fallback = True
        initialize_coverage(state)
        if not added:
            state.phase = RunPhase.PARTIAL
            state.stop_reason = "audit repair produced no new executable task"
        else:
            state.phase = RunPhase.RESEARCHING
            state.stop_reason = ""
        state.bump_version()
        self._record(
            trace,
            state,
            "audit_repair_planned",
            {"task_ids": added, "warnings": warnings, "used_fallback": used_fallback},
        )
        self.store.save_global(state)

    async def _call_main_replan(
        self,
        state: GlobalResearchState,
        *,
        allow_over_round_limit: bool = False,
    ) -> tuple[dict[str, Any], str, bool]:
        if not allow_over_round_limit and state.main_round >= self.config.max_rounds:
            raise RuntimeError("Main planning budget exhausted")
        state.main_round += 1
        state.budget.main_calls += 1
        decision, response, used_fallback, usage = await self.main.replan(state)
        state.budget.add(usage)
        return decision, response, used_fallback

    def _budget_exhausted(self, state: GlobalResearchState) -> bool:
        minimum_task_budget = 3 if self.config.fetch_full_content and self.content_fetcher is not None else 2
        return (
            self.config.max_total_tool_calls - state.budget.tool_calls < minimum_task_budget
            or state.budget.search_calls >= self.config.max_total_searches
            or state.budget.total_tokens >= self.config.max_total_tokens
        )

    @staticmethod
    def _release_phase(state: GlobalResearchState) -> RunPhase:
        return RunPhase.PARTIAL if state.stop_reason else RunPhase.COMPLETED

    @staticmethod
    def _evidence_packet_for_audit(state: GlobalResearchState) -> list[dict[str, Any]]:
        from .agents import build_evidence_packet

        return build_evidence_packet(state)

    def _record(
        self,
        trace: list[dict[str, Any]],
        state: GlobalResearchState,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        event = self.store.append_event(state.run_id, event_type, payload)
        trace.append(event)

    def _sync_cancelled(self, state: GlobalResearchState) -> bool:
        persisted = self.store.load_global(state.run_id) if self.store.enabled else None
        if persisted is None or persisted.phase != RunPhase.CANCELLED:
            return False
        state.phase = RunPhase.CANCELLED
        state.stop_reason = persisted.stop_reason or "cancelled"
        state.state_version = max(state.state_version, persisted.state_version)
        state.updated_at = persisted.updated_at
        return True

    def _result(self, state: GlobalResearchState, trace: list[dict[str, Any]]) -> dict[str, Any]:
        result = {
            "run_id": state.run_id,
            "article": state.article,
            "state": state.to_dict(),
            "trace": trace,
            "audit": state.audit.to_dict() if state.audit else None,
        }
        metrics = getattr(self.llm, "metrics", None)
        if metrics is not None and hasattr(metrics, "to_dict"):
            result["inference"] = metrics.to_dict()
        return result


def derive_run_id(task: dict[str, Any]) -> str:
    explicit = str(task.get("run_id", "")).strip()
    if explicit:
        return explicit
    task_id = str(task.get("id", "task"))
    return stable_id("run", task_id, task.get("prompt", ""), length=20)


def same_research_task(persisted: dict[str, Any], requested: dict[str, Any]) -> bool:
    """Prevent an explicit run_id from silently resuming another question's state."""

    persisted_prompt = normalize_text(persisted.get("prompt", ""))
    requested_prompt = normalize_text(requested.get("prompt", ""))
    if persisted_prompt != requested_prompt:
        return False
    persisted_id = normalize_text(persisted.get("id", ""))
    requested_id = normalize_text(requested.get("id", ""))
    return not (persisted_id and requested_id and persisted_id != requested_id)


def failed_bundle(state: GlobalResearchState, subtask_id: str, exc: Any) -> ResearchExecutionBundle:
    error = str(exc) if isinstance(exc, BaseException) else "unknown researcher failure"
    task = state.tasks[subtask_id]
    local = LocalResearchState(
        run_id=state.run_id,
        subtask_id=subtask_id,
        objective=task.objective,
        status=TaskStatus.FAILED,
        stop_reason=error,
    )
    result = AgentResult(
        subtask_id=subtask_id,
        status=TaskStatus.FAILED,
        answer_summary="",
        unresolved_gaps=[f"Subtask failed: {task.objective}"],
        error=error,
    )
    return ResearchExecutionBundle(
        result=result,
        local_state=local,
        events=[{"type": "researcher_error", "subtask_id": subtask_id, "error": error}],
    )


def audit_repair_patch(state: GlobalResearchState) -> dict[str, Any]:
    repair_tasks = state.audit.repair_tasks if state.audit else []
    operations: list[dict[str, Any]] = []
    for index, item in enumerate(repair_tasks, 1):
        objective = str(item.get("objective", "")).strip()
        if not objective:
            continue
        operations.append(
            {
                "op": "ADD_TASK",
                "task": {
                    "id": f"repair_{state.audit_round}_{index}",
                    "task_type": "repair",
                    "objective": objective,
                    "coverage_targets": item.get("coverage_targets", [f"audit:repair:{index}"]),
                    "depends_on": [],
                    "priority": 95,
                    "max_steps": min(2, 3),
                    "max_tool_calls": 12,
                },
            }
        )
    return {
        "base_state_version": state.state_version,
        "action": "continue",
        "reason": "citation audit repair",
        "operations": operations,
    }
