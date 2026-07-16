from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable

from ..json_utils import extract_json
from ..url_utils import canonicalize_url, extract_urls
from .context import ContextWindowExceeded, ResearcherContextBuilder, TokenCounter
from .inference import is_context_length_error
from .llm import add_token_usage, call_chat
from .prompts import (
    AUDITOR_SYSTEM_PROMPT,
    MAIN_SYSTEM_PROMPT,
    RESEARCHER_SYSTEM_PROMPT,
    WRITER_SYSTEM_PROMPT,
    build_audit_prompt,
    build_initial_plan_prompt,
    build_replan_prompt,
    build_writer_prompt,
)
from .schemas import (
    AgentResult,
    AuditResult,
    ClaimRecord,
    EvidenceRecord,
    GlobalResearchState,
    LocalResearchState,
    ResearchBrief,
    ResearchExecutionBundle,
    SourceRecord,
    SubTask,
    TaskStatus,
    normalize_text,
    safe_int,
    string_list,
    texts_semantically_equivalent,
    utc_now,
)
from .protocols import (
    AUDIT_SCHEMA,
    MAIN_PLAN_SCHEMA,
    MAIN_REPLAN_SCHEMA,
    RESEARCHER_DECISION_SCHEMA,
)
from .store import RunStore
from .tools import ResearchTools


REACT_STEP_TERMINAL = "__react_step_terminal__"


@dataclass
class InitialPlanResult:
    brief: ResearchBrief
    tasks: list[dict[str, Any]]
    wakeup_policy: dict[str, Any]
    raw_response: str
    used_fallback: bool = False
    usage: dict[str, int] | None = None


class MainAgent:
    def __init__(
        self,
        *,
        llm: Any,
        max_initial_tasks: int,
        max_new_tasks_per_round: int,
        max_react_steps: int,
        max_tool_calls_per_subtask: int,
        planner_max_tokens: int,
        replan_max_tokens: int,
        temperature: float,
    ) -> None:
        self.llm = llm
        self.max_initial_tasks = max_initial_tasks
        self.max_new_tasks_per_round = max_new_tasks_per_round
        self.max_react_steps = max_react_steps
        self.max_tool_calls_per_subtask = max_tool_calls_per_subtask
        self.planner_max_tokens = planner_max_tokens
        self.replan_max_tokens = replan_max_tokens
        self.temperature = temperature

    async def initial_plan(self, task: dict[str, Any], *, run_id: str = "") -> InitialPlanResult:
        prompt = build_initial_plan_prompt(
            task,
            max_initial_tasks=self.max_initial_tasks,
            max_steps=self.max_react_steps,
            max_tool_calls=self.max_tool_calls_per_subtask,
        )
        response, usage = await call_chat(
            self.llm,
            user_prompt=prompt,
            system_prompt=MAIN_SYSTEM_PROMPT,
            temperature=self.temperature,
            max_tokens=self.planner_max_tokens,
            role="main_planner",
            run_id=run_id,
            request_id=f"{run_id}:main:plan:0" if run_id else "",
            response_schema=MAIN_PLAN_SCHEMA,
            schema_name="main_plan",
        )
        parsed = parse_object(response)
        if parsed is None:
            fallback = fallback_initial_plan(task, response, self.max_initial_tasks)
            fallback.usage = usage
            return fallback
        brief_value = parsed.get("research_brief")
        tasks_value = parsed.get("tasks")
        if not isinstance(brief_value, dict) or not isinstance(tasks_value, list):
            fallback = fallback_initial_plan(task, response, self.max_initial_tasks)
            fallback.usage = usage
            return fallback
        brief_value.setdefault("question", str(task.get("prompt", "")))
        brief_value.setdefault("language", str(task.get("language", "en")))
        tasks = [value for value in tasks_value if isinstance(value, dict)][: self.max_initial_tasks]
        if not tasks:
            fallback = fallback_initial_plan(task, response, self.max_initial_tasks)
            fallback.usage = usage
            return fallback
        return InitialPlanResult(
            brief=ResearchBrief.from_dict(brief_value),
            tasks=tasks,
            wakeup_policy=parsed.get("wakeup_policy", {}) if isinstance(parsed.get("wakeup_policy"), dict) else {},
            raw_response=response,
            usage=usage,
        )

    async def replan(
        self,
        state: GlobalResearchState,
    ) -> tuple[dict[str, Any], str, bool, dict[str, int]]:
        prompt = build_replan_prompt(
            state,
            max_new_tasks=self.max_new_tasks_per_round,
            max_steps=self.max_react_steps,
            max_tool_calls=self.max_tool_calls_per_subtask,
        )
        response, usage = await call_chat(
            self.llm,
            user_prompt=prompt,
            system_prompt=MAIN_SYSTEM_PROMPT,
            temperature=self.temperature,
            max_tokens=self.replan_max_tokens,
            role="main_replanner",
            run_id=state.run_id,
            request_id=f"{state.run_id}:main:replan:{state.state_version}",
            response_schema=MAIN_REPLAN_SCHEMA,
            schema_name="main_replan",
        )
        parsed = parse_object(response)
        if parsed is None:
            return fallback_replan(state), response, True, usage
        action = str(parsed.get("action", "continue")).lower()
        if action not in {"continue", "write", "partial"}:
            action = "continue"
        operations = parsed.get("operations", [])
        if not isinstance(operations, list):
            operations = []
        parsed["action"] = action
        parsed["operations"] = operations
        parsed.setdefault("base_state_version", state.state_version)
        return parsed, response, False, usage


class ResearcherAgent:
    def __init__(
        self,
        *,
        llm: Any,
        tools: ResearchTools,
        store: RunStore,
        context_builder: ResearcherContextBuilder,
        max_queries_per_step: int,
        researcher_max_tokens: int,
        temperature: float,
        recent_observation_limit: int = 8,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.store = store
        self.context_builder = context_builder
        self.max_queries_per_step = max_queries_per_step
        self.researcher_max_tokens = researcher_max_tokens
        self.temperature = temperature
        self.recent_observation_limit = recent_observation_limit

    async def execute(
        self,
        state: GlobalResearchState,
        subtask: SubTask,
        *,
        search_call_budget: int | None = None,
    ) -> ResearchExecutionBundle:
        cached_bundle = self.store.load_bundle(state.run_id, subtask.id)
        if cached_bundle is not None:
            return cached_bundle

        checkpoint = self.store.load_research_checkpoint(state.run_id, subtask.id)
        local = checkpoint["local_state"] if checkpoint is not None else self.store.load_local(state.run_id, subtask.id)
        if local is None or (
            checkpoint is None and local.status in {TaskStatus.COMPLETED, TaskStatus.PARTIAL, TaskStatus.FAILED}
        ):
            local = LocalResearchState(
                run_id=state.run_id,
                subtask_id=subtask.id,
                objective=subtask.objective,
                status=TaskStatus.RUNNING,
            )
        else:
            local.status = TaskStatus.RUNNING
        self.store.save_local(local)

        sources: dict[str, SourceRecord] = {
            item.id: item for item in (checkpoint.get("sources", []) if checkpoint else [])
        }
        evidence: dict[str, EvidenceRecord] = {
            item.id: item for item in (checkpoint.get("evidence", []) if checkpoint else [])
        }
        claims: dict[str, ClaimRecord] = {
            item.id: item for item in (checkpoint.get("claims", []) if checkpoint else [])
        }
        events: list[dict[str, Any]] = list(checkpoint.get("events", [])) if checkpoint else []
        usage = {
            "researcher_calls": 0,
            "reader_calls": 0,
            "search_calls": 0,
            "fetch_calls": 0,
            "tool_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        if checkpoint:
            for key, value in checkpoint.get("usage", {}).items():
                if key in usage:
                    usage[key] = int(value)
        max_search_calls = (
            subtask.max_steps * self.max_queries_per_step
            if search_call_budget is None
            else max(0, int(search_call_budget))
        )
        last_decision: dict[str, Any] = dict(checkpoint.get("last_decision", {})) if checkpoint else {}
        fatal_error = ""

        self.store.save_research_checkpoint(
            state.run_id,
            local_state=local,
            sources=list(sources.values()),
            evidence=list(evidence.values()),
            claims=list(claims.values()),
            events=events,
            usage=usage,
            last_decision=last_decision,
        )

        try:
            while (
                local.step < subtask.max_steps
                and local.tool_calls < subtask.max_tool_calls
                and local.stop_reason != REACT_STEP_TERMINAL
                and (usage["search_calls"] < max_search_calls or bool(claims))
            ):
                allowed_queries = min(
                    self.max_queries_per_step,
                    max(0, max_search_calls - usage["search_calls"]),
                )
                global_context = build_global_context_slice(state, subtask)
                context = self.context_builder.build(
                    original_task=state.task,
                    brief=state.brief or ResearchBrief(question=str(state.task.get("prompt", ""))),
                    subtask=subtask,
                    local=local,
                    global_context=global_context,
                    remaining_steps=subtask.max_steps - local.step,
                    remaining_tool_calls=subtask.max_tool_calls - local.tool_calls,
                    max_queries=allowed_queries,
                )
                response, token_usage = await call_chat(
                    self.llm,
                    user_prompt=context.prompt,
                    system_prompt=RESEARCHER_SYSTEM_PROMPT,
                    temperature=self.temperature,
                    max_tokens=self.researcher_max_tokens,
                    role="researcher",
                    run_id=state.run_id,
                    subtask_id=subtask.id,
                    request_id=f"{state.run_id}:{subtask.id}:react:{local.version}",
                    response_schema=RESEARCHER_DECISION_SCHEMA,
                    schema_name="researcher_decision",
                    estimated_input_tokens=context.estimated_tokens,
                )
                usage["researcher_calls"] += 1
                add_token_usage(usage, token_usage)
                parsed_decision = parse_object(response)
                if (
                    parsed_decision is None
                    or safe_int(parsed_decision.get("base_local_version"), -1) != local.version
                ):
                    decision = fallback_researcher_decision(local, claims, subtask)
                    decision["validation_error"] = "missing or stale base_local_version"
                else:
                    decision = parsed_decision
                local.step += 1

                queries = extract_search_queries(decision, allowed_queries)
                prior_queries = [*state.query_ledger, *local.queries]
                queries = [
                    query
                    for query in queries
                    if not any(
                        texts_semantically_equivalent(query, prior, threshold=0.94)
                        for prior in prior_queries
                    )
                ]
                max_affordable_queries = min(
                    max(0, subtask.max_tool_calls - local.tool_calls),
                    max(0, max_search_calls - usage["search_calls"]),
                )
                queries = queries[:max_affordable_queries]
                finish_requested = bool(decision.get("finish", False))
                assessment = decision.get("assessment", {})
                assessment_coverage = (
                    str(assessment.get("coverage", "none")).lower()
                    if isinstance(assessment, dict)
                    else "none"
                )
                finish_effective = (
                    finish_requested or assessment_coverage == "sufficient"
                ) and not queries
                if queries and (finish_requested or assessment_coverage == "sufficient"):
                    decision = dict(decision)
                    decision["finish"] = False
                    decision["stop_reason"] = ""
                    assessment_value = decision.get("assessment", {})
                    if isinstance(assessment_value, dict):
                        decision["assessment"] = {
                            **assessment_value,
                            "coverage": "partial",
                            "primary_gap": assessment_value.get("primary_gap")
                            or "new search observations still require a later synthesis step",
                        }
                elif finish_effective and not finish_requested:
                    # Treat an explicit sufficient assessment as the terminal
                    # signal even when the model omitted the redundant flag.
                    decision = dict(decision)
                    decision["finish"] = True
                    decision.setdefault("stop_reason", "evidence sufficient")
                add_local_semantic_patch(local, decision)
                last_decision = decision

                event: dict[str, Any] = {
                    "type": "researcher_step",
                    "subtask_id": subtask.id,
                    "step": local.step,
                    "assessment": decision.get("assessment", {}),
                    "queries": queries,
                    "finish_requested": finish_requested,
                    "finish_effective": finish_effective,
                    "validation_error": decision.get("validation_error", ""),
                    "input_state_version": local.version,
                    "estimated_input_tokens": context.estimated_tokens,
                    "max_input_tokens": context.max_input_tokens,
                    "token_count_exact": context.token_count_exact,
                    "dropped_observations": context.dropped_observations,
                    "dropped_global_claims": context.dropped_global_claims,
                }
                if queries:
                    tool_result = await self.tools.search_and_read(
                        run_id=state.run_id,
                        original_task=state.task,
                        subtask=subtask,
                        queries=queries,
                        tool_call_budget=subtask.max_tool_calls - local.tool_calls,
                    )
                    for source in tool_result.sources:
                        sources[source.id] = source
                    for item in tool_result.evidence:
                        evidence[item.id] = item
                    for claim in tool_result.claims:
                        existing = claims.get(claim.id)
                        if existing is None:
                            claims[claim.id] = claim
                        else:
                            for evidence_id in claim.evidence_ids:
                                if evidence_id not in existing.evidence_ids:
                                    existing.evidence_ids.append(evidence_id)
                    add_tool_result_to_local(local, queries, tool_result)
                    for key in usage:
                        usage[key] += int(tool_result.usage.get(key, 0))
                    event.update(
                        {
                            "num_sources": len(tool_result.sources),
                            "num_evidence": len(tool_result.evidence),
                            "num_claims": len(tool_result.claims),
                            "errors": tool_result.errors,
                            "observations": tool_result.observations,
                        }
                    )
                events.append(event)
                local.version += 1
                event["output_state_version"] = local.version
                step_should_stop = (
                    finish_effective
                    or (not queries and local.step >= subtask.max_steps)
                )
                local.stop_reason = REACT_STEP_TERMINAL if step_should_stop else ""
                local.updated_at = utc_now()
                self.store.save_local(local)
                self.store.save_research_checkpoint(
                    state.run_id,
                    local_state=local,
                    sources=list(sources.values()),
                    evidence=list(evidence.values()),
                    claims=list(claims.values()),
                    events=events,
                    usage=usage,
                    last_decision=last_decision,
                )

                if step_should_stop:
                    break
        except Exception as exc:
            fatal_error = str(exc)
            events.append({"type": "researcher_error", "subtask_id": subtask.id, "error": fatal_error})

        summary = str(last_decision.get("answer_summary", "")).strip()
        if not summary:
            summary = summarize_claims(claims)
        final_assessment = last_decision.get("assessment", {})
        final_coverage = (
            str(
                final_assessment.get(
                    "coverage",
                    "sufficient" if last_decision.get("finish", False) else "none",
                )
            ).lower()
            if isinstance(final_assessment, dict)
            else "none"
        )
        terminal_decision = bool(last_decision.get("finish", False)) or final_coverage == "sufficient"
        if fatal_error and not claims:
            status = TaskStatus.FAILED
            stop_reason = "researcher runtime failed"
        elif claims and terminal_decision and final_coverage == "sufficient":
            status = TaskStatus.COMPLETED
            stop_reason = str(last_decision.get("stop_reason", "evidence sufficient"))
        elif claims:
            status = TaskStatus.PARTIAL
            stop_reason = "local step or tool budget reached"
        else:
            status = TaskStatus.PARTIAL
            stop_reason = fatal_error or "no usable evidence found within local budget"
        local.status = status
        local.answer_summary = summary
        local.stop_reason = stop_reason
        local.claim_ids = list(claims)
        local.evidence_ids = list(evidence)
        local.source_ids = list(sources)
        local.version += 1
        local.updated_at = utc_now()
        self.store.save_local(local)

        result = AgentResult(
            subtask_id=subtask.id,
            status=status,
            answer_summary=summary,
            claim_ids=list(claims),
            evidence_ids=list(evidence),
            source_ids=list(sources),
            unresolved_gaps=local.gaps,
            resolved_gaps=local.resolved_gaps,
            conflicts=local.conflicts,
            suggested_followups=string_list(last_decision.get("suggested_followups"), 10),
            usage=usage,
            error=fatal_error,
        )
        bundle = ResearchExecutionBundle(
            result=result,
            local_state=local,
            sources=list(sources.values()),
            evidence=list(evidence.values()),
            claims=list(claims.values()),
            events=events,
        )
        self.store.save_bundle(state.run_id, bundle)
        self.store.clear_research_checkpoint(state.run_id, subtask.id)
        return bundle


def build_semantically_bounded_prompt(
    *,
    builder: Callable[[int, int], str],
    token_counter: TokenCounter,
    system_prompt: str,
    max_model_len: int,
    max_output_tokens: int,
    context_safety_tokens: int,
    state_max_chars: int,
    evidence_max_chars: int,
) -> tuple[str, int]:
    """Shrink structured prompt sections before the gateway's final guard.

    Each rebuild keeps complete evidence objects and valid enclosing prompt
    sections. This avoids relying on a raw middle truncation for normal Writer
    and Auditor overflows.
    """

    max_input_tokens = (
        max(1, int(max_model_len))
        - max(1, int(max_output_tokens))
        - max(1, int(context_safety_tokens))
    )
    if max_input_tokens <= 0:
        raise ContextWindowExceeded("Model window leaves no room for structured Agent input")

    state_chars = max(1000, int(state_max_chars))
    evidence_chars = max(1000, int(evidence_max_chars))
    min_state_chars = min(state_chars, 4000)
    min_evidence_chars = min(evidence_chars, 6000)
    prompt = ""
    estimated_tokens = 0
    for _ in range(12):
        prompt = builder(state_chars, evidence_chars)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        estimated_tokens = token_counter.count_messages(messages)
        if estimated_tokens <= max_input_tokens:
            return prompt, estimated_tokens

        ratio = min(0.85, max(0.2, (max_input_tokens / estimated_tokens) * 0.9))
        next_state_chars = max(min_state_chars, int(state_chars * ratio))
        next_evidence_chars = max(min_evidence_chars, int(evidence_chars * ratio))
        if next_state_chars >= state_chars and state_chars > min_state_chars:
            next_state_chars = max(min_state_chars, state_chars - 1000)
        if next_evidence_chars >= evidence_chars and evidence_chars > min_evidence_chars:
            next_evidence_chars = max(min_evidence_chars, evidence_chars - 1000)
        if next_state_chars == state_chars and next_evidence_chars == evidence_chars:
            break
        state_chars = next_state_chars
        evidence_chars = next_evidence_chars

    # The generic gateway will preserve the protocol prefix/suffix if even the
    # minimum semantic view does not fit (for example, an exceptionally long
    # original user task). Return the measured size so it can make that choice.
    return prompt, estimated_tokens


class ReportWriter:
    def __init__(
        self,
        *,
        llm: Any,
        max_tokens: int,
        temperature: float,
        state_prompt_max_chars: int = 30000,
        evidence_prompt_max_chars: int = 52000,
        token_counter: TokenCounter | None = None,
        max_model_len: int = 32768,
        context_safety_tokens: int = 512,
    ) -> None:
        self.llm = llm
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.state_prompt_max_chars = max(1000, int(state_prompt_max_chars))
        self.evidence_prompt_max_chars = max(1000, int(evidence_prompt_max_chars))
        self.token_counter = token_counter or TokenCounter()
        self.max_model_len = max(1, int(max_model_len))
        self.context_safety_tokens = max(1, int(context_safety_tokens))

    async def write(
        self,
        state: GlobalResearchState,
    ) -> tuple[str, list[dict[str, Any]], dict[str, int], bool]:
        packet = build_evidence_packet(state)
        prompt, estimated_input_tokens = build_semantically_bounded_prompt(
            builder=lambda state_chars, evidence_chars: build_writer_prompt(
                state,
                packet,
                state_max_chars=state_chars,
                evidence_max_chars=evidence_chars,
            ),
            token_counter=self.token_counter,
            system_prompt=WRITER_SYSTEM_PROMPT,
            max_model_len=self.max_model_len,
            max_output_tokens=self.max_tokens,
            context_safety_tokens=self.context_safety_tokens,
            state_max_chars=self.state_prompt_max_chars,
            evidence_max_chars=self.evidence_prompt_max_chars,
        )
        try:
            response, usage = await call_chat(
                self.llm,
                user_prompt=prompt,
                system_prompt=WRITER_SYSTEM_PROMPT,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                role="writer",
                run_id=state.run_id,
                request_id=f"{state.run_id}:writer:{state.state_version}",
                estimated_input_tokens=estimated_input_tokens,
            )
        except Exception as exc:
            if not isinstance(exc, ContextWindowExceeded) and not is_context_length_error(exc):
                raise
            response = ""
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        article = str(response or "").strip()
        used_fallback = not bool(article)
        if not article:
            article = fallback_report(state, packet)
        return article, packet, usage, used_fallback


class CitationAuditor:
    def __init__(
        self,
        *,
        llm: Any,
        max_tokens: int,
        max_repair_tasks: int,
        temperature: float = 0.0,
        state_prompt_max_chars: int = 30000,
        evidence_prompt_max_chars: int = 42000,
        token_counter: TokenCounter | None = None,
        max_model_len: int = 32768,
        context_safety_tokens: int = 512,
    ) -> None:
        self.llm = llm
        self.max_tokens = max_tokens
        self.max_repair_tasks = max_repair_tasks
        self.temperature = temperature
        self.state_prompt_max_chars = max(1000, int(state_prompt_max_chars))
        self.evidence_prompt_max_chars = max(1000, int(evidence_prompt_max_chars))
        self.token_counter = token_counter or TokenCounter()
        self.max_model_len = max(1, int(max_model_len))
        self.context_safety_tokens = max(1, int(context_safety_tokens))

    async def audit(
        self,
        state: GlobalResearchState,
        evidence_packet: list[dict[str, Any]],
    ) -> tuple[AuditResult, str, dict[str, int]]:
        prompt, estimated_input_tokens = build_semantically_bounded_prompt(
            builder=lambda state_chars, evidence_chars: build_audit_prompt(
                state,
                evidence_packet,
                max_repair_tasks=self.max_repair_tasks,
                state_max_chars=state_chars,
                evidence_max_chars=evidence_chars,
            ),
            token_counter=self.token_counter,
            system_prompt=AUDITOR_SYSTEM_PROMPT,
            max_model_len=self.max_model_len,
            max_output_tokens=self.max_tokens,
            context_safety_tokens=self.context_safety_tokens,
            state_max_chars=self.state_prompt_max_chars,
            evidence_max_chars=self.evidence_prompt_max_chars,
        )
        try:
            response, usage = await call_chat(
                self.llm,
                user_prompt=prompt,
                system_prompt=AUDITOR_SYSTEM_PROMPT,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                role="auditor",
                run_id=state.run_id,
                request_id=f"{state.run_id}:auditor:{state.audit_round}:{state.state_version}",
                response_schema=AUDIT_SCHEMA,
                schema_name="citation_audit",
                estimated_input_tokens=estimated_input_tokens,
            )
        except Exception as exc:
            if not isinstance(exc, ContextWindowExceeded) and not is_context_length_error(exc):
                raise
            response = ""
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        parsed = parse_object(response)
        if parsed is None:
            audit = fallback_audit(state)
        else:
            parsed["passed"] = safe_bool(parsed.get("passed"))
            parsed["issues"] = [item for item in parsed.get("issues", []) if isinstance(item, dict)]
            normalized_repairs: list[dict[str, Any]] = []
            for item in parsed.get("repair_tasks", []):
                if not isinstance(item, dict):
                    continue
                repair = dict(item)
                repair_kind = str(repair.get("repair_kind", "research")).strip().lower()
                repair["repair_kind"] = repair_kind if repair_kind in {"research", "rewrite"} else "research"
                repair["requires_search"] = (
                    safe_bool(repair["requires_search"])
                    if "requires_search" in repair
                    else repair["repair_kind"] != "rewrite"
                )
                normalized_repairs.append(repair)
            parsed["repair_tasks"] = normalized_repairs[: self.max_repair_tasks]
            audit = AuditResult.from_dict(parsed)

        exact_urls = {
            canonicalize_url(source.url): source.url
            for source in state.sources.values()
            if canonicalize_url(source.url)
        }
        cited_urls = {
            canonicalize_url(url): url for url in extract_urls(state.article) if canonicalize_url(url)
        }
        unknown_urls = sorted(cited_urls[key] for key in cited_urls.keys() - exact_urls.keys())
        if unknown_urls:
            audit.passed = False
            audit.issues.append(
                {
                    "severity": "critical",
                    "claim": "citation URL validity",
                    "reason": f"draft cites URLs outside the evidence store: {unknown_urls[:5]}",
                    "evidence_ids": [],
                }
            )
            if len(audit.repair_tasks) < self.max_repair_tasks:
                audit.repair_tasks.append(
                    {
                        "objective": "Replace unregistered citation URLs with evidence-store sources or remove the unsupported statements.",
                        "coverage_targets": ["audit:citation_url_validity"],
                        "repair_kind": "rewrite",
                        "requires_search": False,
                    }
                )
        if state.claims and not cited_urls.keys() & exact_urls.keys():
            audit.passed = False
            audit.issues.append(
                {
                    "severity": "critical",
                    "claim": "report citation coverage",
                    "reason": "no exact evidence URL is cited in the draft",
                    "evidence_ids": [],
                }
            )
            if len(audit.repair_tasks) < self.max_repair_tasks:
                audit.repair_tasks.append(
                    {
                        "objective": "Verify the report's major factual claims and provide exact source URLs for citation repair.",
                        "coverage_targets": ["audit:citation_coverage"],
                        "repair_kind": "rewrite",
                        "requires_search": False,
                    }
                )
        if not state.article.strip():
            audit.passed = False
            audit.issues.append(
                {"severity": "critical", "claim": "draft", "reason": "draft is empty", "evidence_ids": []}
            )
        return audit, response, usage


def parse_object(response: str) -> dict[str, Any] | None:
    try:
        parsed = extract_json(response)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def fallback_initial_plan(task: dict[str, Any], raw_response: str, max_tasks: int) -> InitialPlanResult:
    prompt = str(task.get("prompt", ""))
    language = str(task.get("language", "en"))
    if language == "zh":
        objectives = [
            f"澄清研究问题的核心定义、范围、背景和关键参与者：{prompt}",
            f"寻找回答研究问题所需的一手事实、数据、案例和方法依据：{prompt}",
            f"研究相反证据、风险、限制、争议和不同利益相关方视角：{prompt}",
        ]
        coverage = ["定义与范围", "事实与数据", "风险与反例"]
    else:
        objectives = [
            f"Clarify the core definitions, scope, background, and actors for: {prompt}",
            f"Find primary facts, data, cases, and methodological evidence needed to answer: {prompt}",
            f"Investigate counterevidence, risks, limitations, disputes, and stakeholder perspectives for: {prompt}",
        ]
        coverage = ["definitions_and_scope", "facts_and_data", "risks_and_counterevidence"]
    objectives = objectives[: max(1, max_tasks)]
    tasks = [
        {
            "id": f"st_{index:02d}",
            "task_type": "research",
            "objective": objective,
            "coverage_targets": [coverage[index - 1]],
            "depends_on": [],
            "priority": 80 - index,
            "required_source_types": ["primary", "independent"],
        }
        for index, objective in enumerate(objectives, 1)
    ]
    brief = ResearchBrief(
        question=prompt,
        language=language,
        scope=str(task.get("topic", "")),
        deliverables=["evidence-grounded research report"],
        coverage_targets=coverage[: len(tasks)],
        critical_questions=objectives,
        source_policy=["prefer primary sources", "cross-check critical claims"],
        ambiguities=[],
    )
    return InitialPlanResult(brief, tasks, {"mode": "ON_WAVE_OR_CONFLICT"}, raw_response, True)


def fallback_replan(state: GlobalResearchState) -> dict[str, Any]:
    pending = any(task.status in {TaskStatus.PENDING, TaskStatus.RUNNING} for task in state.tasks.values())
    if pending:
        action = "continue"
        reason = "existing DAG still has executable work"
    elif state.claims:
        action = "write"
        reason = "available evidence can support a qualified report"
    else:
        action = "partial"
        reason = "no usable evidence was collected"
    return {
        "base_state_version": state.state_version,
        "action": action,
        "reason": reason,
        "operations": [],
    }


def fallback_researcher_decision(
    local: LocalResearchState,
    claims: dict[str, ClaimRecord],
    subtask: SubTask,
) -> dict[str, Any]:
    if claims:
        return {
            "base_local_version": local.version,
            "assessment": {"coverage": "partial", "primary_gap": "model output was not parseable"},
            "actions": [],
            "add_gaps": ["Researcher response was not parseable; result synthesized from tool evidence."],
            "answer_summary": summarize_claims(claims),
            "finish": True,
            "stop_reason": "fallback synthesis",
        }
    return {
        "base_local_version": local.version,
        "assessment": {"coverage": "none", "primary_gap": "initial evidence search"},
        "actions": [{"type": "SEARCH", "query": subtask.objective, "reason": "fallback query"}],
        "add_gaps": [],
        "answer_summary": "",
        "finish": False,
    }


def extract_search_queries(decision: dict[str, Any], limit: int) -> list[str]:
    actions = decision.get("actions", [])
    if not isinstance(actions, list):
        return []
    values: list[str] = []
    for action in actions:
        if not isinstance(action, dict) or str(action.get("type", "")).upper() != "SEARCH":
            continue
        query = str(action.get("query", "")).strip()
        if query:
            values.append(query)
    return string_list(values, limit)


def add_local_semantic_patch(local: LocalResearchState, decision: dict[str, Any]) -> None:
    assessment = decision.get("assessment", {})
    if isinstance(assessment, dict):
        coverage = str(assessment.get("coverage", "none")).lower()
        primary_gap = str(assessment.get("primary_gap", "")).strip()
        if coverage == "sufficient":
            local.gaps = []
        elif primary_gap:
            append_semantic_unique(local.gaps, [primary_gap], max_items=40)
    resolved = string_list(decision.get("resolved_gaps"), 20)
    if resolved:
        append_semantic_unique(local.resolved_gaps, resolved, max_items=40)
        local.gaps = [
            gap
            for gap in local.gaps
            if not any(texts_semantically_equivalent(gap, item) for item in resolved)
        ]
    added_gaps = string_list(decision.get("add_gaps"), 20)
    if added_gaps:
        local.resolved_gaps = [
            gap
            for gap in local.resolved_gaps
            if not any(texts_semantically_equivalent(gap, item) for item in added_gaps)
        ]
    append_semantic_unique(local.gaps, added_gaps, max_items=40)
    conflicts = decision.get("add_conflicts", [])
    if isinstance(conflicts, list):
        local.conflicts.extend(item for item in conflicts if isinstance(item, dict))


def add_tool_result_to_local(local: LocalResearchState, queries: list[str], result: Any) -> None:
    append_unique(local.queries, queries)
    append_unique(local.source_ids, [source.id for source in result.sources])
    append_unique(local.evidence_ids, [item.id for item in result.evidence])
    append_unique(local.claim_ids, [item.id for item in result.claims])
    local.tool_calls += int(result.usage.get("tool_calls", 0))
    local.search_calls += int(result.usage.get("search_calls", 0))
    local.recent_observations.extend(result.observations)
    local.recent_observations = local.recent_observations[-8:]
    for observation in result.observations:
        conflicts = observation.get("conflicts", [])
        if isinstance(conflicts, list):
            local.conflicts.extend(item for item in conflicts if isinstance(item, dict))


def append_unique(target: list[str], values: list[str]) -> None:
    seen = {normalize_text(value) for value in target}
    for value in values:
        key = normalize_text(value)
        if key and key not in seen:
            target.append(value)
            seen.add(key)


def append_semantic_unique(target: list[str], values: list[str], *, max_items: int) -> None:
    for value in values:
        text = str(value or "").strip()
        if not text or any(texts_semantically_equivalent(text, existing) for existing in target):
            continue
        target.append(text)
        if len(target) >= max(1, int(max_items)):
            break


def summarize_claims(claims: dict[str, ClaimRecord]) -> str:
    if not claims:
        return "No usable source-grounded claims were found."
    return "\n".join(f"- {claim.text}" for claim in list(claims.values())[:20])


def build_global_context_slice(state: GlobalResearchState, subtask: SubTask) -> dict[str, Any]:
    claims: list[dict[str, Any]] = []
    for claim in list(state.claims.values())[-40:]:
        source_urls = []
        for evidence_id in claim.evidence_ids:
            evidence = state.evidence.get(evidence_id)
            source = state.sources.get(evidence.source_id) if evidence else None
            if source and source.url and source.url not in source_urls:
                source_urls.append(source.url)
        claims.append(
            {
                "id": claim.id,
                "text": claim.text,
                "confidence": claim.confidence,
                "source_urls": source_urls,
            }
        )
    return {
        "subtask_coverage_targets": subtask.coverage_targets,
        "global_query_ledger": state.query_ledger[-128:],
        "existing_claims": claims,
        "global_gaps": state.gaps[-20:],
        "global_conflicts": state.conflicts[-20:],
        "coverage": state.coverage,
    }


def build_evidence_packet(state: GlobalResearchState) -> list[dict[str, Any]]:
    packet: list[dict[str, Any]] = []
    for claim in state.claims.values():
        evidence_items: list[dict[str, Any]] = []
        by_relation: dict[str, list[dict[str, Any]]] = {
            "supports": [],
            "refutes": [],
            "qualifies": [],
        }
        for evidence_id in claim.evidence_ids:
            evidence = state.evidence.get(evidence_id)
            if evidence is None:
                continue
            source = state.sources.get(evidence.source_id)
            if source is None:
                continue
            item = {
                "evidence_id": evidence.id,
                "relation": evidence.relation,
                "excerpt": evidence.excerpt,
                "locator": evidence.locator,
                "confidence": evidence.confidence,
                "source_id": source.id,
                "source_title": source.title,
                "source_url": source.url,
                "publish_date": source.publish_date,
                "source_quality": source.source_quality,
                "source_type": source.source_type,
                "authority_score": source.authority_score,
                "independence_group": source.independence_group,
            }
            evidence_items.append(item)
            by_relation.setdefault(evidence.relation, []).append(item)
        if evidence_items:
            task = state.tasks.get(claim.subtask_id)
            packet.append(
                {
                    "claim_id": claim.id,
                    "claim": claim.text,
                    "confidence": claim.confidence,
                    "status": claim.status,
                    "qualifiers": claim.qualifiers,
                    "subtask_id": claim.subtask_id,
                    "subtask_objective": task.objective if task else "",
                    "coverage_targets": list(task.coverage_targets) if task else [],
                    "evidence": evidence_items,
                    "supports": by_relation["supports"],
                    "refutes": by_relation["refutes"],
                    "qualifies": by_relation["qualifies"],
                }
            )
    return order_evidence_packet(state, packet)


def order_evidence_packet(
    state: GlobalResearchState,
    packet: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rank evidence quality while round-robining across deliverable coverage."""

    def score(item: dict[str, Any]) -> float:
        status_score = {
            "corroborated": 5.0,
            "qualified": 3.0,
            "single_source": 2.0,
            "contested": 1.5,
            "disputed": 0.5,
        }.get(str(item.get("status", "")), 1.0)
        evidence_items = item.get("evidence", []) if isinstance(item.get("evidence"), list) else []
        authority = max(
            (float(value.get("authority_score", 0.0)) for value in evidence_items if isinstance(value, dict)),
            default=0.0,
        )
        independent_groups = {
            str(value.get("independence_group", ""))
            for value in evidence_items
            if isinstance(value, dict) and value.get("independence_group")
        }
        quantitative = 0.75 if re.search(r"\d", str(item.get("claim", ""))) else 0.0
        return status_score + authority * 3.0 + min(2, len(independent_groups)) * 0.5 + quantitative

    groups: dict[str, list[dict[str, Any]]] = {}
    for item in packet:
        targets = item.get("coverage_targets", [])
        group = str(targets[0]) if isinstance(targets, list) and targets else "__unassigned__"
        item["writer_priority"] = round(score(item), 3)
        groups.setdefault(group, []).append(item)
    for values in groups.values():
        values.sort(key=lambda item: (-float(item.get("writer_priority", 0.0)), str(item.get("claim_id", ""))))

    target_order = list(state.coverage)
    for target in groups:
        if target not in target_order:
            target_order.append(target)
    output: list[dict[str, Any]] = []
    while any(groups.values()):
        for target in target_order:
            values = groups.get(target, [])
            if values:
                output.append(values.pop(0))
    return output


def fallback_report(state: GlobalResearchState, packet: list[dict[str, Any]]) -> str:
    language = state.brief.language if state.brief else str(state.task.get("language", "en"))
    if language == "zh":
        lines = ["# 研究报告", "", "## 结论摘要", ""]
        if not packet:
            lines.append("当前预算内未获得足够的可核验证据，以下不能形成可靠结论。")
        for item in packet:
            positive = item["supports"] or item["qualifies"]
            if positive:
                lines.append(f"- {item['claim']}（来源：{positive[0]['source_url']}）")
            elif item["refutes"]:
                lines.append(f"- 现有证据反驳或质疑“{item['claim']}”（来源：{item['refutes'][0]['source_url']}）")
        lines.extend(["", "## 限制", "", "本报告仅使用系统已采集并登记的证据。"])
        return "\n".join(lines)
    lines = ["# Research Report", "", "## Executive summary", ""]
    if not packet:
        lines.append("The available research budget did not yield enough verifiable evidence for a reliable conclusion.")
    for item in packet:
        positive = item["supports"] or item["qualifies"]
        if positive:
            lines.append(f"- {item['claim']} (Source: {positive[0]['source_url']})")
        elif item["refutes"]:
            lines.append(
                f"- Available evidence refutes or disputes “{item['claim']}” "
                f"(Source: {item['refutes'][0]['source_url']})"
            )
    lines.extend(["", "## Limitations", "", "This report uses only evidence collected and registered by the research system."])
    return "\n".join(lines)


def fallback_audit(state: GlobalResearchState) -> AuditResult:
    exact_urls = {canonicalize_url(source.url) for source in state.sources.values() if source.url}
    cited_urls = {canonicalize_url(url) for url in extract_urls(state.article)}
    passed = bool(state.article.strip() and state.claims and exact_urls.intersection(cited_urls))
    issues = [] if passed else [
        {
            "severity": "critical",
            "claim": "report evidence coverage",
            "reason": "the auditor response was unparseable and deterministic citation checks did not pass",
            "evidence_ids": [],
        }
    ]
    repairs = [] if passed else [
        {
            "objective": "Verify major draft claims and obtain exact source URLs for unsupported statements.",
            "coverage_targets": ["audit:deterministic_check"],
        }
    ]
    return AuditResult(passed=passed, issues=issues, repair_tasks=repairs, summary="fallback audit")


def safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "pass", "passed"}
