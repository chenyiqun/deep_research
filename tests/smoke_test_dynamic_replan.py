from __future__ import annotations

import asyncio
import json

from drb_qwen.deep_research_workflow import AsyncDeepResearchWorkflow, DeepResearchConfig
from drb_qwen.multi_agent.schemas import (
    AuditResult,
    GlobalResearchState,
    RunPhase,
    SubTask,
    TaskStatus,
)
from drb_qwen.web_search import SearchResult


class DynamicFakeLLM:
    def __init__(self) -> None:
        self.replan_calls = 0
        self.writer_calls = 0
        self.audit_calls = 0

    async def chat(
        self,
        user_prompt: str,
        system_prompt: str = "",
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 4096,
    ) -> str:
        if "<protocol>MAIN_PLAN_V1</protocol>" in user_prompt:
            return json.dumps(
                {
                    "research_brief": {
                        "question": "dynamic research",
                        "language": "en",
                        "coverage_targets": ["base"],
                        "critical_questions": ["base evidence"],
                    },
                    "tasks": [
                        {
                            "id": "base",
                            "objective": "Research the base evidence.",
                            "coverage_targets": ["base"],
                            "depends_on": [],
                            "max_steps": 2,
                            "max_tool_calls": 3,
                        }
                    ],
                }
            )
        if "<protocol>RESEARCHER_STEP_V1</protocol>" in user_prompt:
            if '"evidence_ids": []' in user_prompt:
                if '"id": "repair_1_1"' in user_prompt:
                    query = "repair audit evidence"
                elif '"id": "followup"' in user_prompt:
                    query = "followup evidence"
                else:
                    query = "base evidence"
                return json.dumps(
                    {
                        "base_local_version": 0,
                        "assessment": {"coverage": "none", "primary_gap": "evidence"},
                        "actions": [{"type": "SEARCH", "query": query}],
                        "finish": False,
                    }
                )
            return json.dumps(
                {
                    "base_local_version": 1,
                    "assessment": {"coverage": "sufficient", "primary_gap": ""},
                    "actions": [],
                    "answer_summary": "Evidence collected.",
                    # The runtime should treat an explicit sufficient
                    # assessment as terminal even if the model omits finish.
                    "finish": False,
                }
            )
        if "<protocol>READER_EXTRACT_V1</protocol>" in user_prompt:
            if "repair-source" in user_prompt:
                label = "repair"
            elif "followup-source" in user_prompt:
                label = "followup"
            else:
                label = "base"
            return json.dumps(
                {
                    "relevance": 1.0,
                    "claims": [
                        {
                            "text": f"The {label} source supports the {label} research claim.",
                            "excerpt": f"Evidence excerpt for {label}-source.",
                            "confidence": "medium",
                            "relation": "supports",
                        }
                    ],
                    "gaps": [],
                    "conflicts": [],
                    "limitations": [],
                }
            )
        if "<protocol>MAIN_REPLAN_V1</protocol>" in user_prompt:
            self.replan_calls += 1
            if self.replan_calls == 1:
                return json.dumps(
                    {
                        "action": "continue",
                        "reason": "a follow-up gap remains",
                        "operations": [
                            {
                                "op": "ADD_TASK",
                                "task": {
                                    "id": "followup",
                                    "task_type": "verify",
                                    "objective": "Verify the follow-up evidence gap.",
                                    "coverage_targets": ["followup"],
                                    "depends_on": ["base"],
                                    "priority": 90,
                                    "max_steps": 2,
                                    "max_tool_calls": 3,
                                },
                            }
                        ],
                    }
                )
            if self.replan_calls == 2:
                return json.dumps({"action": "write", "reason": "research complete", "operations": []})
            return json.dumps({"action": "continue", "reason": "repair audit failure", "operations": []})
        if "<protocol>WRITER_V1</protocol>" in user_prompt:
            self.writer_calls += 1
            article = (
                "# Report\n\nEvidence is available from "
                "https://example.com/base-source and https://example.com/followup-source."
            )
            if self.writer_calls > 1:
                article += " Repair evidence: https://example.com/repair-source."
            return article
        if "<protocol>AUDITOR_V1</protocol>" in user_prompt:
            self.audit_calls += 1
            if self.audit_calls == 1:
                return json.dumps(
                    {
                        "passed": False,
                        "summary": "one claim needs independent repair",
                        "issues": [
                            {"severity": "major", "claim": "repair claim", "reason": "missing evidence", "evidence_ids": []}
                        ],
                        "repair_tasks": [
                            {
                                "objective": "Find independent evidence for the missing audit claim.",
                                "coverage_targets": ["audit:repair"],
                            }
                        ],
                    }
                )
            return json.dumps({"passed": True, "summary": "repair passed", "issues": [], "repair_tasks": []})
        raise AssertionError(f"unexpected prompt: {user_prompt[:120]}")


class DynamicSearchClient:
    async def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if "repair" in query:
            slug = "repair-source"
        elif "followup" in query:
            slug = "followup-source"
        else:
            slug = "base-source"
        return [
            SearchResult(
                title=slug,
                content=f"Evidence excerpt for {slug}.",
                link=f"https://example.com/{slug}",
                search_query=query,
            )
        ]


class RewriteOnlyAuditor:
    async def audit(self, state: GlobalResearchState, evidence_packet: list[dict]) -> tuple[AuditResult, str, dict]:
        return (
            AuditResult(
                passed=False,
                issues=[{"severity": "major", "claim": "draft", "reason": "citation formatting"}],
                repair_tasks=[
                    {
                        "objective": "repair citation formatting",
                        "repair_kind": "rewrite",
                        "requires_search": False,
                    }
                ],
            ),
            "{}",
            {},
        )


async def main_async() -> None:
    llm = DynamicFakeLLM()
    workflow = AsyncDeepResearchWorkflow(
        llm=llm,
        search_client=DynamicSearchClient(),
        content_fetcher=None,
        config=DeepResearchConfig(
            max_rounds=3,
            max_initial_tasks=1,
            max_researchers=2,
            max_react_steps=2,
            max_search_queries_per_round=1,
            max_tool_calls_per_subtask=3,
            max_total_tool_calls=20,
            search_top_k=1,
            fetch_full_content=False,
            min_total_claims=3,
            min_coverage_ratio=0.5,
            max_audit_rounds=2,
            citation_audit_enabled=True,
        ),
    )
    deferred_state = GlobalResearchState(
        run_id="deferred",
        task={"prompt": "deferred"},
        phase=RunPhase.RESEARCHING,
        tasks={
            "done": SubTask(id="done", objective="done", status=TaskStatus.COMPLETED),
            "dependent": SubTask(
                id="dependent",
                objective="existing dependent work",
                depends_on=["done"],
            ),
        },
    )
    deferred_trace: list[dict] = []
    await workflow._strategic_boundary(deferred_state, deferred_trace)
    assert llm.replan_calls == 0
    assert deferred_state.phase == RunPhase.RESEARCHING
    assert deferred_trace[-1]["type"] == "main_replan_deferred"
    assert workflow._research_limits(deferred_state) == (16, 24)
    assert workflow._token_limit(deferred_state) == 850_000
    deferred_state.audit_round = 1
    deferred_state.audit = AuditResult(passed=False)
    assert workflow._research_limits(deferred_state) == (20, 30)
    assert workflow._token_limit(deferred_state) == 1_000_000

    rewrite_state = GlobalResearchState(
        run_id="rewrite-final",
        task={"prompt": "rewrite"},
        phase=RunPhase.AUDITING,
        audit_round=1,
        article="draft",
    )
    original_auditor = workflow.auditor
    workflow.auditor = RewriteOnlyAuditor()  # type: ignore[assignment]
    await workflow._audit_phase(rewrite_state, [])
    workflow.auditor = original_auditor
    assert rewrite_state.audit_round == 2
    assert rewrite_state.phase == RunPhase.WRITING

    result = await workflow.run({"id": 200, "language": "en", "prompt": "dynamic research"})
    assert result["state"]["phase"] == "completed"
    assert set(result["state"]["tasks"]) == {"base", "followup", "repair_1_1"}
    assert result["state"]["audit_round"] == 2
    assert len(result["state"]["claims"]) == 3, result["state"]
    assert llm.replan_calls == 3
    assert llm.writer_calls == 2
    assert llm.audit_calls == 2
    event_types = [event["type"] for event in result["trace"]]
    assert "audit_repair_planned" in event_types
    assert event_types.count("scheduler_wave_completed") == 3
    print("smoke_test_dynamic_replan passed")


if __name__ == "__main__":
    asyncio.run(main_async())
