from __future__ import annotations

import asyncio
import json
from tempfile import TemporaryDirectory

from drb_qwen.async_llm_client import AsyncChatResponse
from drb_qwen.deep_research_workflow import AsyncDeepResearchWorkflow, DeepResearchConfig
from drb_qwen.generate_reports_async_research import summarize_research_result
from drb_qwen.multi_agent.schemas import (
    AgentResult,
    ClaimRecord,
    EvidenceRecord,
    GlobalResearchState,
    LocalResearchState,
    ResearchBrief,
    ResearchExecutionBundle,
    RunPhase,
    SourceRecord,
    SubTask,
    TaskStatus,
)
from drb_qwen.url_fetcher import URLFetchResult
from drb_qwen.web_search import SearchResult


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def chat(
        self,
        user_prompt: str,
        system_prompt: str = "",
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 4096,
    ) -> str:
        if "<protocol>MAIN_PLAN_V1</protocol>" in user_prompt:
            self.calls.append("plan")
            return json.dumps(
                {
                    "research_brief": {
                        "question": "How does AI interaction affect interpersonal relations?",
                        "language": "en",
                        "scope": "social effects",
                        "deliverables": ["evidence-grounded report"],
                        "coverage_targets": ["mechanisms", "evidence"],
                        "critical_questions": ["mechanisms", "observed evidence"],
                        "source_policy": ["use independent sources"],
                        "ambiguities": [],
                    },
                    "tasks": [
                        {
                            "id": "st_mechanisms",
                            "task_type": "research",
                            "objective": "Research mechanisms by which AI interaction changes relationships.",
                            "coverage_targets": ["mechanisms"],
                            "depends_on": [],
                            "priority": 90,
                            "max_steps": 2,
                            "max_tool_calls": 5,
                        },
                        {
                            "id": "st_evidence",
                            "task_type": "research",
                            "objective": "Find empirical evidence about AI interaction and relationships.",
                            "coverage_targets": ["evidence"],
                            "depends_on": [],
                            "priority": 80,
                            "max_steps": 2,
                            "max_tool_calls": 5,
                        },
                    ],
                    "wakeup_policy": {"mode": "ON_WAVE_OR_CONFLICT"},
                }
            )
        if "<protocol>RESEARCHER_STEP_V1</protocol>" in user_prompt:
            self.calls.append("researcher")
            if '"evidence_ids": []' in user_prompt:
                query = (
                    "AI relationship mechanisms evidence"
                    if "st_mechanisms" in user_prompt
                    else "AI interaction empirical relationship study"
                )
                return json.dumps(
                    {
                        "base_local_version": 0,
                        "assessment": {"coverage": "none", "primary_gap": "source evidence"},
                        "actions": [{"type": "SEARCH", "query": query, "reason": "find evidence"}],
                        "add_gaps": [],
                        "add_conflicts": [],
                        "suggested_followups": [],
                        "answer_summary": "",
                        "finish": False,
                        "stop_reason": "",
                    }
                )
            return json.dumps(
                {
                    "base_local_version": 1,
                    "assessment": {"coverage": "sufficient", "primary_gap": ""},
                    "actions": [],
                    "add_gaps": [],
                    "add_conflicts": [],
                    "suggested_followups": [],
                    "answer_summary": "The collected source provides relevant evidence for this subtask.",
                    "finish": True,
                    "stop_reason": "evidence sufficient",
                }
            )
        if "<protocol>READER_EXTRACT_V1</protocol>" in user_prompt:
            self.calls.append("reader")
            assert "FULL ARTICLE TEXT" in user_prompt
            is_mechanism = "relationship-mechanisms" in user_prompt
            return json.dumps(
                {
                    "relevance": 0.95,
                    "claims": [
                        {
                            "text": (
                                "AI companions can change perceived companionship needs."
                                if is_mechanism
                                else "Empirical studies report changes in perceived social support after AI interaction."
                            ),
                            "excerpt": "FULL ARTICLE TEXT reports a measurable relationship effect from AI interaction.",
                            "confidence": "medium",
                            "relation": "supports",
                            "locator": "paragraph 1",
                            "qualifiers": ["synthetic smoke source"],
                        }
                    ],
                    "gaps": [],
                    "conflicts": [],
                    "limitations": ["Synthetic smoke source."],
                    "injection_detected": False,
                }
            )
        if "<protocol>MAIN_REPLAN_V1</protocol>" in user_prompt:
            self.calls.append("replan")
            return json.dumps(
                {
                    "action": "write",
                    "reason": "both coverage targets have evidence",
                    "operations": [],
                }
            )
        if "<protocol>WRITER_V1</protocol>" in user_prompt:
            self.calls.append("writer")
            return (
                "# Executive summary\n\nAI interaction can affect social support and companionship needs "
                "(https://example.com/relationship-mechanisms; "
                "https://example.com/empirical-evidence)."
            )
        if "<protocol>AUDITOR_V1</protocol>" in user_prompt:
            self.calls.append("auditor")
            return json.dumps({"passed": True, "summary": "citations pass", "issues": [], "repair_tasks": []})
        raise AssertionError(f"Unexpected prompt: {user_prompt[:200]}")

    async def chat_with_usage(
        self,
        user_prompt: str,
        system_prompt: str = "",
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 4096,
    ) -> AsyncChatResponse:
        content = await self.chat(
            user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        return AsyncChatResponse(
            content=content,
            usage={"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        )


class FakeSearchClient:
    async def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        is_mechanism = "mechanisms" in query
        slug = "relationship-mechanisms" if is_mechanism else "empirical-evidence"
        return [
            SearchResult(
                title=slug,
                content="AI interaction can change companionship and perceived social support.",
                link=f"https://example.com/{slug}",
                media="example",
                publish_date="2026-01-01",
                search_query=query,
            )
        ][:top_k]


class FakeContentFetcher:
    async def fetch(self, url: str, goal: str = "") -> URLFetchResult:
        assert "Subtask" in goal
        return URLFetchResult(
            url=url,
            ok=True,
            status=200,
            content_type="text/html",
            final_url=url,
            text="FULL ARTICLE TEXT reports a measurable relationship effect from AI interaction.",
            extraction_method="html_test",
            raw_text_chars=82,
        )


async def main_async() -> None:
    with TemporaryDirectory() as state_dir:
        llm = FakeLLM()
        workflow = AsyncDeepResearchWorkflow(
            llm=llm,  # type: ignore[arg-type]
            search_client=FakeSearchClient(),  # type: ignore[arg-type]
            content_fetcher=FakeContentFetcher(),  # type: ignore[arg-type]
            config=DeepResearchConfig(
                max_rounds=3,
                max_initial_tasks=2,
                max_researchers=2,
                max_react_steps=2,
                max_search_queries_per_round=1,
                max_tool_calls_per_subtask=5,
                search_top_k=1,
                min_fetched_content_chars=10,
                min_total_claims=2,
                min_coverage_ratio=1.0,
                run_state_dir=state_dir,
                citation_audit_enabled=True,
            ),
        )
        task = {
            "id": 100,
            "language": "en",
            "topic": "Society",
            "prompt": "Write a paper to discuss the influence of AI interaction on interpersonal relations.",
        }
        result = await workflow.run(task)
        assert "Executive summary" in result["article"]
        assert result["state"]["phase"] == "completed"
        assert len(result["state"]["tasks"]) == 2
        assert len(result["state"]["claims"]) == 2
        assert len(result["state"]["evidence"]) == 2
        assert result["audit"]["passed"] is True
        assert result["diagnostics"]["query_calls"] == 2
        assert result["diagnostics"]["unique_queries"] == 2
        assert result["diagnostics"]["task_statuses"] == {"completed": 2}
        assert set(result["state"]["coverage_details"]) == {"mechanisms", "evidence"}
        assert result["state"]["budget"]["total_tokens"] == len(llm.calls) * 10
        assert result["inference"]["requests"] == len(llm.calls)
        summary = summarize_research_result(result)
        assert summary["subtasks"] == 2
        assert summary["state_evidence"] == 2
        assert summary["audit_passed"] is True
        assert summary["total_tokens"] == len(llm.calls) * 10
        event_types = [event["type"] for event in result["trace"]]
        assert "scheduler_wave_started" in event_types
        assert "subtask_merged" in event_types
        assert "main_replan" in event_types
        assert "citation_audit" in event_types
        researcher_events = [event["payload"] for event in result["trace"] if event["type"] == "researcher_step"]
        assert {(event["input_state_version"], event["output_state_version"]) for event in researcher_events} == {
            (0, 1),
            (1, 2),
        }
        assert all(event["estimated_input_tokens"] <= event["max_input_tokens"] for event in researcher_events)
        assert llm.calls.count("researcher") == 4
        assert workflow.store.load_bundle(result["run_id"], "st_mechanisms") is not None
        assert workflow.store.load_bundle(result["run_id"], "st_evidence") is not None

        calls_before_resume = len(llm.calls)
        events_before_resume = len(result["trace"])
        resumed = await workflow.run(task, resume=True)
        assert resumed["state"]["phase"] == "completed"
        assert len(llm.calls) == calls_before_resume
        assert len(resumed["trace"]) == events_before_resume

        try:
            await workflow.run(
                {**task, "prompt": "a different research question"},
                run_id=result["run_id"],
                resume=True,
            )
        except ValueError as exc:
            assert "different research task" in str(exc)
        else:
            raise AssertionError("resume must reject a run_id belonging to another task")

        budget_llm = FakeLLM()
        budget_workflow = AsyncDeepResearchWorkflow(
            llm=budget_llm,  # type: ignore[arg-type]
            search_client=FakeSearchClient(),  # type: ignore[arg-type]
            content_fetcher=FakeContentFetcher(),  # type: ignore[arg-type]
            config=DeepResearchConfig(
                max_rounds=3,
                max_initial_tasks=2,
                max_researchers=2,
                max_react_steps=2,
                max_search_queries_per_round=1,
                max_tool_calls_per_subtask=5,
                max_total_tool_calls=10,
                max_total_searches=1,
                search_top_k=1,
                min_fetched_content_chars=10,
                min_total_claims=1,
                min_coverage_ratio=0.5,
                run_state_dir=f"{state_dir}/search_budget",
                citation_audit_enabled=False,
            ),
        )
        budget_result = await budget_workflow.run({**task, "id": 101})
        assert budget_result["state"]["budget"]["search_calls"] == 1
        assert sum(
            item["status"] in {"completed", "partial", "failed"}
            for item in budget_result["state"]["tasks"].values()
        ) == 1

        recovery_llm = FakeLLM()
        recovery_task = {**task, "id": 102}
        recovery_workflow = AsyncDeepResearchWorkflow(
            llm=recovery_llm,  # type: ignore[arg-type]
            search_client=FakeSearchClient(),  # type: ignore[arg-type]
            content_fetcher=None,
            config=DeepResearchConfig(
                max_rounds=2,
                max_initial_tasks=1,
                max_total_searches=1,
                max_total_tool_calls=3,
                min_total_claims=1,
                min_coverage_ratio=1.0,
                run_state_dir=f"{state_dir}/bundle_recovery",
                citation_audit_enabled=False,
                fetch_full_content=False,
            ),
        )
        recovery_state = GlobalResearchState(
            run_id="recover_bundle",
            task=recovery_task,
            phase=RunPhase.RESEARCHING,
            brief=ResearchBrief(question=recovery_task["prompt"], coverage_targets=["recovered"]),
            tasks={
                "st": SubTask(id="st", objective="recover completed work", coverage_targets=["recovered"])
            },
            main_round=1,
        )
        recovery_source = SourceRecord(
            id="src",
            url="https://example.com/recovered",
            title="recovered",
            query="q",
            independence_group="example.com",
        )
        recovery_evidence = EvidenceRecord(
            id="ev",
            source_id="src",
            subtask_id="st",
            claim_text="Recovered claim.",
            excerpt="Recovered evidence.",
        )
        recovery_claim = ClaimRecord(
            id="claim",
            text="Recovered claim.",
            subtask_id="st",
            evidence_ids=["ev"],
        )
        recovery_bundle = ResearchExecutionBundle(
            result=AgentResult(
                subtask_id="st",
                status=TaskStatus.COMPLETED,
                answer_summary="Recovered claim.",
                source_ids=["src"],
                evidence_ids=["ev"],
                claim_ids=["claim"],
                usage={"search_calls": 1, "reader_calls": 1, "tool_calls": 2},
            ),
            local_state=LocalResearchState(
                run_id="recover_bundle",
                subtask_id="st",
                objective="recover completed work",
                status=TaskStatus.COMPLETED,
            ),
            sources=[recovery_source],
            evidence=[recovery_evidence],
            claims=[recovery_claim],
        )
        recovery_workflow.store.save_global(recovery_state)
        recovery_workflow.store.save_bundle(recovery_state.run_id, recovery_bundle)
        recovered_result = await recovery_workflow.run(
            recovery_task,
            run_id=recovery_state.run_id,
            resume=True,
        )
        assert "claim" in recovered_result["state"]["claims"]
        assert recovered_result["state"]["tasks"]["st"]["status"] == "completed"
        assert "researcher" not in recovery_llm.calls
        assert "cached_bundles_recovered" in [event["type"] for event in recovered_result["trace"]]
        print("smoke_test_async_workflow passed")


if __name__ == "__main__":
    asyncio.run(main_async())
