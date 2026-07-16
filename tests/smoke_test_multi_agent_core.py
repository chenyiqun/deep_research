from __future__ import annotations

from tempfile import TemporaryDirectory

from drb_qwen.multi_agent.dag import (
    DecisionValidationError,
    add_initial_tasks,
    apply_decision_patch,
    ready_tasks,
)
from drb_qwen.multi_agent.agents import build_evidence_packet
from drb_qwen.multi_agent.reducer import merge_research_bundle
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
from drb_qwen.multi_agent.security import (
    classify_source_authority,
    source_independence_group,
    validate_external_url,
)
from drb_qwen.multi_agent.store import RunStore
from drb_qwen.multi_agent.tools import (
    claim_preserves_scope,
    excerpt_is_grounded,
    normalize_confidence,
    source_quality,
)
from drb_qwen.url_fetcher import URLFetchResult
from drb_qwen.url_utils import canonicalize_url, extract_urls


def main() -> None:
    chinese_citations = (
        "第一条（http://example.com/a.html）。此后继续叙述；"
        "第二条见[官方来源](https://EXAMPLE.com:443/b?q=1)。"
    )
    assert extract_urls(chinese_citations) == [
        "http://example.com/a.html",
        "https://EXAMPLE.com:443/b?q=1",
    ]
    assert canonicalize_url("https://EXAMPLE.com:443/b?q=1）。") == "https://example.com/b?q=1"

    assert validate_external_url("https://example.com/report")[0] is True
    assert validate_external_url("http://127.0.0.1/admin")[0] is False
    assert validate_external_url("http://169.254.169.254/latest/meta-data")[0] is False
    assert validate_external_url("file:///etc/passwd")[0] is False
    assert source_independence_group("https://docs.example.com/a") == "example.com"
    assert source_independence_group("https://blog.example.com/b") == "example.com"
    assert source_independence_group("https://news.example.co.uk/a") == "example.co.uk"
    assert classify_source_authority("https://www.gov.cn/data")[0] == "official"
    assert classify_source_authority("https://baijiahao.baidu.com/s?id=1")[0] == "community"
    assert classify_source_authority(
        "https://baijiahao.baidu.com/s?id=2",
        "某公司年度报告解读",
    )[0] == "community"
    assert claim_preserves_scope("全国税收收入同比下降5.3%", "全国税收收入同比下降5.3%")
    assert not claim_preserves_scope(
        "地方税收收入同比下降5.3%",
        "全国税收收入同比下降5.3%",
    )
    assert not claim_preserves_scope("同比下降6%", "同比下降5.3%")

    state = GlobalResearchState(run_id="run_test", task={"prompt": "test"})
    state.brief = ResearchBrief(question="test", coverage_targets=["a", "b"])
    added = add_initial_tasks(
        state,
        [
            {"id": "a", "objective": "research a", "coverage_targets": ["a"]},
            {"id": "b", "objective": "research b", "depends_on": ["a"], "coverage_targets": ["b"]},
        ],
        max_subtasks=5,
        max_steps=3,
        max_tool_calls=9,
    )
    assert added == ["a", "b"]
    assert [task.id for task in ready_tasks(state, 5)] == ["a"]
    state.tasks["a"].status = TaskStatus.COMPLETED
    assert [task.id for task in ready_tasks(state, 5)] == ["b"]

    state.bump_version()
    result = apply_decision_patch(
        state,
        {
            "base_state_version": state.state_version,
            "operations": [
                {"op": "ADD_TASK", "task": {"id": "c", "objective": "verify c", "task_type": "verify"}},
                {"op": "ADD_DEPENDENCY", "from": "b", "to": "c"},
            ],
        },
        max_subtasks=5,
        max_steps=3,
        max_tool_calls=9,
    )
    assert result.added_task_ids == ["c"]
    assert state.tasks["c"].depends_on == ["b"]

    try:
        apply_decision_patch(
            state,
            {"base_state_version": state.state_version - 1, "operations": []},
            max_subtasks=5,
            max_steps=3,
            max_tool_calls=9,
        )
    except DecisionValidationError:
        pass
    else:
        raise AssertionError("stale DecisionPatch should be rejected")

    try:
        apply_decision_patch(
            state,
            {
                "base_state_version": state.state_version,
                "operations": [{"op": "ADD_DEPENDENCY", "from": "c", "to": "a"}],
            },
            max_subtasks=5,
            max_steps=3,
            max_tool_calls=9,
        )
    except DecisionValidationError:
        pass
    else:
        raise AssertionError("cyclic DecisionPatch should be rejected")

    limited = GlobalResearchState(run_id="limited", task={"prompt": "limit"})
    limited.bump_version()
    limited_result = apply_decision_patch(
        limited,
        {
            "base_state_version": limited.state_version,
            "operations": [
                {"op": "ADD_TASK", "task": {"id": "one", "objective": "first new task"}},
                {"op": "ADD_TASK", "task": {"id": "two", "objective": "second new task"}},
            ],
        },
        max_subtasks=10,
        max_steps=3,
        max_tool_calls=9,
        max_new_tasks=1,
    )
    assert limited_result.added_task_ids == ["one"]
    assert "two" not in limited.tasks
    assert any("max_new_tasks" in warning for warning in limited_result.warnings)

    deduped = GlobalResearchState(run_id="deduped", task={"prompt": "dedupe"})
    deduped.tasks["finance"] = SubTask(
        id="finance",
        objective="收集保险公司的融资和分红数据",
        coverage_targets=["融资情况", "实际分红"],
    )
    deduped.bump_version()
    duplicate_result = apply_decision_patch(
        deduped,
        {
            "base_state_version": deduped.state_version,
            "operations": [
                {
                    "op": "ADD_TASK",
                    "task": {
                        "id": "finance_en",
                        "task_type": "research",
                        "objective": "Collect financing and dividend data for the insurance companies",
                        "coverage_targets": ["financing", "dividend"],
                    },
                },
                {
                    "op": "ADD_TASK",
                    "task": {
                        "id": "finance_verify",
                        "task_type": "verify",
                        "objective": "收集保险公司的融资和分红数据",
                        "coverage_targets": ["融资情况"],
                    },
                },
            ],
        },
        max_subtasks=5,
        max_steps=3,
        max_tool_calls=9,
    )
    assert duplicate_result.added_task_ids == ["finance_verify"]
    assert any("duplicate objective" in warning for warning in duplicate_result.warnings)

    assert excerpt_is_grounded("Quoted   evidence.", "Before quoted evidence. After")
    assert not excerpt_is_grounded("invented evidence", "The source says something else.")
    no_fetch = URLFetchResult(url="https://example.com/sogou", ok=False)
    assert source_quality(no_fetch, False, "search_native_content") == "search_native_content"
    assert normalize_confidence("high", "search_native_content") == "medium"
    assert normalize_confidence("high", "search_native_content", "official", 0.95) == "high"
    assert normalize_confidence("high", "search_snippet") == "medium"

    resolved_state = GlobalResearchState(
        run_id="resolved",
        task={"prompt": "resolved"},
        gaps=["Missing annual revenue data"],
    )
    resolved_state.tasks["st"] = SubTask(id="st", objective="resolve gap")
    merge_research_bundle(
        resolved_state,
        ResearchExecutionBundle(
            result=AgentResult(
                subtask_id="st",
                status=TaskStatus.PARTIAL,
                answer_summary="gap handled",
                resolved_gaps=["Missing annual revenue data"],
            ),
            local_state=LocalResearchState(
                run_id="resolved",
                subtask_id="st",
                objective="resolve gap",
                resolved_gaps=["Missing annual revenue data"],
            ),
        ),
    )
    assert resolved_state.gaps == []

    integrity_state = GlobalResearchState(run_id="integrity", task={"prompt": "integrity"})
    integrity_state.tasks["st"] = SubTask(id="st", objective="verify")
    invalid_bundle = ResearchExecutionBundle(
        result=AgentResult(
            subtask_id="st",
            status=TaskStatus.COMPLETED,
            answer_summary="invalid",
            source_ids=["bad"],
            evidence_ids=["bad_ev"],
            claim_ids=["bad_claim"],
        ),
        local_state=LocalResearchState(run_id="integrity", subtask_id="st", objective="verify"),
        sources=[SourceRecord(id="bad", url="http://127.0.0.1/private", title="bad", query="q")],
        evidence=[
            EvidenceRecord(
                id="bad_ev",
                source_id="bad",
                subtask_id="st",
                claim_text="bad claim",
                excerpt="bad excerpt",
            )
        ],
        claims=[ClaimRecord(id="bad_claim", text="bad claim", subtask_id="st", evidence_ids=["bad_ev"])],
    )
    invalid_summary = merge_research_bundle(integrity_state, invalid_bundle)
    assert invalid_summary["rejected_sources"] == 1
    assert integrity_state.tasks["st"].status == TaskStatus.PARTIAL
    assert not integrity_state.claims

    relation_state = GlobalResearchState(run_id="relations", task={"prompt": "relations"})
    relation_state.tasks["st"] = SubTask(id="st", objective="verify")
    relation_bundle = ResearchExecutionBundle(
        result=AgentResult(
            subtask_id="st",
            status=TaskStatus.COMPLETED,
            answer_summary="contested",
            source_ids=["s1", "s2"],
            evidence_ids=["e1", "e2"],
            claim_ids=["claim"],
        ),
        local_state=LocalResearchState(run_id="relations", subtask_id="st", objective="verify"),
        sources=[
            SourceRecord(
                id="s1",
                url="https://example.com/a",
                title="one",
                query="q",
                independence_group="example.com",
            ),
            SourceRecord(
                id="s2",
                url="https://example.org/b",
                title="two",
                query="q",
                independence_group="example.org",
            ),
        ],
        evidence=[
            EvidenceRecord(
                id="e1",
                source_id="s1",
                subtask_id="st",
                claim_text="claim",
                excerpt="support",
                relation="supports",
            ),
            EvidenceRecord(
                id="e2",
                source_id="s2",
                subtask_id="st",
                claim_text="claim",
                excerpt="refutation",
                relation="refutes",
            ),
        ],
        claims=[ClaimRecord(id="claim", text="claim", subtask_id="st", evidence_ids=["e1", "e2"])],
    )
    merge_research_bundle(relation_state, relation_bundle)
    assert relation_state.claims["claim"].status == "contested"
    packet = build_evidence_packet(relation_state)
    assert len(packet[0]["supports"]) == 1
    assert len(packet[0]["refutes"]) == 1

    with TemporaryDirectory() as directory:
        store = RunStore(directory)
        active = GlobalResearchState(
            run_id="cancel_guard",
            task={"prompt": "cancel"},
            phase=RunPhase.RESEARCHING,
        )
        store.save_global(active)
        stale = GlobalResearchState.from_dict(active.to_dict())
        active.phase = RunPhase.CANCELLED
        active.stop_reason = "cancelled during a wave"
        active.bump_version()
        store.save_global(active)
        stale.bump_version()
        store.save_global(stale)
        persisted = store.load_global(active.run_id)
        assert persisted is not None
        assert persisted.phase == RunPhase.CANCELLED
        assert persisted.stop_reason == "cancelled during a wave"

    print("smoke_test_multi_agent_core passed")


if __name__ == "__main__":
    main()
