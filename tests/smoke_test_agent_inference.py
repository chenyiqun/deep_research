from __future__ import annotations

import asyncio
import json
from tempfile import TemporaryDirectory
from typing import Any

from drb_qwen.async_llm_client import AsyncChatConfig, AsyncChatResponse, LLMHTTPError
from drb_qwen.multi_agent.agents import REACT_STEP_TERMINAL, ResearcherAgent
from drb_qwen.multi_agent.context import ResearcherContextBuilder, TokenCounter
from drb_qwen.multi_agent.inference import AgentInferenceConfig, AgentInferenceGateway
from drb_qwen.multi_agent.protocols import RESEARCHER_DECISION_SCHEMA
from drb_qwen.multi_agent.schemas import (
    ClaimRecord,
    EvidenceRecord,
    GlobalResearchState,
    LocalResearchState,
    ResearchBrief,
    SourceRecord,
    SubTask,
    TaskStatus,
)
from drb_qwen.multi_agent.store import RunStore
from drb_qwen.multi_agent.tools import ToolBatchResult


class RecordingAdvancedLLM:
    supports_agent_inference_options = True

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    async def chat_with_usage(self, **kwargs: Any) -> AsyncChatResponse:
        self.kwargs = kwargs
        return AsyncChatResponse(
            content='{"finish": true}',
            usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        )


class OrderedLLM:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.first_started = asyncio.Event()
        self.release_first = asyncio.Event()

    async def chat_with_usage(self, **kwargs: Any) -> AsyncChatResponse:
        prompt = str(kwargs["user_prompt"])
        self.started.append(prompt)
        if prompt == "reader-first":
            self.first_started.set()
            await self.release_first.wait()
        return AsyncChatResponse(content="{}", usage={})


class RejectAdvancedLLM:
    supports_agent_inference_options = True

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def chat_with_usage(self, **kwargs: Any) -> AsyncChatResponse:
        self.calls.append(kwargs)
        if "response_format" in kwargs:
            raise LLMHTTPError(400, "unsupported response_format")
        return AsyncChatResponse(content="{}", usage={})


class NeverLLM:
    async def chat(self, *_: Any, **__: Any) -> str:
        raise AssertionError("terminal checkpoint must not trigger another LLM call")


class NeverTools:
    async def search_and_read(self, **_: Any) -> Any:
        raise AssertionError("terminal checkpoint must not repeat tool calls")


class FinishAndSearchLLM:
    async def chat(self, *_: Any, **__: Any) -> str:
        return json.dumps(
            {
                "base_local_version": 0,
                "assessment": {"coverage": "sufficient", "primary_gap": ""},
                "actions": [{"type": "SEARCH", "query": "one final query"}],
                "answer_summary": "premature summary",
                "finish": True,
                "stop_reason": "premature finish",
            }
        )


class OneClaimTools:
    def __init__(self) -> None:
        self.calls = 0

    async def search_and_read(self, **_: Any) -> ToolBatchResult:
        self.calls += 1
        source = SourceRecord(id="src", url="https://example.com", title="source", query="query")
        evidence = EvidenceRecord(
            id="ev",
            source_id="src",
            subtask_id="subtask",
            claim_text="claim",
            excerpt="exact excerpt",
        )
        claim = ClaimRecord(id="claim", text="claim", subtask_id="subtask", evidence_ids=["ev"])
        return ToolBatchResult(
            sources=[source],
            evidence=[evidence],
            claims=[claim],
            usage={"search_calls": 1, "reader_calls": 1, "tool_calls": 2},
        )


async def check_gateway() -> None:
    try:
        AsyncChatConfig(max_concurrent_requests=0)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid LLM concurrency must fail before creating a deadlocked semaphore")

    advanced = RecordingAdvancedLLM()
    gateway = AgentInferenceGateway(
        advanced,
        config=AgentInferenceConfig(
            max_concurrent_requests=2,
            control_concurrency=1,
            reader_concurrency=1,
            long_output_concurrency=1,
            max_concurrent_per_run=2,
            forward_priority=True,
        ),
    )
    response = await gateway.infer_with_usage(
        user_prompt="decide",
        system_prompt="system",
        temperature=0.0,
        top_p=1.0,
        max_tokens=100,
        role="researcher",
        run_id="run",
        subtask_id="subtask",
        request_id="run:subtask:react:0",
        response_schema=RESEARCHER_DECISION_SCHEMA,
        schema_name="researcher_decision",
    )
    assert advanced.kwargs["response_format"]["type"] == "json_schema"
    assert advanced.kwargs["priority"] == 0
    assert advanced.kwargs["request_id"] == "run:subtask:react:0"
    assert advanced.kwargs["chat_template_kwargs"] == {"enable_thinking": False}
    assert response.metadata["agent_role"] == "researcher"

    rejecting = RejectAdvancedLLM()
    compatible = AgentInferenceGateway(rejecting)
    for version in range(2):
        await compatible.infer_with_usage(
            user_prompt="decide",
            system_prompt="system",
            temperature=0.0,
            top_p=1.0,
            max_tokens=20,
            role="researcher",
            run_id="run",
            request_id=f"request-{version}",
            response_schema=RESEARCHER_DECISION_SCHEMA,
        )
    assert len(rejecting.calls) == 3
    assert "response_format" in rejecting.calls[0]
    assert all("response_format" not in call for call in rejecting.calls[1:])
    assert compatible.metrics.advanced_fallbacks == 1
    assert compatible._should_fallback(RuntimeError("request failed with HTTP 404"))
    assert compatible._should_fallback(RuntimeError("request failed with HTTP 405"))

    ordered = OrderedLLM()
    serialized = AgentInferenceGateway(
        ordered,
        config=AgentInferenceConfig(
            max_concurrent_requests=1,
            control_concurrency=1,
            reader_concurrency=1,
            long_output_concurrency=1,
            max_concurrent_per_run=1,
        ),
    )

    async def call(prompt: str, role: str) -> None:
        await serialized.infer_with_usage(
            user_prompt=prompt,
            system_prompt="",
            temperature=0.0,
            top_p=1.0,
            max_tokens=10,
            role=role,
            run_id="run",
        )

    first = asyncio.create_task(call("reader-first", "reader"))
    await ordered.first_started.wait()
    writer = asyncio.create_task(call("writer-last", "writer"))
    researcher = asyncio.create_task(call("researcher-next", "researcher"))
    await asyncio.sleep(0)
    ordered.release_first.set()
    await asyncio.gather(first, writer, researcher)
    assert ordered.started == ["reader-first", "researcher-next", "writer-last"]


def check_context_builder() -> None:
    observations = [
        {"source_id": f"old-{index}", "claim_summaries": ["x" * 3000]}
        for index in range(5)
    ]
    observations.append({"source_id": "newest", "claim_summaries": ["LATEST_MARKER " + "y" * 1200]})
    local = LocalResearchState(
        run_id="run",
        subtask_id="subtask",
        objective="Research one bounded question.",
        status=TaskStatus.RUNNING,
        recent_observations=observations,
    )
    result = ResearcherContextBuilder(
        token_counter=TokenCounter(),
        max_input_tokens=2500,
        recent_observation_limit=6,
    ).build(
        original_task={"prompt": "Research question", "language": "en"},
        brief=ResearchBrief(question="Research question", language="en"),
        subtask=SubTask(id="subtask", objective=local.objective),
        local=local,
        global_context={"existing_claims": [], "global_gaps": [], "global_conflicts": []},
        remaining_steps=2,
        remaining_tool_calls=4,
        max_queries=2,
    )
    assert result.estimated_tokens <= result.max_input_tokens
    assert result.dropped_observations > 0
    assert "LATEST_MARKER" in result.prompt
    assert '"evidence_ids": []' in result.prompt


def check_checkpoint() -> None:
    with TemporaryDirectory() as directory:
        store = RunStore(directory)
        local = LocalResearchState(
            run_id="run",
            subtask_id="subtask",
            objective="objective",
            status=TaskStatus.RUNNING,
            version=1,
            step=1,
            evidence_ids=["ev"],
        )
        source = SourceRecord(id="src", url="https://example.com", title="source", query="query")
        evidence = EvidenceRecord(
            id="ev",
            source_id="src",
            subtask_id="subtask",
            claim_text="claim",
            excerpt="excerpt",
        )
        claim = ClaimRecord(id="claim", text="claim", subtask_id="subtask", evidence_ids=["ev"])
        store.save_research_checkpoint(
            "run",
            local_state=local,
            sources=[source],
            evidence=[evidence],
            claims=[claim],
            events=[{"type": "researcher_step"}],
            usage={"researcher_calls": 1},
            last_decision={"finish": False},
        )
        loaded = store.load_research_checkpoint("run", "subtask")
        assert loaded is not None
        assert loaded["local_state"].version == 1
        assert loaded["evidence"][0].id == "ev"
        assert loaded["usage"]["researcher_calls"] == 1
        store.clear_research_checkpoint("run", "subtask")
        assert store.load_research_checkpoint("run", "subtask") is None


async def check_terminal_checkpoint_resume() -> None:
    with TemporaryDirectory() as directory:
        store = RunStore(directory)
        local = LocalResearchState(
            run_id="run",
            subtask_id="subtask",
            objective="objective",
            status=TaskStatus.RUNNING,
            version=1,
            step=1,
            source_ids=["src"],
            evidence_ids=["ev"],
            claim_ids=["claim"],
            stop_reason=REACT_STEP_TERMINAL,
        )
        source = SourceRecord(id="src", url="https://example.com", title="source", query="query")
        evidence = EvidenceRecord(
            id="ev",
            source_id="src",
            subtask_id="subtask",
            claim_text="claim",
            excerpt="excerpt",
        )
        claim = ClaimRecord(id="claim", text="claim", subtask_id="subtask", evidence_ids=["ev"])
        store.save_research_checkpoint(
            "run",
            local_state=local,
            sources=[source],
            evidence=[evidence],
            claims=[claim],
            events=[{"type": "researcher_step", "output_state_version": 1}],
            usage={"researcher_calls": 1, "reader_calls": 1},
            last_decision={"finish": True, "answer_summary": "checkpoint answer"},
        )
        subtask = SubTask(id="subtask", objective="objective", max_steps=3, max_tool_calls=5)
        state = GlobalResearchState(
            run_id="run",
            task={"prompt": "question", "language": "en"},
            brief=ResearchBrief(question="question", language="en"),
            tasks={"subtask": subtask},
        )
        researcher = ResearcherAgent(
            llm=NeverLLM(),
            tools=NeverTools(),  # type: ignore[arg-type]
            store=store,
            context_builder=ResearcherContextBuilder(
                token_counter=TokenCounter(),
                max_input_tokens=3000,
            ),
            max_queries_per_step=1,
            researcher_max_tokens=200,
            temperature=0.0,
        )
        bundle = await researcher.execute(state, subtask)
        assert bundle.result.status == TaskStatus.COMPLETED
        assert bundle.result.answer_summary == "checkpoint answer"
        assert bundle.result.claim_ids == ["claim"]
        assert store.load_research_checkpoint("run", "subtask") is None
        assert store.load_bundle("run", "subtask") is not None


async def check_search_then_finish_is_not_completed() -> None:
    with TemporaryDirectory() as directory:
        tools = OneClaimTools()
        researcher = ResearcherAgent(
            llm=FinishAndSearchLLM(),
            tools=tools,  # type: ignore[arg-type]
            store=RunStore(directory),
            context_builder=ResearcherContextBuilder(
                token_counter=TokenCounter(),
                max_input_tokens=3000,
            ),
            max_queries_per_step=1,
            researcher_max_tokens=200,
            temperature=0.0,
        )
        subtask = SubTask(id="subtask", objective="objective", max_steps=1, max_tool_calls=3)
        state = GlobalResearchState(
            run_id="run",
            task={"prompt": "question", "language": "en"},
            brief=ResearchBrief(question="question", language="en"),
            tasks={"subtask": subtask},
        )
        bundle = await researcher.execute(state, subtask, search_call_budget=1)
        assert tools.calls == 1
        assert bundle.result.status == TaskStatus.PARTIAL
        assert bundle.events[0]["finish_requested"] is True
        assert bundle.events[0]["finish_effective"] is False


async def main_async() -> None:
    await check_gateway()
    check_context_builder()
    check_checkpoint()
    await check_terminal_checkpoint_resume()
    await check_search_then_finish_is_not_completed()
    print("smoke_test_agent_inference passed")


if __name__ == "__main__":
    asyncio.run(main_async())
