from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import time
from typing import Any

from ..async_llm_client import AsyncChatResponse, LLMHTTPError
from .context import ContextWindowExceeded, TokenCounter, fit_user_prompt_to_budget
from .protocols import json_response_format


ROLE_PRIORITY = {
    "main_replanner": 0,
    "researcher": 0,
    "main_planner": 1,
    "auditor": 1,
    "reader": 2,
    "writer": 3,
}


@dataclass
class AgentInferenceConfig:
    max_concurrent_requests: int = 16
    control_concurrency: int = 8
    reader_concurrency: int = 12
    long_output_concurrency: int = 2
    max_concurrent_per_run: int = 12
    max_inflight_tokens: int = 262_144
    structured_outputs: bool = True
    forward_priority: bool = False
    disable_thinking_for_json: bool = True
    advanced_option_fallback: bool = True
    max_model_len: int = 32768
    context_safety_tokens: int = 512
    context_retry_shrink_tokens: int = 1024
    context_retry_attempts: int = 1

    def __post_init__(self) -> None:
        self.max_concurrent_requests = max(1, int(self.max_concurrent_requests))
        self.control_concurrency = max(1, min(int(self.control_concurrency), self.max_concurrent_requests))
        self.reader_concurrency = max(1, min(int(self.reader_concurrency), self.max_concurrent_requests))
        self.long_output_concurrency = max(
            1, min(int(self.long_output_concurrency), self.max_concurrent_requests)
        )
        self.max_concurrent_per_run = max(
            1, min(int(self.max_concurrent_per_run), self.max_concurrent_requests)
        )
        self.max_inflight_tokens = max(1, int(self.max_inflight_tokens))
        self.max_model_len = max(1, int(self.max_model_len))
        self.context_safety_tokens = max(1, int(self.context_safety_tokens))
        self.context_retry_shrink_tokens = max(1, int(self.context_retry_shrink_tokens))
        self.context_retry_attempts = max(0, int(self.context_retry_attempts))
        if self.context_safety_tokens >= self.max_model_len:
            raise ValueError("context_safety_tokens must be smaller than max_model_len")


@dataclass
class _Waiter:
    sequence: int
    role: str
    priority: int
    run_id: str
    token_reservation: int


class _AdmissionController:
    def __init__(self, config: AgentInferenceConfig) -> None:
        self.config = config
        self._condition = asyncio.Condition()
        self._waiters: list[_Waiter] = []
        self._sequence = 0
        self._active_total = 0
        self._active_by_group: dict[str, int] = {}
        self._active_by_run: dict[str, int] = {}
        self._active_tokens = 0

    async def acquire(self, role: str, run_id: str, token_reservation: int) -> _Waiter:
        async with self._condition:
            self._sequence += 1
            waiter = _Waiter(
                sequence=self._sequence,
                role=role,
                priority=ROLE_PRIORITY.get(role, 2),
                run_id=run_id or "__unscoped__",
                token_reservation=max(1, min(int(token_reservation), self.config.max_inflight_tokens)),
            )
            self._waiters.append(waiter)
            try:
                while True:
                    candidate = self._first_eligible()
                    if candidate is waiter:
                        self._waiters.remove(waiter)
                        group = role_group(waiter.role)
                        self._active_total += 1
                        self._active_by_group[group] = self._active_by_group.get(group, 0) + 1
                        self._active_by_run[waiter.run_id] = self._active_by_run.get(waiter.run_id, 0) + 1
                        self._active_tokens += waiter.token_reservation
                        self._condition.notify_all()
                        return waiter
                    await self._condition.wait()
            except BaseException:
                if waiter in self._waiters:
                    self._waiters.remove(waiter)
                    self._condition.notify_all()
                raise

    async def release(self, waiter: _Waiter) -> None:
        async with self._condition:
            group = role_group(waiter.role)
            self._active_total = max(0, self._active_total - 1)
            self._active_by_group[group] = max(0, self._active_by_group.get(group, 0) - 1)
            self._active_by_run[waiter.run_id] = max(0, self._active_by_run.get(waiter.run_id, 0) - 1)
            self._active_tokens = max(0, self._active_tokens - waiter.token_reservation)
            self._condition.notify_all()

    def _first_eligible(self) -> _Waiter | None:
        for waiter in sorted(self._waiters, key=lambda item: (item.priority, item.sequence)):
            if self._eligible(waiter):
                return waiter
        return None

    def _eligible(self, waiter: _Waiter) -> bool:
        if self._active_total >= self.config.max_concurrent_requests:
            return False
        group = role_group(waiter.role)
        if self._active_by_group.get(group, 0) >= group_limit(self.config, group):
            return False
        if self._active_by_run.get(waiter.run_id, 0) >= self.config.max_concurrent_per_run:
            return False
        token_capacity_ok = self._active_tokens + waiter.token_reservation <= self.config.max_inflight_tokens
        return token_capacity_ok or self._active_total == 0


@dataclass
class GatewayMetrics:
    requests: int = 0
    advanced_fallbacks: int = 0
    context_truncations: int = 0
    context_retries: int = 0
    context_retry_failures: int = 0
    queue_wait_ms: float = 0.0
    requests_by_role: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "advanced_fallbacks": self.advanced_fallbacks,
            "context_truncations": self.context_truncations,
            "context_retries": self.context_retries,
            "context_retry_failures": self.context_retry_failures,
            "queue_wait_ms": round(self.queue_wait_ms, 3),
            "requests_by_role": dict(self.requests_by_role),
        }


class AgentInferenceGateway:
    """Role-aware inference boundary for stateless Agent turns.

    Durable Agent state remains outside this class. A call represents exactly one
    input/output turn and releases its admission/KV pressure before tools run.
    """

    def __init__(
        self,
        llm: Any,
        *,
        config: AgentInferenceConfig | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.llm = llm
        self.config = config or AgentInferenceConfig()
        self.token_counter = token_counter or TokenCounter()
        self.admission = _AdmissionController(self.config)
        self.metrics = GatewayMetrics()
        self._advanced_options_disabled = False

    async def infer_with_usage(
        self,
        *,
        user_prompt: str,
        system_prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        role: str,
        run_id: str = "",
        subtask_id: str = "",
        request_id: str = "",
        response_schema: dict[str, Any] | None = None,
        schema_name: str = "agent_response",
        estimated_input_tokens: int = 0,
    ) -> AsyncChatResponse:
        output_tokens = max(1, int(max_tokens))
        max_input_tokens = (
            self.config.max_model_len
            - output_tokens
            - self.config.context_safety_tokens
        )
        if max_input_tokens <= 0:
            raise ContextWindowExceeded(
                f"Requested {output_tokens} output tokens plus "
                f"{self.config.context_safety_tokens} safety tokens for a "
                f"{self.config.max_model_len}-token model window"
            )

        fitted = fit_user_prompt_to_budget(
            token_counter=self.token_counter,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_input_tokens=max_input_tokens,
            original_tokens=estimated_input_tokens,
        )
        current_prompt = fitted.prompt
        input_tokens = fitted.estimated_tokens
        if fitted.truncated:
            self.metrics.context_truncations += 1
        reservation = input_tokens + max(1, int(max_tokens))
        queued_at = time.monotonic()
        permit = await self.admission.acquire(role, run_id, reservation)
        queue_wait_ms = (time.monotonic() - queued_at) * 1000.0
        try:
            response = None
            retry_budget = max_input_tokens
            for context_attempt in range(self.config.context_retry_attempts + 1):
                try:
                    response = await self._call_base(
                        user_prompt=current_prompt,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        role=role,
                        request_id=request_id,
                        response_schema=response_schema,
                        schema_name=schema_name,
                    )
                    break
                except Exception as exc:
                    if (
                        context_attempt >= self.config.context_retry_attempts
                        or not is_context_length_error(exc)
                    ):
                        if is_context_length_error(exc) and context_attempt > 0:
                            self.metrics.context_retry_failures += 1
                        raise
                    retry_budget = max(
                        1,
                        retry_budget - self.config.context_retry_shrink_tokens,
                    )
                    refitted = fit_user_prompt_to_budget(
                        token_counter=self.token_counter,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_input_tokens=retry_budget,
                    )
                    if refitted.prompt == current_prompt:
                        self.metrics.context_retry_failures += 1
                        raise
                    current_prompt = refitted.prompt
                    input_tokens = refitted.estimated_tokens
                    self.metrics.context_truncations += 1
                    self.metrics.context_retries += 1
            if response is None:
                raise RuntimeError("Agent inference ended without a response")
        finally:
            await self.admission.release(permit)

        self.metrics.requests += 1
        self.metrics.queue_wait_ms += queue_wait_ms
        self.metrics.requests_by_role[role] = self.metrics.requests_by_role.get(role, 0) + 1
        metadata = dict(getattr(response, "metadata", {}) or {})
        metadata.update(
            {
                "agent_role": role,
                "run_id": run_id,
                "subtask_id": subtask_id,
                "queue_wait_ms": round(queue_wait_ms, 3),
                "estimated_input_tokens": input_tokens,
                "original_estimated_input_tokens": fitted.original_tokens,
                "max_input_tokens": max_input_tokens,
                "input_truncated": current_prompt != user_prompt,
                "dropped_input_chars": max(0, len(user_prompt) - len(current_prompt)),
                "token_reservation": reservation,
            }
        )
        return AsyncChatResponse(
            content=str(getattr(response, "content", "")),
            usage=dict(getattr(response, "usage", {}) or {}),
            metadata=metadata,
        )

    async def _call_base(
        self,
        *,
        user_prompt: str,
        system_prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        role: str,
        request_id: str,
        response_schema: dict[str, Any] | None,
        schema_name: str,
    ) -> AsyncChatResponse:
        basic_kwargs = {
            "user_prompt": user_prompt,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        supports_options = bool(getattr(self.llm, "supports_agent_inference_options", False))
        use_advanced = supports_options and not self._advanced_options_disabled
        if use_advanced:
            advanced_kwargs = dict(basic_kwargs)
            if self.config.structured_outputs and response_schema:
                advanced_kwargs["response_format"] = json_response_format(schema_name, response_schema)
            if self.config.forward_priority:
                advanced_kwargs["priority"] = ROLE_PRIORITY.get(role, 2)
            if request_id:
                advanced_kwargs["request_id"] = request_id
            if self.config.disable_thinking_for_json and response_schema:
                advanced_kwargs["chat_template_kwargs"] = {"enable_thinking": False}
            try:
                return await self.llm.chat_with_usage(**advanced_kwargs)
            except Exception as exc:
                if not self._should_fallback(exc):
                    raise
                self._advanced_options_disabled = True
                self.metrics.advanced_fallbacks += 1

        if hasattr(self.llm, "chat_with_usage"):
            return await self.llm.chat_with_usage(**basic_kwargs)
        content = await self.llm.chat(**basic_kwargs)
        return AsyncChatResponse(content=str(content), usage={})

    def _should_fallback(self, exc: Exception) -> bool:
        if not self.config.advanced_option_fallback:
            return False
        if isinstance(exc, LLMHTTPError):
            if exc.status in {404, 405}:
                return True
            if exc.status not in {400, 422} or is_context_length_error(exc):
                return False
            return has_advanced_option_error(exc.body)
        text = str(exc).lower()
        if is_context_length_error(exc):
            return False
        if any(f"http {status}" in text for status in (404, 405)):
            return True
        return (
            any(f"http {status}" in text for status in (400, 422))
            and has_advanced_option_error(text)
        ) or "unexpected keyword" in text


def is_context_length_error(exc: BaseException | str) -> bool:
    if isinstance(exc, LLMHTTPError):
        text = exc.body.lower()
    else:
        text = str(exc).lower()
    markers = (
        "maximum context length",
        "context length",
        "input_tokens",
        "prompt is too long",
        "too many tokens",
        "max_model_len",
    )
    return any(marker in text for marker in markers)


def has_advanced_option_error(value: str) -> bool:
    text = str(value).lower()
    option_markers = (
        "response_format",
        "json_schema",
        "structured output",
        "structured_outputs",
        "priority",
        "request_id",
        "chat_template_kwargs",
        "unknown field",
        "unknown parameter",
        "unsupported parameter",
        "unexpected keyword",
        "extra_forbidden",
    )
    return any(marker in text for marker in option_markers)


def role_group(role: str) -> str:
    if role in {"main_planner", "main_replanner", "researcher", "auditor"}:
        return "control"
    if role == "writer":
        return "long"
    return "reader"


def group_limit(config: AgentInferenceConfig, group: str) -> int:
    if group == "control":
        return config.control_concurrency
    if group == "long":
        return config.long_output_concurrency
    return config.reader_concurrency
