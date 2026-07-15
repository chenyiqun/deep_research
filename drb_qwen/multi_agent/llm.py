from __future__ import annotations

from typing import Any


async def call_chat(
    llm: Any,
    *,
    user_prompt: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    top_p: float = 0.95,
    role: str = "reader",
    run_id: str = "",
    subtask_id: str = "",
    request_id: str = "",
    response_schema: dict[str, Any] | None = None,
    schema_name: str = "agent_response",
    estimated_input_tokens: int = 0,
) -> tuple[str, dict[str, int]]:
    if hasattr(llm, "infer_with_usage"):
        response = await llm.infer_with_usage(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            role=role,
            run_id=run_id,
            subtask_id=subtask_id,
            request_id=request_id,
            response_schema=response_schema,
            schema_name=schema_name,
            estimated_input_tokens=estimated_input_tokens,
        )
        content = str(getattr(response, "content", ""))
        raw_usage = getattr(response, "usage", {})
        return content, normalize_token_usage(raw_usage)
    if hasattr(llm, "chat_with_usage"):
        response = await llm.chat_with_usage(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        content = str(getattr(response, "content", ""))
        raw_usage = getattr(response, "usage", {})
        return content, normalize_token_usage(raw_usage)
    content = await llm.chat(
        user_prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    return str(content), {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def normalize_token_usage(value: Any) -> dict[str, int]:
    value = value if isinstance(value, dict) else {}
    usage = {
        "prompt_tokens": safe_int(value.get("prompt_tokens")),
        "completion_tokens": safe_int(value.get("completion_tokens")),
        "total_tokens": safe_int(value.get("total_tokens")),
    }
    if not usage["total_tokens"]:
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    return usage


def add_token_usage(target: dict[str, int], value: dict[str, int]) -> None:
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        target[key] = int(target.get(key, 0)) + int(value.get(key, 0))


def safe_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0
