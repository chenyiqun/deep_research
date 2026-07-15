from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from .vllm_chat import strip_thinking_blocks


@dataclass
class AsyncChatConfig:
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "qwen3-32b"
    api_key: str = "EMPTY"
    timeout_s: int = 600
    max_concurrent_requests: int = 16
    max_retries: int = 3
    retry_sleep_s: float = 2.0
    strip_thinking: bool = True

    def __post_init__(self) -> None:
        if self.timeout_s <= 0:
            raise ValueError("AsyncChatConfig.timeout_s must be positive")
        if self.max_concurrent_requests <= 0:
            raise ValueError("AsyncChatConfig.max_concurrent_requests must be positive")
        if self.max_retries <= 0:
            raise ValueError("AsyncChatConfig.max_retries must be positive")
        if self.retry_sleep_s < 0:
            raise ValueError("AsyncChatConfig.retry_sleep_s cannot be negative")


@dataclass
class AsyncChatResponse:
    content: str
    usage: dict[str, int]
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMHTTPError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        self.status = int(status)
        self.body = str(body)
        super().__init__(f"vLLM chat request failed with HTTP {self.status}: {self.body[:1000]}")


class AsyncChatClient:
    """Async OpenAI-compatible chat client for vLLM serve."""

    supports_agent_inference_options = True

    def __init__(self, config: AsyncChatConfig) -> None:
        self.config = config
        self._session: Any | None = None
        self._semaphore = asyncio.Semaphore(config.max_concurrent_requests)

    async def __aenter__(self) -> "AsyncChatClient":
        try:
            import aiohttp
        except ImportError as exc:
            raise RuntimeError("Install aiohttp to use async vLLM server calls.") from exc

        timeout = aiohttp.ClientTimeout(total=self.config.timeout_s)
        self._session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._session is not None:
            await self._session.close()
        self._session = None

    async def chat(
        self,
        user_prompt: str,
        system_prompt: str = "",
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 4096,
        response_format: dict[str, Any] | None = None,
        priority: int | None = None,
        request_id: str = "",
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> str:
        response = await self.chat_with_usage(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            response_format=response_format,
            priority=priority,
            request_id=request_id,
            chat_template_kwargs=chat_template_kwargs,
        )
        return response.content

    async def chat_with_usage(
        self,
        user_prompt: str,
        system_prompt: str = "",
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 4096,
        response_format: dict[str, Any] | None = None,
        priority: int | None = None,
        request_id: str = "",
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> AsyncChatResponse:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return await self.chat_messages_with_usage(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            response_format=response_format,
            priority=priority,
            request_id=request_id,
            chat_template_kwargs=chat_template_kwargs,
        )

    async def chat_messages(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 4096,
        response_format: dict[str, Any] | None = None,
        priority: int | None = None,
        request_id: str = "",
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> str:
        response = await self.chat_messages_with_usage(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            response_format=response_format,
            priority=priority,
            request_id=request_id,
            chat_template_kwargs=chat_template_kwargs,
        )
        return response.content

    async def chat_messages_with_usage(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 4096,
        response_format: dict[str, Any] | None = None,
        priority: int | None = None,
        request_id: str = "",
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> AsyncChatResponse:
        if self._session is None:
            raise RuntimeError("AsyncChatClient must be used as an async context manager.")

        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format
        if priority is not None:
            payload["priority"] = int(priority)
        if request_id:
            payload["request_id"] = request_id
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = chat_template_kwargs
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        async with self._semaphore:
            last_error: Exception | None = None
            for attempt in range(1, self.config.max_retries + 1):
                try:
                    async with self._session.post(url, json=payload, headers=headers) as response:
                        text = await response.text()
                        if response.status >= 400:
                            raise LLMHTTPError(response.status, text)
                        data = await response.json()
                        content = data["choices"][0]["message"].get("content", "")
                        content = str(content).strip()
                        if self.config.strip_thinking:
                            content = strip_thinking_blocks(content).strip()
                        raw_usage = data.get("usage", {})
                        usage = {
                            "prompt_tokens": safe_usage_int(raw_usage.get("prompt_tokens")),
                            "completion_tokens": safe_usage_int(raw_usage.get("completion_tokens")),
                            "total_tokens": safe_usage_int(raw_usage.get("total_tokens")),
                        }
                        if not usage["total_tokens"]:
                            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
                        return AsyncChatResponse(
                            content=content,
                            usage=usage,
                            metadata={
                                "request_id": str(data.get("id") or request_id),
                                "finish_reason": str(data.get("choices", [{}])[0].get("finish_reason", "")),
                            },
                        )
                except Exception as exc:  # vLLM can temporarily reject bursts; retry a few times.
                    last_error = exc
                    if (
                        isinstance(exc, LLMHTTPError)
                        and 400 <= exc.status < 500
                        and exc.status not in {408, 409, 425, 429}
                    ):
                        break
                    if attempt >= self.config.max_retries:
                        break
                    await asyncio.sleep(self.config.retry_sleep_s * attempt)
            if isinstance(last_error, LLMHTTPError):
                raise last_error
            raise RuntimeError(f"vLLM chat request failed after retries: {last_error}") from last_error


def safe_usage_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0
