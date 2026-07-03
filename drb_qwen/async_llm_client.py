from __future__ import annotations

import asyncio
from dataclasses import dataclass
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


class AsyncChatClient:
    """Async OpenAI-compatible chat client for vLLM serve."""

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
    ) -> str:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return await self.chat_messages(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    async def chat_messages(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 4096,
    ) -> str:
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
                            raise RuntimeError(
                                f"vLLM chat request failed with HTTP {response.status}: {text[:1000]}"
                            )
                        data = await response.json()
                        content = data["choices"][0]["message"].get("content", "")
                        content = str(content).strip()
                        if self.config.strip_thinking:
                            content = strip_thinking_blocks(content).strip()
                        return content
                except Exception as exc:  # vLLM can temporarily reject bursts; retry a few times.
                    last_error = exc
                    if attempt >= self.config.max_retries:
                        break
                    await asyncio.sleep(self.config.retry_sleep_s * attempt)
            raise RuntimeError(f"vLLM chat request failed after retries: {last_error}") from last_error
