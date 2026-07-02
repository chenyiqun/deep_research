from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class GenerationConfig:
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 8192
    stop: list[str] | None = None


class VLLMChatModel:
    """Small chat wrapper around vLLM.

    Imports vLLM lazily so utility modules and tests can run without GPUs.
    """

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        trust_remote_code: bool = True,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int | None = None,
        enforce_eager: bool = False,
    ) -> None:
        try:
            from vllm import LLM
        except ImportError as exc:
            raise RuntimeError(
                "vLLM is not installed. Install requirements.txt on a GPU machine first."
            ) from exc

        kwargs = {
            "model": model_name,
            "tensor_parallel_size": tensor_parallel_size,
            "dtype": dtype,
            "trust_remote_code": trust_remote_code,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enforce_eager": enforce_eager,
        }
        if max_model_len is not None:
            kwargs["max_model_len"] = max_model_len

        self.model_name = model_name
        self.llm = LLM(**kwargs)
        self.tokenizer = self.llm.get_tokenizer()

    def _render_chat(self, user_prompt: str, system_prompt: str = "") -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        if system_prompt:
            return f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n"
        return f"<|user|>\n{user_prompt}\n<|assistant|>\n"

    def generate_batch(
        self,
        user_prompts: Iterable[str],
        system_prompt: str = "",
        config: GenerationConfig | None = None,
    ) -> list[str]:
        try:
            from vllm import SamplingParams
        except ImportError as exc:
            raise RuntimeError("vLLM is not installed.") from exc

        config = config or GenerationConfig()
        sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            stop=config.stop,
        )
        rendered = [self._render_chat(prompt, system_prompt) for prompt in user_prompts]
        outputs = self.llm.generate(rendered, sampling_params)

        texts: list[str] = []
        for output in outputs:
            if not output.outputs:
                texts.append("")
            else:
                texts.append(output.outputs[0].text.strip())
        return texts

