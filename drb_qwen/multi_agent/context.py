from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any

from .prompts import RESEARCHER_SYSTEM_PROMPT, build_researcher_step_prompt
from .schemas import LocalResearchState, ResearchBrief, SubTask


class ContextWindowExceeded(RuntimeError):
    pass


class TokenCounter:
    """Counts with a local model tokenizer when configured, otherwise conservatively estimates.

    The fallback deliberately uses UTF-8 bytes so Chinese text is budgeted more
    conservatively than an English-only chars/4 heuristic.
    """

    def __init__(self, tokenizer_path: str = "") -> None:
        self.tokenizer_path = str(tokenizer_path or "").strip()
        self.tokenizer: Any | None = None
        self.load_error = ""
        if self.tokenizer_path:
            try:
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizer_path,
                    trust_remote_code=True,
                    local_files_only=True,
                )
            except Exception as exc:
                self.load_error = str(exc)

    @property
    def exact(self) -> bool:
        return self.tokenizer is not None

    def count_text(self, text: str) -> int:
        if self.tokenizer is not None:
            try:
                encoded = self.tokenizer.encode(str(text), add_special_tokens=False)
                return max(1, len(encoded))
            except Exception:
                pass
        value = str(text)
        return max(
            1,
            math.ceil(len(value.encode("utf-8")) / 3),
            math.ceil(len(value) / 2),
        )

    def count_messages(self, messages: list[dict[str, str]]) -> int:
        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            kwargs: dict[str, Any] = {
                "tokenize": True,
                "add_generation_prompt": True,
                "enable_thinking": False,
            }
            try:
                encoded = self.tokenizer.apply_chat_template(messages, **kwargs)
                return max(1, len(encoded))
            except TypeError:
                kwargs.pop("enable_thinking", None)
                try:
                    encoded = self.tokenizer.apply_chat_template(messages, **kwargs)
                    return max(1, len(encoded))
                except Exception:
                    pass
            except Exception:
                pass
        return sum(self.count_text(message.get("content", "")) + 8 for message in messages) + 8


@dataclass
class PromptBuildResult:
    prompt: str
    estimated_tokens: int
    max_input_tokens: int
    dropped_observations: int = 0
    dropped_global_claims: int = 0
    token_count_exact: bool = False


@dataclass
class PromptFitResult:
    prompt: str
    estimated_tokens: int
    original_tokens: int
    max_input_tokens: int
    truncated: bool = False
    dropped_chars: int = 0
    token_count_exact: bool = False


def fit_user_prompt_to_budget(
    *,
    token_counter: TokenCounter,
    system_prompt: str,
    user_prompt: str,
    max_input_tokens: int,
    original_tokens: int = 0,
) -> PromptFitResult:
    """Fit one stateless Agent turn into its input-token budget.

    Role-specific prompt builders should compact semantic data first. This is the
    final safety boundary: it preserves the prompt instructions at both ends and
    removes material from the middle until the rendered chat template fits.
    """

    budget = max(1, int(max_input_tokens))

    def count(prompt: str) -> int:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return token_counter.count_messages(messages)

    prompt = str(user_prompt)
    measured_original = max(0, int(original_tokens)) or count(prompt)
    if measured_original <= budget:
        return PromptFitResult(
            prompt=prompt,
            estimated_tokens=measured_original,
            original_tokens=measured_original,
            max_input_tokens=budget,
            token_count_exact=token_counter.exact,
        )

    marker = "\n\n...[middle context omitted to fit the model window]...\n\n"
    marker_tokens = count(marker)
    if marker_tokens > budget:
        raise ContextWindowExceeded(
            f"System prompt and chat template need about {marker_tokens} tokens, "
            f"exceeding the {budget}-token input budget"
        )

    # Keep most of the beginning (task, protocol, requirements) and the end
    # (closing tags and output contract). Token counts are checked on the fully
    # rendered messages, so this also covers chat-template overhead.
    low = 0
    high = len(prompt)
    best_prompt = marker
    best_tokens = marker_tokens
    while low <= high:
        kept_chars = (low + high) // 2
        head_chars = int(kept_chars * 0.8)
        tail_chars = kept_chars - head_chars
        tail = prompt[-tail_chars:] if tail_chars else ""
        candidate = prompt[:head_chars] + marker + tail
        candidate_tokens = count(candidate)
        if candidate_tokens <= budget:
            best_prompt = candidate
            best_tokens = candidate_tokens
            low = kept_chars + 1
        else:
            high = kept_chars - 1

    return PromptFitResult(
        prompt=best_prompt,
        estimated_tokens=best_tokens,
        original_tokens=measured_original,
        max_input_tokens=budget,
        truncated=True,
        dropped_chars=max(0, len(prompt) - (len(best_prompt) - len(marker))),
        token_count_exact=token_counter.exact,
    )


class ResearcherContextBuilder:
    """Builds the next stateless Researcher input from durable semantic state."""

    def __init__(
        self,
        *,
        token_counter: TokenCounter,
        max_input_tokens: int,
        recent_observation_limit: int = 8,
    ) -> None:
        self.token_counter = token_counter
        self.max_input_tokens = max(512, int(max_input_tokens))
        self.recent_observation_limit = max(1, int(recent_observation_limit))

    def build(
        self,
        *,
        original_task: dict[str, Any],
        brief: ResearchBrief,
        subtask: SubTask,
        local: LocalResearchState,
        global_context: dict[str, Any],
        remaining_steps: int,
        remaining_tool_calls: int,
        max_queries: int,
    ) -> PromptBuildResult:
        local_view = local_research_view(local, self.recent_observation_limit)
        global_view = deepcopy(global_context)
        initial_observations = len(local_view["recent_observations"])
        initial_global_claims = len(global_view.get("existing_claims", []))

        def render() -> tuple[str, int]:
            prompt = build_researcher_step_prompt(
                original_task,
                brief,
                subtask,
                local_view,
                global_view,
                remaining_steps=remaining_steps,
                remaining_tool_calls=remaining_tool_calls,
                max_queries=max_queries,
            )
            tokens = self.token_counter.count_messages(
                [
                    {"role": "system", "content": RESEARCHER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
            )
            return prompt, tokens

        prompt, tokens = render()
        while tokens > self.max_input_tokens and local_view["recent_observations"]:
            local_view["recent_observations"].pop(0)
            prompt, tokens = render()

        claims = global_view.get("existing_claims", [])
        while tokens > self.max_input_tokens and isinstance(claims, list) and claims:
            del claims[: max(1, len(claims) // 2)]
            prompt, tokens = render()

        for key in ("global_conflicts", "global_gaps"):
            values = global_view.get(key, [])
            while tokens > self.max_input_tokens and isinstance(values, list) and values:
                del values[: max(1, len(values) // 2)]
                prompt, tokens = render()

        for key, minimum in (("source_ids", 16), ("evidence_ids", 24), ("claim_ids", 24)):
            values = local_view.get(key, [])
            while tokens > self.max_input_tokens and isinstance(values, list) and len(values) > minimum:
                del values[: max(1, len(values) // 2)]
                prompt, tokens = render()

        if tokens > self.max_input_tokens:
            raise ContextWindowExceeded(
                f"Researcher input needs about {tokens} tokens, exceeding its {self.max_input_tokens}-token budget"
            )

        return PromptBuildResult(
            prompt=prompt,
            estimated_tokens=tokens,
            max_input_tokens=self.max_input_tokens,
            dropped_observations=initial_observations - len(local_view["recent_observations"]),
            dropped_global_claims=initial_global_claims - len(global_view.get("existing_claims", [])),
            token_count_exact=self.token_counter.exact,
        )


def local_research_view(local: LocalResearchState, recent_observation_limit: int) -> dict[str, Any]:
    """Stable, bounded projection used for inference; it is not the persisted state itself."""

    return {
        "run_id": local.run_id,
        "subtask_id": local.subtask_id,
        "objective": local.objective,
        "status": local.status.value,
        "version": local.version,
        "step": local.step,
        "queries": local.queries[-64:],
        "query_ledger": local.queries[-64:],
        "source_ids": local.source_ids[-96:],
        "evidence_ids": local.evidence_ids[-160:],
        "claim_ids": local.claim_ids[-160:],
        "gaps": local.gaps[-32:],
        "conflicts": local.conflicts[-24:],
        "recent_observations": local.recent_observations[-recent_observation_limit:],
        "tool_calls": local.tool_calls,
        "search_calls": local.search_calls,
        "answer_summary": local.answer_summary[:6000],
        "stop_reason": local.stop_reason,
    }
