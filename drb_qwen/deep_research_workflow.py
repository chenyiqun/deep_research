from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from .async_llm_client import AsyncChatClient
from .json_utils import extract_json
from .web_search import SearchResult, WebSearchClient


MAIN_AGENT_SYSTEM_PROMPT = (
    "You are the main agent in a multi-agent deep research system. "
    "You plan searchable web queries, maintain a compact global information state, "
    "and write final reports grounded in cited evidence."
)

READER_SYSTEM_PROMPT = (
    "You are a fast research reader. Extract source-specific core information "
    "faithfully from one web search result. Do not invent facts not supported by the source text."
)

SUMMARIZER_SYSTEM_PROMPT = (
    "You synthesize source-specific reader notes for one search query. "
    "Keep evidence attribution clear and preserve uncertainty."
)

STATE_UPDATER_SYSTEM_PROMPT = (
    "You update a global research information state. Output only genuinely new information, "
    "corrections to previous information, unresolved conflicts, and useful next-search hints."
)


@dataclass
class DeepResearchConfig:
    max_rounds: int = 3
    min_rounds: int = 1
    max_search_queries_per_round: int = 3
    search_top_k: int = 5
    search_count: int = 15
    max_concurrent_readers: int = 12
    planner_max_tokens: int = 2048
    reader_max_tokens: int = 2048
    summarizer_max_tokens: int = 3072
    state_updater_max_tokens: int = 4096
    report_max_tokens: int = 8192
    source_content_max_chars: int = 12000
    state_prompt_max_chars: int = 24000
    evidence_prompt_max_chars: int = 36000
    temperature_planner: float = 0.2
    temperature_reader: float = 0.0
    temperature_summarizer: float = 0.0
    temperature_report: float = 0.2


class AsyncDeepResearchWorkflow:
    def __init__(
        self,
        llm: AsyncChatClient,
        search_client: WebSearchClient,
        config: DeepResearchConfig | None = None,
    ) -> None:
        self.llm = llm
        self.search_client = search_client
        self.config = config or DeepResearchConfig()
        self._reader_semaphore = asyncio.Semaphore(self.config.max_concurrent_readers)

    async def run(self, task: dict[str, Any]) -> dict[str, Any]:
        state = initial_global_state(task)
        trace: list[dict[str, Any]] = []

        for round_idx in range(1, self.config.max_rounds + 1):
            plan = await self._plan_search_queries(task, state, round_idx)
            search_queries = sanitize_search_queries(
                plan.get("search_queries", []),
                max_queries=self.config.max_search_queries_per_round,
            )
            if not search_queries and round_idx == 1:
                search_queries = [str(task["prompt"])]
            elif not search_queries and round_idx > self.config.min_rounds:
                trace.append({"round": round_idx, "plan": plan, "skipped": "no search queries"})
                break

            search_results_by_query = await self._search_all(search_queries)
            reader_notes = await self._read_all_results(search_results_by_query, task)
            query_summaries = await self._summarize_by_query(task, search_results_by_query, reader_notes)
            state_patch = await self._update_state(task, state, query_summaries, round_idx)
            apply_state_patch(state, state_patch)
            state["rounds"].append(
                {
                    "round": round_idx,
                    "search_queries": search_queries,
                    "num_search_results": sum(len(v) for v in search_results_by_query.values()),
                    "num_reader_notes": len(reader_notes),
                }
            )
            trace.append(
                {
                    "round": round_idx,
                    "plan": plan,
                    "search_results": {
                        query: [result.to_dict() for result in results]
                        for query, results in search_results_by_query.items()
                    },
                    "reader_notes": reader_notes,
                    "query_summaries": query_summaries,
                    "state_patch": state_patch,
                }
            )

            if plan.get("should_continue") is False and round_idx >= self.config.min_rounds:
                break

        article = await self._write_final_report(task, state)
        return {"article": article, "state": state, "trace": trace}

    async def _plan_search_queries(
        self,
        task: dict[str, Any],
        state: dict[str, Any],
        round_idx: int,
    ) -> dict[str, Any]:
        prompt = build_planner_prompt(task, state, round_idx, self.config)
        response = await self.llm.chat(
            prompt,
            system_prompt=MAIN_AGENT_SYSTEM_PROMPT,
            temperature=self.config.temperature_planner,
            max_tokens=self.config.planner_max_tokens,
        )
        try:
            parsed = extract_json(response)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return {
            "should_continue": True,
            "reason": "fallback because planner did not return parseable JSON",
            "search_queries": [str(task["prompt"])],
        }

    async def _search_all(self, search_queries: list[str]) -> dict[str, list[SearchResult]]:
        async def run_one(query: str) -> tuple[str, list[SearchResult]]:
            results = await self.search_client.search(query, top_k=self.config.search_top_k)
            return query, results

        pairs = await asyncio.gather(*(run_one(query) for query in search_queries), return_exceptions=True)
        output: dict[str, list[SearchResult]] = {}
        for query, pair in zip(search_queries, pairs):
            if isinstance(pair, Exception):
                output[query] = []
            else:
                output[pair[0]] = pair[1]
        return output

    async def _read_all_results(
        self,
        search_results_by_query: dict[str, list[SearchResult]],
        task: dict[str, Any],
    ) -> list[dict[str, Any]]:
        deduped: list[SearchResult] = []
        seen: set[str] = set()
        for results in search_results_by_query.values():
            for result in results:
                key = result.link or f"{result.title}:{result.content[:120]}"
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(result)

        async def read_one(result: SearchResult) -> dict[str, Any]:
            async with self._reader_semaphore:
                return await self._read_result(task, result)

        notes = await asyncio.gather(*(read_one(result) for result in deduped), return_exceptions=True)
        output: list[dict[str, Any]] = []
        for result, note in zip(deduped, notes):
            if isinstance(note, Exception):
                output.append(
                    {
                        "source_url": result.link,
                        "source_title": result.title,
                        "search_query": result.search_query,
                        "error": str(note),
                    }
                )
            else:
                output.append(note)
        return output

    async def _read_result(self, task: dict[str, Any], result: SearchResult) -> dict[str, Any]:
        prompt = build_reader_prompt(task, result, self.config)
        response = await self.llm.chat(
            prompt,
            system_prompt=READER_SYSTEM_PROMPT,
            temperature=self.config.temperature_reader,
            max_tokens=self.config.reader_max_tokens,
        )
        try:
            parsed = extract_json(response)
            if isinstance(parsed, dict):
                parsed.setdefault("source_url", result.link)
                parsed.setdefault("source_title", result.title)
                parsed.setdefault("publish_date", result.publish_date)
                parsed.setdefault("search_query", result.search_query)
                return parsed
        except Exception:
            pass
        return {
            "source_url": result.link,
            "source_title": result.title,
            "publish_date": result.publish_date,
            "search_query": result.search_query,
            "raw_summary": response,
        }

    async def _summarize_by_query(
        self,
        task: dict[str, Any],
        search_results_by_query: dict[str, list[SearchResult]],
        reader_notes: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        notes_by_query: dict[str, list[dict[str, Any]]] = {query: [] for query in search_results_by_query}
        for note in reader_notes:
            query = str(note.get("search_query", ""))
            if query in notes_by_query:
                notes_by_query[query].append(note)

        async def summarize_one(query: str, notes: list[dict[str, Any]]) -> dict[str, Any]:
            prompt = build_query_summarizer_prompt(task, query, notes)
            response = await self.llm.chat(
                prompt,
                system_prompt=SUMMARIZER_SYSTEM_PROMPT,
                temperature=self.config.temperature_summarizer,
                max_tokens=self.config.summarizer_max_tokens,
            )
            try:
                parsed = extract_json(response)
                if isinstance(parsed, dict):
                    parsed.setdefault("search_query", query)
                    return parsed
            except Exception:
                pass
            return {"search_query": query, "raw_summary": response}

        return await asyncio.gather(
            *(summarize_one(query, notes) for query, notes in notes_by_query.items())
        )

    async def _update_state(
        self,
        task: dict[str, Any],
        state: dict[str, Any],
        query_summaries: list[dict[str, Any]],
        round_idx: int,
    ) -> dict[str, Any]:
        prompt = build_state_updater_prompt(task, state, query_summaries, round_idx, self.config)
        response = await self.llm.chat(
            prompt,
            system_prompt=STATE_UPDATER_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=self.config.state_updater_max_tokens,
        )
        try:
            parsed = extract_json(response)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return {"raw_state_update": response}

    async def _write_final_report(self, task: dict[str, Any], state: dict[str, Any]) -> str:
        prompt = build_final_report_prompt(task, state, self.config)
        return await self.llm.chat(
            prompt,
            system_prompt=MAIN_AGENT_SYSTEM_PROMPT,
            temperature=self.config.temperature_report,
            max_tokens=self.config.report_max_tokens,
        )


def initial_global_state(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "original_question": task["prompt"],
        "language": task.get("language", "en"),
        "topic": task.get("topic"),
        "rounds": [],
        "findings": [],
        "evidence": [],
        "open_questions": [],
        "conflicts": [],
        "corrections": [],
        "next_search_hints": [],
    }


def build_planner_prompt(
    task: dict[str, Any],
    state: dict[str, Any],
    round_idx: int,
    config: DeepResearchConfig,
) -> str:
    state_text = compact_json(state, config.state_prompt_max_chars)
    lang = task.get("language", "en")
    if lang == "zh":
        return f"""
你是 deep research 的 main agent。你必须基于当前 global information state 规划下一轮可直接用于网页搜索的 query。

要求：
1. 每个 search query 都必须是能直接放进搜索引擎的自然语言检索词。
2. 第 1 轮应覆盖原始问题的核心面；后续轮次应基于已有 state 找信息缺口、冲突和需要更新的数据。
3. 不要输出泛泛的内部思考，只输出 JSON。
4. 最多输出 {config.max_search_queries_per_round} 个 search query。
5. 如果已有信息已经足够，可以 should_continue=false 且 search_queries=[]。

<task>
{task["prompt"]}
</task>

<round_idx>
{round_idx}
</round_idx>

<global_information_state>
{state_text}
</global_information_state>

输出 JSON schema:
{{
  "should_continue": true,
  "reason": "为什么这一轮需要搜索，以及主要信息缺口是什么",
  "search_queries": [
    "可直接搜索的 query 1",
    "可直接搜索的 query 2"
  ]
}}
"""

    return f"""
You are the main agent in a deep research workflow. Plan the next round of directly searchable web queries based on the current global information state.

Rules:
1. Every search query must be a natural-language query that can be sent directly to a web search engine.
2. Round 1 should cover the main facets of the original question. Later rounds should target gaps, conflicts, and data freshness needs discovered in the state.
3. Return JSON only.
4. Return at most {config.max_search_queries_per_round} search queries.
5. If the state is already sufficient, set should_continue=false and search_queries=[].

<task>
{task["prompt"]}
</task>

<round_idx>
{round_idx}
</round_idx>

<global_information_state>
{state_text}
</global_information_state>

JSON schema:
{{
  "should_continue": true,
  "reason": "why another search round is needed and what gaps it targets",
  "search_queries": [
    "directly searchable query 1",
    "directly searchable query 2"
  ]
}}
"""


def build_reader_prompt(task: dict[str, Any], result: SearchResult, config: DeepResearchConfig) -> str:
    content = truncate(result.content, config.source_content_max_chars)
    lang = task.get("language", "en")
    if lang == "zh":
        return f"""
请阅读下面一个搜索结果，抽取对原始研究问题有用的核心信息。

<original_question>
{task["prompt"]}
</original_question>

<search_query>
{result.search_query}
</search_query>

<source>
title: {result.title}
url: {result.link}
media: {result.media}
publish_date: {result.publish_date}
content:
{content}
</source>

只输出 JSON:
{{
  "source_title": "{json_escape(result.title)}",
  "source_url": "{json_escape(result.link)}",
  "publish_date": "{json_escape(result.publish_date)}",
  "search_query": "{json_escape(result.search_query)}",
  "relevance": 0.0,
  "core_information": [
    {{
      "claim": "可用于回答问题的事实、数据或观点",
      "evidence": "来源中支持该 claim 的简要依据",
      "confidence": "high/medium/low"
    }}
  ],
  "useful_statistics": [],
  "limitations": [],
  "possible_followups": []
}}
"""

    return f"""
Read this single search result and extract core information useful for the original research question.

<original_question>
{task["prompt"]}
</original_question>

<search_query>
{result.search_query}
</search_query>

<source>
title: {result.title}
url: {result.link}
media: {result.media}
publish_date: {result.publish_date}
content:
{content}
</source>

Return JSON only:
{{
  "source_title": "{json_escape(result.title)}",
  "source_url": "{json_escape(result.link)}",
  "publish_date": "{json_escape(result.publish_date)}",
  "search_query": "{json_escape(result.search_query)}",
  "relevance": 0.0,
  "core_information": [
    {{
      "claim": "fact, statistic, or viewpoint useful for answering the question",
      "evidence": "brief support from this source",
      "confidence": "high/medium/low"
    }}
  ],
  "useful_statistics": [],
  "limitations": [],
  "possible_followups": []
}}
"""


def build_query_summarizer_prompt(
    task: dict[str, Any],
    search_query: str,
    reader_notes: list[dict[str, Any]],
) -> str:
    notes_text = compact_json(reader_notes, 30000)
    lang = task.get("language", "en")
    if lang == "zh":
        return f"""
请把同一个 search query 下多个 reader 的结果综合成一份 query-level core information。
保留 URL 归因，区分事实、推断、不确定性和冲突。

<original_question>
{task["prompt"]}
</original_question>

<search_query>
{search_query}
</search_query>

<reader_notes>
{notes_text}
</reader_notes>

只输出 JSON:
{{
  "search_query": "{json_escape(search_query)}",
  "key_findings": ["本 query 得到的关键发现"],
  "evidence_items": [
    {{
      "claim": "可进入 global state 的证据性信息",
      "source_url": "URL",
      "source_title": "title",
      "publish_date": "date",
      "confidence": "high/medium/low"
    }}
  ],
  "open_questions": ["仍未解决的问题"],
  "conflicts": ["来源之间或与已有常识之间的冲突"],
  "low_value_sources": ["低相关或低质量来源 URL"]
}}
"""

    return f"""
Synthesize multiple reader notes for one search query into query-level core information.
Preserve URL attribution and separate facts, inference, uncertainty, and conflicts.

<original_question>
{task["prompt"]}
</original_question>

<search_query>
{search_query}
</search_query>

<reader_notes>
{notes_text}
</reader_notes>

Return JSON only:
{{
  "search_query": "{json_escape(search_query)}",
  "key_findings": ["key finding from this query"],
  "evidence_items": [
    {{
      "claim": "evidence-backed information ready for the global state",
      "source_url": "URL",
      "source_title": "title",
      "publish_date": "date",
      "confidence": "high/medium/low"
    }}
  ],
  "open_questions": ["remaining unresolved question"],
  "conflicts": ["conflicts across sources or with the current understanding"],
  "low_value_sources": ["low relevance or low quality source URL"]
}}
"""


def build_state_updater_prompt(
    task: dict[str, Any],
    state: dict[str, Any],
    query_summaries: list[dict[str, Any]],
    round_idx: int,
    config: DeepResearchConfig,
) -> str:
    state_text = compact_json(state, config.state_prompt_max_chars)
    summaries_text = compact_json(query_summaries, config.evidence_prompt_max_chars)
    lang = task.get("language", "en")
    if lang == "zh":
        return f"""
你是 State Updater。根据上一版 global information state 和当前轮次 summarizer 输出，只输出增量 patch。

关键要求：
1. 之前 state 中已经有的信息不要重复输出。
2. 只输出新增信息、对旧信息的修正、被解决的问题、仍存在的冲突、下一轮搜索提示。
3. evidence 必须保留 source_url。
4. 如果当前轮次发现之前信息可能错误，放入 corrected_findings。

<original_question>
{task["prompt"]}
</original_question>

<round_idx>
{round_idx}
</round_idx>

<previous_global_information_state>
{state_text}
</previous_global_information_state>

<current_round_summaries>
{summaries_text}
</current_round_summaries>

只输出 JSON:
{{
  "new_findings": ["之前没有的关键发现"],
  "new_evidence": [
    {{
      "claim": "证据性信息",
      "source_url": "URL",
      "source_title": "title",
      "publish_date": "date",
      "confidence": "high/medium/low"
    }}
  ],
  "corrected_findings": [
    {{
      "old": "之前可能错误或不完整的信息",
      "new": "修正后的信息",
      "source_url": "URL"
    }}
  ],
  "new_open_questions": ["新增或仍未解决的信息缺口"],
  "resolved_open_questions": ["本轮已解决的问题"],
  "conflicts": ["仍未解决的证据冲突"],
  "next_search_hints": ["下一轮适合搜索的方向"]
}}
"""

    return f"""
You are the State Updater. Given the previous global information state and current-round summaries, output only an incremental patch.

Rules:
1. Do not repeat information already present in the previous state.
2. Output only new information, corrections, resolved questions, unresolved conflicts, and next-search hints.
3. Evidence must preserve source_url.
4. If this round shows previous information may be wrong, put it in corrected_findings.

<original_question>
{task["prompt"]}
</original_question>

<round_idx>
{round_idx}
</round_idx>

<previous_global_information_state>
{state_text}
</previous_global_information_state>

<current_round_summaries>
{summaries_text}
</current_round_summaries>

Return JSON only:
{{
  "new_findings": ["new key finding not already in the state"],
  "new_evidence": [
    {{
      "claim": "evidence-backed information",
      "source_url": "URL",
      "source_title": "title",
      "publish_date": "date",
      "confidence": "high/medium/low"
    }}
  ],
  "corrected_findings": [
    {{
      "old": "previously wrong or incomplete information",
      "new": "corrected information",
      "source_url": "URL"
    }}
  ],
  "new_open_questions": ["new or still unresolved information gap"],
  "resolved_open_questions": ["question resolved in this round"],
  "conflicts": ["unresolved evidence conflict"],
  "next_search_hints": ["good direction for the next search round"]
}}
"""


def build_final_report_prompt(
    task: dict[str, Any],
    state: dict[str, Any],
    config: DeepResearchConfig,
) -> str:
    state_text = compact_json(state, config.evidence_prompt_max_chars)
    lang = task.get("language", "en")
    if lang == "zh":
        return f"""
请基于 global information state，为原始 deep research 任务撰写最终报告。

要求：
1. 直接回答原始问题，先给结论摘要，再展开分析。
2. 覆盖所有显式子问题和 state 中的重要信息。
3. 对关键事实、数据、判断尽量标注来源 URL。
4. 明确说明不确定性、证据冲突和局限。
5. 使用清晰标题、表格或列表组织内容。
6. 不要编造 state 中没有依据的具体数据。

<original_question>
{task["prompt"]}
</original_question>

<global_information_state>
{state_text}
</global_information_state>

请输出最终报告正文，不要输出 JSON。
"""

    return f"""
Write the final deep research report based on the global information state.

Requirements:
1. Directly answer the original question. Start with an executive summary, then develop the analysis.
2. Cover all explicit sub-questions and important information in the state.
3. Cite source URLs near key facts, data, and judgments where possible.
4. State uncertainty, evidence conflicts, and limitations.
5. Use clear headings, tables, or bullet lists where useful.
6. Do not invent specific data not supported by the state.

<original_question>
{task["prompt"]}
</original_question>

<global_information_state>
{state_text}
</global_information_state>

Return the final report text, not JSON.
"""


def apply_state_patch(state: dict[str, Any], patch: dict[str, Any]) -> None:
    append_unique_strings(state, "findings", patch.get("new_findings", []))
    append_unique_dicts(state, "evidence", patch.get("new_evidence", []), key_fields=("claim", "source_url"))
    append_unique_dicts(state, "corrections", patch.get("corrected_findings", []), key_fields=("old", "new"))
    append_unique_strings(state, "open_questions", patch.get("new_open_questions", []))
    append_unique_strings(state, "conflicts", patch.get("conflicts", []))
    append_unique_strings(state, "next_search_hints", patch.get("next_search_hints", []))

    resolved = {normalize_text(item) for item in ensure_string_list(patch.get("resolved_open_questions", []))}
    if resolved:
        state["open_questions"] = [
            item for item in state.get("open_questions", []) if normalize_text(item) not in resolved
        ]
    if "raw_state_update" in patch:
        state.setdefault("raw_state_updates", []).append(patch["raw_state_update"])


def append_unique_strings(state: dict[str, Any], key: str, values: Any) -> None:
    items = ensure_string_list(values)
    existing = {normalize_text(item) for item in state.get(key, [])}
    for item in items:
        norm = normalize_text(item)
        if norm and norm not in existing:
            state.setdefault(key, []).append(item)
            existing.add(norm)


def append_unique_dicts(
    state: dict[str, Any],
    key: str,
    values: Any,
    key_fields: tuple[str, ...],
) -> None:
    if not isinstance(values, list):
        return
    existing = {
        tuple(normalize_text(str(item.get(field, ""))) for field in key_fields)
        for item in state.get(key, [])
        if isinstance(item, dict)
    }
    for item in values:
        if not isinstance(item, dict):
            continue
        dedupe_key = tuple(normalize_text(str(item.get(field, ""))) for field in key_fields)
        if any(dedupe_key) and dedupe_key not in existing:
            state.setdefault(key, []).append(item)
            existing.add(dedupe_key)


def sanitize_search_queries(values: Any, max_queries: int) -> list[str]:
    if not isinstance(values, list):
        return []
    queries: list[str] = []
    seen: set[str] = set()
    for value in values:
        query = str(value).strip()
        if not query:
            continue
        norm = normalize_text(query)
        if norm in seen:
            continue
        seen.add(norm)
        queries.append(query)
        if len(queries) >= max_queries:
            break
    return queries


def ensure_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def compact_json(value: Any, max_chars: int) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2)
    return truncate(text, max_chars)


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def json_escape(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace('"', '\\"')
