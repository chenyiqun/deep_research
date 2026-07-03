from __future__ import annotations

import asyncio
import json

from drb_qwen.deep_research_workflow import AsyncDeepResearchWorkflow, DeepResearchConfig
from drb_qwen.url_fetcher import URLFetchResult
from drb_qwen.web_search import SearchResult


class FakeLLM:
    async def chat(
        self,
        user_prompt: str,
        system_prompt: str = "",
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 4096,
    ) -> str:
        if "规划下一轮" in user_prompt or "Plan the next round" in user_prompt:
            return json.dumps(
                {
                    "should_continue": False,
                    "reason": "one smoke-test search is enough",
                    "search_queries": ["AI interaction interpersonal relations evidence"],
                }
            )
        if "抽取对原始研究问题有用" in user_prompt or "extract core information" in user_prompt:
            assert "FULL ARTICLE TEXT" in user_prompt
            return json.dumps(
                {
                    "relevance": 0.9,
                    "core_information": [
                        {
                            "claim": "AI companions can change perceived companionship needs.",
                            "evidence": "The source discusses AI companions and social needs.",
                            "confidence": "medium",
                        }
                    ],
                    "useful_statistics": [],
                    "limitations": ["Synthetic smoke source."],
                    "possible_followups": [],
                }
            )
        if "query-level" in user_prompt or "同一个 search query" in user_prompt:
            return json.dumps(
                {
                    "key_findings": ["AI interaction may reshape why people seek companionship."],
                    "evidence_items": [
                        {
                            "claim": "AI companions can affect companionship motivations.",
                            "source_url": "https://example.com/ai-relations",
                            "source_title": "AI and relations",
                            "publish_date": "2026-01-01",
                            "confidence": "medium",
                        }
                    ],
                    "open_questions": [],
                    "conflicts": [],
                    "low_value_sources": [],
                }
            )
        if "State Updater" in user_prompt:
            return json.dumps(
                {
                    "new_findings": ["AI interaction may reshape companionship motivations."],
                    "new_evidence": [
                        {
                            "claim": "AI companions can affect companionship motivations.",
                            "source_url": "https://example.com/ai-relations",
                            "source_title": "AI and relations",
                            "publish_date": "2026-01-01",
                            "confidence": "medium",
                        }
                    ],
                    "corrected_findings": [],
                    "new_open_questions": [],
                    "resolved_open_questions": [],
                    "conflicts": [],
                    "next_search_hints": [],
                }
            )
        return "Executive summary\n\nAI interaction may reshape interpersonal relations, with cited evidence from https://example.com/ai-relations."


class FakeSearchClient:
    async def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        return [
            SearchResult(
                title="AI and relations",
                content="AI companions and social algorithms can change companionship needs.",
                link="https://example.com/ai-relations",
                media="example",
                publish_date="2026-01-01",
                search_query=query,
            )
        ][:top_k]


class FakeContentFetcher:
    async def fetch(self, url: str) -> URLFetchResult:
        return URLFetchResult(
            url=url,
            ok=True,
            status=200,
            content_type="text/html",
            final_url=url,
            text="FULL ARTICLE TEXT. AI companions can change perceived companionship needs.",
        )


async def main_async() -> None:
    workflow = AsyncDeepResearchWorkflow(
        llm=FakeLLM(),  # type: ignore[arg-type]
        search_client=FakeSearchClient(),  # type: ignore[arg-type]
        content_fetcher=FakeContentFetcher(),  # type: ignore[arg-type]
        config=DeepResearchConfig(
            max_rounds=1,
            max_search_queries_per_round=1,
            search_top_k=1,
            min_fetched_content_chars=10,
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
    assert result["state"]["findings"]
    assert result["state"]["evidence"]
    assert result["trace"][0]["search_results"]
    assert result["trace"][0]["source_fetches"][0]["used_full_content"] is True
    print("smoke_test_async_workflow passed")


if __name__ == "__main__":
    asyncio.run(main_async())
