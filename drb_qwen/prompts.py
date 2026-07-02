from __future__ import annotations

import json
from typing import Any


DIMENSIONS = ["comprehensiveness", "insight", "instruction_following", "readability"]


REPORT_SYSTEM_PROMPT = (
    "You are a careful deep research analyst. Produce rigorous, well-structured, "
    "source-aware research reports that directly answer the user task."
)


def build_report_prompt(task: dict[str, Any], evidence: str | None = None) -> str:
    language = task.get("language", "en")
    prompt = task["prompt"]
    if language == "zh":
        instruction = (
            "请针对下面的深度研究任务撰写一份完整研究报告。要求：\n"
            "1. 先给出结论摘要，再展开分析。\n"
            "2. 覆盖任务中的所有子问题，不要泛泛而谈。\n"
            "3. 明确说明关键依据、假设、不确定性和局限。\n"
            "4. 使用清晰的标题、表格或列表组织信息。\n"
            "5. 如果你使用了外部证据材料，请在相关句子后标注来源 URL。\n"
        )
    else:
        instruction = (
            "Write a complete deep research report for the task below. Requirements:\n"
            "1. Start with an executive summary, then develop the analysis.\n"
            "2. Cover every explicit sub-question in the task.\n"
            "3. State key evidence, assumptions, uncertainty, and limitations.\n"
            "4. Use clear headings, tables, or bullet lists where helpful.\n"
            "5. If external evidence is provided, cite source URLs near the relevant claims.\n"
        )

    evidence_block = ""
    if evidence:
        evidence_block = f"\n\n<evidence>\n{evidence}\n</evidence>"
    return f"{instruction}\n<task>\n{prompt}\n</task>{evidence_block}\n\n请输出最终报告。"


def format_criteria_for_judge(criteria_data: dict[str, Any]) -> str:
    """Pass only criterion text and explanations to the judge, not weights."""
    formatted: dict[str, list[dict[str, str]]] = {}
    for dim in DIMENSIONS:
        formatted[dim] = []
        for item in criteria_data.get("criterions", {}).get(dim, []):
            formatted[dim].append(
                {
                    "criterion": str(item.get("criterion", "")),
                    "explanation": str(item.get("explanation", "")),
                }
            )
    return json.dumps(formatted, ensure_ascii=False, indent=2)


def build_race_judge_prompt(
    task_prompt: str,
    article_1: str,
    article_2: str,
    criteria_list: str,
    language: str = "en",
) -> str:
    if language == "zh":
        return f"""
你是一名严格、细致、客观的研究报告评估专家。

你需要比较两篇针对同一深度研究任务的报告。article_1 是待评报告，article_2 是参考报告。

<task>
{task_prompt}
</task>

<article_1>
{article_1}
</article_1>

<article_2>
{article_2}
</article_2>

<criteria_list>
{criteria_list}
</criteria_list>

请严格按照 criteria_list 中每一条 criterion 进行比较评估。对每条 criterion：
1. 简要说明两篇文章分别满足该标准的程度。
2. 给 article_1 和 article_2 分别打 0-10 连续分。
3. 分数含义：0-2 很差，2-4 较差，4-6 中等，6-8 较好，8-10 出色。

只输出可解析 JSON，不要输出其他说明。JSON schema:
{{
  "comprehensiveness": [
    {{
      "criterion": "criterion text copied from criteria_list",
      "analysis": "brief comparative analysis",
      "article_1_score": 0.0,
      "article_2_score": 0.0
    }}
  ],
  "insight": [],
  "instruction_following": [],
  "readability": []
}}
"""

    return f"""
You are a strict, meticulous, and objective research-report evaluator.

Compare two reports written for the same deep research task. article_1 is the target report and article_2 is the reference report.

<task>
{task_prompt}
</task>

<article_1>
{article_1}
</article_1>

<article_2>
{article_2}
</article_2>

<criteria_list>
{criteria_list}
</criteria_list>

Evaluate every criterion in criteria_list. For each criterion:
1. Briefly compare how well the two articles satisfy it.
2. Assign separate continuous 0-10 scores to article_1 and article_2.
3. Score guide: 0-2 very poor, 2-4 poor, 4-6 average, 6-8 good, 8-10 excellent.

Return parseable JSON only, with this schema:
{{
  "comprehensiveness": [
    {{
      "criterion": "criterion text copied from criteria_list",
      "analysis": "brief comparative analysis",
      "article_1_score": 0.0,
      "article_2_score": 0.0
    }}
  ],
  "insight": [],
  "instruction_following": [],
  "readability": []
}}
"""


def build_fact_extract_prompt(article: str, language: str = "en") -> str:
    if language == "zh":
        return f"""
请从研究报告中抽取带有引用 URL 支撑的事实陈述。

输出 JSON 列表，每项包含：
- statement: 被引用支撑的完整事实陈述
- url: 支撑该陈述的 URL

只抽取正文中能定位到 URL 的陈述。不要抽取只有参考文献列表但正文未使用的 URL。

<article>
{article}
</article>

只输出 JSON 列表。
"""

    return f"""
Extract cited factual statements from the research report.

Return a JSON list. Each item must contain:
- statement: the complete factual claim supported by a citation
- url: the URL supporting that claim

Only extract claims that can be linked to a URL in the body of the report.
Do not extract URLs that appear only in a bibliography with no claim.

<article>
{article}
</article>

Return JSON only.
"""


def build_fact_validate_prompt(reference: str, statements: list[str], language: str = "en") -> str:
    numbered = "\n".join(f"{idx + 1}. {statement}" for idx, statement in enumerate(statements))
    if language == "zh":
        return f"""
你会看到一个网页参考资料和若干 statements。请判断每条 statement 相对于参考资料是 supported、unsupported 还是 unknown。

规则：
- 如果参考资料无有效内容或明显抓取失败，该 statement 判为 unknown。
- 如果 statement 的事实或数据能在参考资料中全部或部分找到，判为 supported。
- 如果 statement 的事实和数据在参考资料中找不到，判为 unsupported。

<reference>
{reference}
</reference>

<statements>
{numbered}
</statements>

只输出 JSON 列表，格式为：
[{{"idx": 1, "result": "supported"}}]
"""

    return f"""
You will see a webpage reference and several statements. Judge each statement as supported, unsupported, or unknown with respect to the reference.

Rules:
- If the reference has no valid content or appears to be a failed scrape, mark the statement as unknown.
- If the facts or data in the statement can be found fully or partially in the reference, mark it as supported.
- If the facts and data cannot be found in the reference, mark it as unsupported.

<reference>
{reference}
</reference>

<statements>
{numbered}
</statements>

Return a JSON list only, like:
[{{"idx": 1, "result": "supported"}}]
"""

