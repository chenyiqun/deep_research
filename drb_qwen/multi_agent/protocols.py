from __future__ import annotations

from typing import Any


def json_response_format(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    """OpenAI-compatible JSON Schema response format understood by recent vLLM."""

    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": schema,
        },
    }


STRING_ARRAY = {"type": "array", "items": {"type": "string"}}


MAIN_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "research_brief": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "language": {"type": "string"},
                "scope": {"type": "string"},
                "deliverables": STRING_ARRAY,
                "coverage_targets": STRING_ARRAY,
                "critical_questions": STRING_ARRAY,
                "source_policy": STRING_ARRAY,
                "ambiguities": STRING_ARRAY,
            },
            "required": ["question", "language", "coverage_targets", "critical_questions"],
        },
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "task_type": {"type": "string", "enum": ["research", "verify", "repair"]},
                    "objective": {"type": "string"},
                    "rationale": {"type": "string"},
                    "coverage_targets": STRING_ARRAY,
                    "depends_on": STRING_ARRAY,
                    "priority": {"type": "integer"},
                    "max_steps": {"type": "integer"},
                    "max_tool_calls": {"type": "integer"},
                    "required_source_types": STRING_ARRAY,
                },
                "required": ["id", "objective", "coverage_targets", "depends_on"],
            },
        },
        "wakeup_policy": {"type": "object"},
    },
    "required": ["research_brief", "tasks"],
}


MAIN_REPLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "base_state_version": {"type": "integer"},
        "action": {"type": "string", "enum": ["continue", "write", "partial"]},
        "reason": {"type": "string"},
        "operations": {"type": "array", "items": {"type": "object"}},
    },
    "required": ["base_state_version", "action", "reason", "operations"],
}


RESEARCHER_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "base_local_version": {"type": "integer"},
        "assessment": {
            "type": "object",
            "properties": {
                "coverage": {"type": "string", "enum": ["none", "partial", "sufficient"]},
                "primary_gap": {"type": "string"},
            },
            "required": ["coverage", "primary_gap"],
        },
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["SEARCH"]},
                    "query": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["type", "query"],
            },
        },
        "add_gaps": STRING_ARRAY,
        "resolved_gaps": STRING_ARRAY,
        "add_conflicts": {"type": "array", "items": {"type": "object"}},
        "suggested_followups": STRING_ARRAY,
        "answer_summary": {"type": "string"},
        "finish": {"type": "boolean"},
        "stop_reason": {"type": "string"},
    },
    "required": ["base_local_version", "assessment", "actions", "answer_summary", "finish"],
}


READER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "relevance": {"type": "number"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "excerpt": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "relation": {"type": "string", "enum": ["supports", "refutes", "qualifies"]},
                    "locator": {"type": "string"},
                    "qualifiers": STRING_ARRAY,
                },
                "required": ["text", "excerpt", "confidence", "relation"],
            },
        },
        "gaps": STRING_ARRAY,
        "conflicts": {"type": "array", "items": {"type": "object"}},
        "limitations": STRING_ARRAY,
        "injection_detected": {"type": "boolean"},
    },
    "required": ["relevance", "claims", "gaps", "conflicts", "limitations"],
}


AUDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "summary": {"type": "string"},
        "issues": {"type": "array", "items": {"type": "object"}},
        "repair_tasks": {"type": "array", "items": {"type": "object"}},
    },
    "required": ["passed", "summary", "issues", "repair_tasks"],
}
