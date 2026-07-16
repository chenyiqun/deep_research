from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from difflib import SequenceMatcher
import hashlib
import json
import re
from typing import Any, TypeVar


class RunPhase(str, Enum):
    CREATED = "created"
    SCOPED = "scoped"
    RESEARCHING = "researching"
    WRITING = "writing"
    AUDITING = "auditing"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(str, Enum):
    RESEARCH = "research"
    VERIFY = "verify"
    REPAIR = "repair"


TERMINAL_RUN_PHASES = {
    RunPhase.COMPLETED,
    RunPhase.PARTIAL,
    RunPhase.FAILED,
    RunPhase.CANCELLED,
}
TERMINAL_TASK_STATUSES = {
    TaskStatus.COMPLETED,
    TaskStatus.PARTIAL,
    TaskStatus.FAILED,
    TaskStatus.CANCELLED,
}
SATISFIED_DEPENDENCY_STATUSES = {TaskStatus.COMPLETED, TaskStatus.PARTIAL}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


SEMANTIC_FILLER_RE = re.compile(
    r"(?:尚未|仍未|未能|未提供|未提及|缺少|缺乏|需要补充|无法获得|"
    r"collect|complete|provide|obtain|find|research|analyze|analysis|detailed|specific)",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?")


def semantic_text_key(value: Any) -> str:
    """Normalize boilerplate while retaining entities, metrics, and numbers."""

    text = SEMANTIC_FILLER_RE.sub(" ", normalize_text(value))
    return re.sub(r"[^0-9a-z\u4e00-\u9fff%]+", "", text)


def texts_semantically_equivalent(left: Any, right: Any, threshold: float = 0.88) -> bool:
    """Conservative near-duplicate check for task objectives and active gaps."""

    first = semantic_text_key(left)
    second = semantic_text_key(right)
    if not first or not second:
        return False
    if first == second:
        return True
    if set(NUMBER_RE.findall(first)) != set(NUMBER_RE.findall(second)):
        return False
    shorter, longer = sorted((first, second), key=len)
    if len(shorter) >= 12 and shorter in longer:
        return True
    return SequenceMatcher(None, first, second).ratio() >= threshold


def stable_id(prefix: str, *parts: Any, length: int = 16) -> str:
    raw = "\x1f".join(normalize_text(part) for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def content_hash(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    return value


EnumT = TypeVar("EnumT", bound=Enum)


def enum_value(enum_type: type[EnumT], value: Any, default: EnumT) -> EnumT:
    try:
        return enum_type(str(value))
    except Exception:
        return default


def string_list(value: Any, max_items: int | None = None) -> list[str]:
    if not isinstance(value, list):
        return []
    if max_items is not None and max_items <= 0:
        return []
    output: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item or "").strip()
        key = normalize_text(text)
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(text)
        if max_items is not None and len(output) >= max_items:
            break
    return output


@dataclass
class ResearchBrief:
    question: str
    language: str = "en"
    scope: str = ""
    deliverables: list[str] = field(default_factory=list)
    coverage_targets: list[str] = field(default_factory=list)
    critical_questions: list[str] = field(default_factory=list)
    source_policy: list[str] = field(default_factory=list)
    ambiguities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ResearchBrief":
        return cls(
            question=str(value.get("question", "")),
            language=str(value.get("language", "en")),
            scope=str(value.get("scope", "")),
            deliverables=string_list(value.get("deliverables")),
            coverage_targets=string_list(value.get("coverage_targets")),
            critical_questions=string_list(value.get("critical_questions")),
            source_policy=string_list(value.get("source_policy")),
            ambiguities=string_list(value.get("ambiguities")),
        )


@dataclass
class SubTask:
    id: str
    objective: str
    task_type: TaskType = TaskType.RESEARCH
    rationale: str = ""
    coverage_targets: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    priority: int = 50
    max_steps: int = 3
    max_tool_calls: int = 9
    required_source_types: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    attempts: int = 0
    result_summary: str = ""
    error: str = ""
    created_by: str = "main"
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SubTask":
        return cls(
            id=str(value.get("id", "")),
            objective=str(value.get("objective", "")),
            task_type=enum_value(TaskType, value.get("task_type", value.get("type")), TaskType.RESEARCH),
            rationale=str(value.get("rationale", "")),
            coverage_targets=string_list(value.get("coverage_targets")),
            depends_on=string_list(value.get("depends_on")),
            priority=max(0, min(100, safe_int(value.get("priority"), 50))),
            max_steps=max(1, safe_int(value.get("max_steps"), 3)),
            max_tool_calls=max(1, safe_int(value.get("max_tool_calls"), 9)),
            required_source_types=string_list(value.get("required_source_types")),
            status=enum_value(TaskStatus, value.get("status"), TaskStatus.PENDING),
            attempts=max(0, safe_int(value.get("attempts"), 0)),
            result_summary=str(value.get("result_summary", "")),
            error=str(value.get("error", "")),
            created_by=str(value.get("created_by", "main")),
            created_at=str(value.get("created_at", utc_now())),
            updated_at=str(value.get("updated_at", utc_now())),
        )


@dataclass
class SourceRecord:
    id: str
    url: str
    title: str
    query: str
    publish_date: str = ""
    media: str = ""
    source_quality: str = "snippet_only"
    source_type: str = "unknown"
    authority_score: float = 0.5
    extraction_method: str = "snippet"
    independence_group: str = ""
    artifact_id: str = ""
    content_hash: str = ""
    fetched_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SourceRecord":
        return cls(
            id=str(value.get("id", "")),
            url=str(value.get("url", "")),
            title=str(value.get("title", "")),
            query=str(value.get("query", "")),
            publish_date=str(value.get("publish_date", "")),
            media=str(value.get("media", "")),
            source_quality=str(value.get("source_quality", "snippet_only")),
            source_type=str(value.get("source_type", "unknown")),
            authority_score=max(0.0, min(1.0, safe_float(value.get("authority_score"), 0.5))),
            extraction_method=str(value.get("extraction_method", "snippet")),
            independence_group=str(value.get("independence_group", "")),
            artifact_id=str(value.get("artifact_id", "")),
            content_hash=str(value.get("content_hash", "")),
            fetched_at=str(value.get("fetched_at", utc_now())),
        )


@dataclass
class EvidenceRecord:
    id: str
    source_id: str
    subtask_id: str
    claim_text: str
    excerpt: str
    locator: str = ""
    confidence: str = "medium"
    relation: str = "supports"
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvidenceRecord":
        return cls(
            id=str(value.get("id", "")),
            source_id=str(value.get("source_id", "")),
            subtask_id=str(value.get("subtask_id", "")),
            claim_text=str(value.get("claim_text", "")),
            excerpt=str(value.get("excerpt", "")),
            locator=str(value.get("locator", "")),
            confidence=str(value.get("confidence", "medium")),
            relation=str(value.get("relation", "supports")),
            created_at=str(value.get("created_at", utc_now())),
        )


@dataclass
class ClaimRecord:
    id: str
    text: str
    subtask_id: str
    evidence_ids: list[str] = field(default_factory=list)
    confidence: str = "medium"
    status: str = "provisional"
    qualifiers: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ClaimRecord":
        return cls(
            id=str(value.get("id", "")),
            text=str(value.get("text", "")),
            subtask_id=str(value.get("subtask_id", "")),
            evidence_ids=string_list(value.get("evidence_ids")),
            confidence=str(value.get("confidence", "medium")),
            status=str(value.get("status", "provisional")),
            qualifiers=string_list(value.get("qualifiers")),
            created_at=str(value.get("created_at", utc_now())),
            updated_at=str(value.get("updated_at", utc_now())),
        )


@dataclass
class BudgetUsage:
    main_calls: int = 0
    researcher_calls: int = 0
    reader_calls: int = 0
    writer_calls: int = 0
    auditor_calls: int = 0
    search_calls: int = 0
    fetch_calls: int = 0
    tool_calls: int = 0
    completed_subtasks: int = 0
    failed_subtasks: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        return jsonable(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BudgetUsage":
        return cls(**{name: max(0, safe_int(value.get(name), 0)) for name in cls.__dataclass_fields__})

    def add(self, value: dict[str, Any]) -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + max(0, safe_int(value.get(name), 0)))


@dataclass
class LocalResearchState:
    run_id: str
    subtask_id: str
    objective: str
    status: TaskStatus = TaskStatus.PENDING
    version: int = 0
    step: int = 0
    queries: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    claim_ids: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    resolved_gaps: list[str] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    recent_observations: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: int = 0
    search_calls: int = 0
    answer_summary: str = ""
    stop_reason: str = ""
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LocalResearchState":
        return cls(
            run_id=str(value.get("run_id", "")),
            subtask_id=str(value.get("subtask_id", "")),
            objective=str(value.get("objective", "")),
            status=enum_value(TaskStatus, value.get("status"), TaskStatus.PENDING),
            version=max(0, safe_int(value.get("version"), 0)),
            step=max(0, safe_int(value.get("step"), 0)),
            queries=string_list(value.get("queries")),
            source_ids=string_list(value.get("source_ids")),
            evidence_ids=string_list(value.get("evidence_ids")),
            claim_ids=string_list(value.get("claim_ids")),
            gaps=string_list(value.get("gaps")),
            resolved_gaps=string_list(value.get("resolved_gaps")),
            conflicts=[item for item in value.get("conflicts", []) if isinstance(item, dict)],
            recent_observations=[item for item in value.get("recent_observations", []) if isinstance(item, dict)],
            tool_calls=max(0, safe_int(value.get("tool_calls"), 0)),
            search_calls=max(0, safe_int(value.get("search_calls"), 0)),
            answer_summary=str(value.get("answer_summary", "")),
            stop_reason=str(value.get("stop_reason", "")),
            created_at=str(value.get("created_at", utc_now())),
            updated_at=str(value.get("updated_at", utc_now())),
        )


@dataclass
class AgentResult:
    subtask_id: str
    status: TaskStatus
    answer_summary: str
    claim_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    unresolved_gaps: list[str] = field(default_factory=list)
    resolved_gaps: list[str] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    suggested_followups: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AgentResult":
        return cls(
            subtask_id=str(value.get("subtask_id", "")),
            status=enum_value(TaskStatus, value.get("status"), TaskStatus.PARTIAL),
            answer_summary=str(value.get("answer_summary", "")),
            claim_ids=string_list(value.get("claim_ids")),
            evidence_ids=string_list(value.get("evidence_ids")),
            source_ids=string_list(value.get("source_ids")),
            unresolved_gaps=string_list(value.get("unresolved_gaps")),
            resolved_gaps=string_list(value.get("resolved_gaps")),
            conflicts=[item for item in value.get("conflicts", []) if isinstance(item, dict)],
            suggested_followups=string_list(value.get("suggested_followups")),
            usage={str(k): max(0, safe_int(v, 0)) for k, v in value.get("usage", {}).items()},
            error=str(value.get("error", "")),
        )


@dataclass
class AuditResult:
    passed: bool
    issues: list[dict[str, Any]] = field(default_factory=list)
    repair_tasks: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AuditResult":
        return cls(
            passed=bool(value.get("passed", False)),
            issues=[item for item in value.get("issues", []) if isinstance(item, dict)],
            repair_tasks=[item for item in value.get("repair_tasks", []) if isinstance(item, dict)],
            summary=str(value.get("summary", "")),
        )


@dataclass
class GlobalResearchState:
    run_id: str
    task: dict[str, Any]
    phase: RunPhase = RunPhase.CREATED
    state_version: int = 0
    brief: ResearchBrief | None = None
    tasks: dict[str, SubTask] = field(default_factory=dict)
    sources: dict[str, SourceRecord] = field(default_factory=dict)
    evidence: dict[str, EvidenceRecord] = field(default_factory=dict)
    claims: dict[str, ClaimRecord] = field(default_factory=dict)
    agent_results: dict[str, AgentResult] = field(default_factory=dict)
    query_ledger: list[str] = field(default_factory=list)
    coverage: dict[str, str] = field(default_factory=dict)
    coverage_details: dict[str, dict[str, Any]] = field(default_factory=dict)
    gaps: list[str] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    budget: BudgetUsage = field(default_factory=BudgetUsage)
    main_round: int = 0
    audit_round: int = 0
    article: str = ""
    audit: AuditResult | None = None
    stop_reason: str = ""
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    @property
    def terminal(self) -> bool:
        return self.phase in TERMINAL_RUN_PHASES

    def bump_version(self) -> None:
        self.state_version += 1
        self.updated_at = utc_now()

    def to_dict(self) -> dict[str, Any]:
        output = jsonable(self)
        output["findings"] = [claim.text for claim in self.claims.values()]
        return output

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GlobalResearchState":
        brief_value = value.get("brief")
        audit_value = value.get("audit")
        return cls(
            run_id=str(value.get("run_id", "")),
            task=dict(value.get("task", {})),
            phase=enum_value(RunPhase, value.get("phase"), RunPhase.CREATED),
            state_version=max(0, safe_int(value.get("state_version"), 0)),
            brief=ResearchBrief.from_dict(brief_value) if isinstance(brief_value, dict) else None,
            tasks={str(k): SubTask.from_dict(v) for k, v in value.get("tasks", {}).items() if isinstance(v, dict)},
            sources={str(k): SourceRecord.from_dict(v) for k, v in value.get("sources", {}).items() if isinstance(v, dict)},
            evidence={str(k): EvidenceRecord.from_dict(v) for k, v in value.get("evidence", {}).items() if isinstance(v, dict)},
            claims={str(k): ClaimRecord.from_dict(v) for k, v in value.get("claims", {}).items() if isinstance(v, dict)},
            agent_results={str(k): AgentResult.from_dict(v) for k, v in value.get("agent_results", {}).items() if isinstance(v, dict)},
            query_ledger=string_list(value.get("query_ledger"), 256),
            coverage={str(k): str(v) for k, v in value.get("coverage", {}).items()},
            coverage_details={
                str(k): dict(v)
                for k, v in value.get("coverage_details", {}).items()
                if isinstance(v, dict)
            },
            gaps=string_list(value.get("gaps")),
            conflicts=[item for item in value.get("conflicts", []) if isinstance(item, dict)],
            budget=BudgetUsage.from_dict(value.get("budget", {})),
            main_round=max(0, safe_int(value.get("main_round"), 0)),
            audit_round=max(0, safe_int(value.get("audit_round"), 0)),
            article=str(value.get("article", "")),
            audit=AuditResult.from_dict(audit_value) if isinstance(audit_value, dict) else None,
            stop_reason=str(value.get("stop_reason", "")),
            created_at=str(value.get("created_at", utc_now())),
            updated_at=str(value.get("updated_at", utc_now())),
        )

    def compact_summary(self, max_claims: int = 80) -> dict[str, Any]:
        tasks = [task.to_dict() for task in self.tasks.values()]
        claims = [
            {
                "id": claim.id,
                "text": claim.text,
                "evidence_ids": claim.evidence_ids,
                "confidence": claim.confidence,
                "status": claim.status,
            }
            for claim in list(self.claims.values())[-max_claims:]
        ]
        return {
            "run_id": self.run_id,
            "phase": self.phase.value,
            "state_version": self.state_version,
            "brief": self.brief.to_dict() if self.brief else None,
            "tasks": tasks,
            "query_ledger": self.query_ledger[-128:],
            "claims": claims,
            "coverage": self.coverage,
            "coverage_details": self.coverage_details,
            "gaps": self.gaps,
            "conflicts": self.conflicts,
            "budget": self.budget.to_dict(),
            "main_round": self.main_round,
            "audit_round": self.audit_round,
            "audit": self.audit.to_dict() if self.audit else None,
        }


@dataclass
class ResearchExecutionBundle:
    result: AgentResult
    local_state: LocalResearchState
    sources: list[SourceRecord] = field(default_factory=list)
    evidence: list[EvidenceRecord] = field(default_factory=list)
    claims: list[ClaimRecord] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def compact_json(value: Any, max_chars: int) -> str:
    text = json.dumps(jsonable(value), ensure_ascii=False, indent=2)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"
