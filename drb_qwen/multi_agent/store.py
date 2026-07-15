from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
from typing import Any

from .schemas import (
    AgentResult,
    ClaimRecord,
    EvidenceRecord,
    GlobalResearchState,
    LocalResearchState,
    ResearchExecutionBundle,
    RunPhase,
    SourceRecord,
    content_hash,
    utc_now,
)


SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_name(value: str) -> str:
    cleaned = SAFE_NAME_RE.sub("_", str(value or "").strip()).strip("._")
    return cleaned[:160] or "unnamed"


class RunStore:
    """Small durable JSON store used by the local runtime.

    It deliberately exposes the same logical boundaries a production database or
    durable workflow would use: one global snapshot, per-subtask local snapshots,
    immutable events, and separately stored artifacts.
    """

    def __init__(self, root_dir: str = "") -> None:
        self.root = Path(root_dir).expanduser() if root_dir else None

    @property
    def enabled(self) -> bool:
        return self.root is not None

    def run_dir(self, run_id: str) -> Path | None:
        if self.root is None:
            return None
        return self.root / safe_name(run_id)

    def has_snapshot(self, run_id: str) -> bool:
        path = self._snapshot_path(run_id)
        return bool(path and path.exists())

    def clear_run(self, run_id: str) -> None:
        run_dir = self.run_dir(run_id)
        if run_dir is not None and run_dir.exists():
            shutil.rmtree(run_dir)

    def save_global(self, state: GlobalResearchState) -> None:
        path = self._snapshot_path(state.run_id)
        if path is None:
            return
        if path.exists() and state.phase != RunPhase.CANCELLED:
            try:
                current = json.loads(path.read_text(encoding="utf-8"))
                if current.get("phase") == RunPhase.CANCELLED.value:
                    return
            except Exception:
                pass
        self._write_json_atomic(path, state.to_dict())

    def load_global(self, run_id: str) -> GlobalResearchState | None:
        path = self._snapshot_path(run_id)
        if path is None or not path.exists():
            return None
        try:
            return GlobalResearchState.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            raise RuntimeError(f"Could not load research snapshot {path}: {exc}") from exc

    def save_local(self, state: LocalResearchState) -> None:
        run_dir = self.run_dir(state.run_id)
        if run_dir is None:
            return
        path = run_dir / "local" / f"{safe_name(state.subtask_id)}.json"
        self._write_json_atomic(path, state.to_dict())

    def load_local(self, run_id: str, subtask_id: str) -> LocalResearchState | None:
        run_dir = self.run_dir(run_id)
        if run_dir is None:
            return None
        path = run_dir / "local" / f"{safe_name(subtask_id)}.json"
        if not path.exists():
            return None
        try:
            return LocalResearchState.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            raise RuntimeError(f"Could not load local research state {path}: {exc}") from exc

    def save_bundle(self, run_id: str, bundle: ResearchExecutionBundle) -> None:
        run_dir = self.run_dir(run_id)
        if run_dir is None:
            return
        path = run_dir / "bundles" / f"{safe_name(bundle.result.subtask_id)}.json"
        self._write_json_atomic(
            path,
            {
                "result": bundle.result.to_dict(),
                "local_state": bundle.local_state.to_dict(),
                "sources": [item.to_dict() for item in bundle.sources],
                "evidence": [item.to_dict() for item in bundle.evidence],
                "claims": [item.to_dict() for item in bundle.claims],
                "events": bundle.events,
            },
        )

    def load_bundle(self, run_id: str, subtask_id: str) -> ResearchExecutionBundle | None:
        run_dir = self.run_dir(run_id)
        if run_dir is None:
            return None
        path = run_dir / "bundles" / f"{safe_name(subtask_id)}.json"
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                return None
            result_value = value.get("result")
            local_value = value.get("local_state")
            if not isinstance(result_value, dict) or not isinstance(local_value, dict):
                return None
            return ResearchExecutionBundle(
                result=AgentResult.from_dict(result_value),
                local_state=LocalResearchState.from_dict(local_value),
                sources=[SourceRecord.from_dict(item) for item in value.get("sources", []) if isinstance(item, dict)],
                evidence=[EvidenceRecord.from_dict(item) for item in value.get("evidence", []) if isinstance(item, dict)],
                claims=[ClaimRecord.from_dict(item) for item in value.get("claims", []) if isinstance(item, dict)],
                events=[item for item in value.get("events", []) if isinstance(item, dict)],
            )
        except Exception as exc:
            raise RuntimeError(f"Could not load research bundle {path}: {exc}") from exc

    def save_research_checkpoint(
        self,
        run_id: str,
        *,
        local_state: LocalResearchState,
        sources: list[SourceRecord],
        evidence: list[EvidenceRecord],
        claims: list[ClaimRecord],
        events: list[dict[str, Any]],
        usage: dict[str, int],
        last_decision: dict[str, Any],
    ) -> None:
        """Persist a non-terminal Researcher turn after tools have been reduced."""

        run_dir = self.run_dir(run_id)
        if run_dir is None:
            return
        path = run_dir / "checkpoints" / f"{safe_name(local_state.subtask_id)}.json"
        self._write_json_atomic(
            path,
            {
                "local_state": local_state.to_dict(),
                "sources": [item.to_dict() for item in sources],
                "evidence": [item.to_dict() for item in evidence],
                "claims": [item.to_dict() for item in claims],
                "events": events,
                "usage": usage,
                "last_decision": last_decision,
            },
        )

    def load_research_checkpoint(self, run_id: str, subtask_id: str) -> dict[str, Any] | None:
        run_dir = self.run_dir(run_id)
        if run_dir is None:
            return None
        path = run_dir / "checkpoints" / f"{safe_name(subtask_id)}.json"
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            local_value = value.get("local_state") if isinstance(value, dict) else None
            if not isinstance(local_value, dict):
                return None
            return {
                "local_state": LocalResearchState.from_dict(local_value),
                "sources": [
                    SourceRecord.from_dict(item) for item in value.get("sources", []) if isinstance(item, dict)
                ],
                "evidence": [
                    EvidenceRecord.from_dict(item) for item in value.get("evidence", []) if isinstance(item, dict)
                ],
                "claims": [
                    ClaimRecord.from_dict(item) for item in value.get("claims", []) if isinstance(item, dict)
                ],
                "events": [item for item in value.get("events", []) if isinstance(item, dict)],
                "usage": {
                    str(key): max(0, int(number))
                    for key, number in value.get("usage", {}).items()
                    if isinstance(number, (int, float))
                },
                "last_decision": (
                    dict(value.get("last_decision", {}))
                    if isinstance(value.get("last_decision"), dict)
                    else {}
                ),
            }
        except Exception as exc:
            raise RuntimeError(f"Could not load Researcher checkpoint {path}: {exc}") from exc

    def clear_research_checkpoint(self, run_id: str, subtask_id: str) -> None:
        run_dir = self.run_dir(run_id)
        if run_dir is None:
            return
        path = run_dir / "checkpoints" / f"{safe_name(subtask_id)}.json"
        if path.exists():
            path.unlink()

    def append_event(self, run_id: str, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        event = {
            "event_id": content_hash(f"{run_id}:{event_type}:{utc_now()}:{json.dumps(payload, sort_keys=True, default=str)}")[:24],
            "run_id": run_id,
            "type": event_type,
            "timestamp": utc_now(),
            "payload": payload,
        }
        run_dir = self.run_dir(run_id)
        if run_dir is None:
            return event
        path = run_dir / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        return event

    def load_events(self, run_id: str) -> list[dict[str, Any]]:
        run_dir = self.run_dir(run_id)
        if run_dir is None:
            return []
        path = run_dir / "events.jsonl"
        if not path.exists():
            return []
        output: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    output.append(value)
        return output

    def save_artifact(
        self,
        run_id: str,
        artifact_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        run_dir = self.run_dir(run_id)
        if run_dir is None:
            return ""
        artifact_dir = run_dir / "artifacts"
        text_path = artifact_dir / f"{safe_name(artifact_id)}.txt"
        meta_path = artifact_dir / f"{safe_name(artifact_id)}.json"
        text_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = text_path.with_suffix(".txt.tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(text_path)
        self._write_json_atomic(
            meta_path,
            {
                "artifact_id": artifact_id,
                "text_file": text_path.name,
                "content_hash": content_hash(text),
                "chars": len(text),
                "created_at": utc_now(),
                "metadata": metadata or {},
            },
        )
        return str(text_path)

    def _snapshot_path(self, run_id: str) -> Path | None:
        run_dir = self.run_dir(run_id)
        return run_dir / "global_state.json" if run_dir is not None else None

    @staticmethod
    def _write_json_atomic(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(path)
