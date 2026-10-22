"""Deterministic task state machine with crash-safe persistence and JSONL logging.

Every mutation is flushed atomically to ``state.json`` so HiveCoder can resume
from the exact micro-step after a crash, a thermal cooldown or a reboot.

The machine also enforces *honest completion*: a micro-task can only be closed
when it has real evidence (files written, inspection approved, validation
passed), and the whole task can only finish when every micro-task is done.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class Phase(str, Enum):
    IDLE = "idle"
    PLANNING = "planning"
    CODING = "coding"
    INSPECTING = "inspecting"
    VALIDATING = "validating"
    INFIRMARY = "infirmary"
    COOLING = "cooling"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


TERMINAL_PHASES: frozenset[Phase] = frozenset({Phase.COMPLETED, Phase.FAILED})

TRANSITIONS: dict[Phase, frozenset[Phase]] = {
    Phase.IDLE: frozenset({Phase.PLANNING, Phase.PAUSED, Phase.FAILED}),
    Phase.PLANNING: frozenset({Phase.CODING, Phase.PAUSED, Phase.FAILED}),
    Phase.CODING: frozenset({Phase.INSPECTING, Phase.INFIRMARY, Phase.PAUSED, Phase.FAILED}),
    Phase.INSPECTING: frozenset({Phase.VALIDATING, Phase.INFIRMARY, Phase.PAUSED, Phase.FAILED}),
    Phase.VALIDATING: frozenset(
        {Phase.CODING, Phase.INFIRMARY, Phase.COOLING, Phase.COMPLETED, Phase.PAUSED, Phase.FAILED}
    ),
    Phase.INFIRMARY: frozenset({Phase.INSPECTING, Phase.PAUSED, Phase.FAILED}),
    Phase.COOLING: frozenset({Phase.CODING, Phase.COMPLETED, Phase.PAUSED, Phase.FAILED}),
    Phase.PAUSED: frozenset({Phase.PLANNING, Phase.CODING, Phase.COOLING, Phase.COMPLETED, Phase.FAILED}),
    Phase.COMPLETED: frozenset(),
    Phase.FAILED: frozenset({Phase.PAUSED}),
}


class InvalidTransitionError(RuntimeError):
    """Raised when a phase change is not allowed by :data:`TRANSITIONS`."""


class IllegalCompletionError(RuntimeError):
    """Raised when someone tries to close a micro-task or task without evidence."""


class NoActiveTaskError(RuntimeError):
    """Raised when an operation needs a task but none is loaded."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


TaskStatus = Literal["pending", "in_progress", "done", "failed"]


class MicroTask(BaseModel):
    id: str
    title: str
    description: str = ""
    target_file: str
    kind: Literal["code", "test"] = "code"
    acceptance: list[str] = Field(default_factory=list)
    status: TaskStatus = "pending"
    attempts: int = 0
    medic_attempts: int = 0
    files_written: list[str] = Field(default_factory=list)
    relay_note: str = ""
    inspected: bool = False
    inspection_notes: str = ""
    validated: bool = False
    validation_evidence: str = ""
    last_error: str = ""
    snapshot: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


class Metrics(BaseModel):
    llm_calls: int = 0
    prompt_chars: int = 0
    response_chars: int = 0
    tokens_saved: int = 0
    medic_runs: int = 0
    medic_successes: int = 0
    inspector_rejections: int = 0
    validation_failures: int = 0
    gc_items_removed: int = 0
    cooldowns: int = 0
    work_seconds: float = 0.0


class TaskState(BaseModel):
    task_id: str
    goal: str
    workspace: str
    phase: Phase = Phase.IDLE
    micro_tasks: list[MicroTask] = Field(default_factory=list)
    metrics: Metrics = Field(default_factory=Metrics)
    thermal: dict[str, Any] = Field(default_factory=dict)
    pre_task_snapshot: Optional[str] = None
    error: str = ""
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    started_epoch: float = Field(default_factory=time.time)

    def counts(self) -> dict[str, int]:
        result = {"pending": 0, "in_progress": 0, "done": 0, "failed": 0}
        for mt in self.micro_tasks:
            result[mt.status] += 1
        result["total"] = len(self.micro_tasks)
        return result


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class StateMachine:
    """Owns the single active :class:`TaskState` and its on-disk representation."""

    def __init__(self, state_file: str | Path, log_file: str | Path | None = None) -> None:
        self.state_file = Path(state_file)
        self.log_file = Path(log_file) if log_file else self.state_file.with_name("hive_log.jsonl")
        self.state: Optional[TaskState] = None

    # ------------------------------------------------------------------ persistence
    def save(self) -> None:
        st = self._require()
        st.updated_at = utc_now()
        _atomic_write_text(self.state_file, st.model_dump_json(indent=2))

    def load(self) -> Optional[TaskState]:
        if not self.state_file.exists():
            self.state = None
            return None
        raw = self.state_file.read_text(encoding="utf-8").strip()
        if not raw:
            self.state = None
            return None
        self.state = TaskState.model_validate_json(raw)
        return self.state

    def clear(self) -> None:
        self.state = None
        if self.state_file.exists():
            self.state_file.unlink()

    def has_resumable(self) -> bool:
        st = self.state if self.state is not None else self.load()
        return st is not None and st.phase != Phase.COMPLETED

    def log_event(self, event: str, **data: Any) -> None:
        record = {
            "ts": utc_now(),
            "task_id": self.state.task_id if self.state else None,
            "phase": self.state.phase.value if self.state else None,
            "event": event,
            "data": data,
        }
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with self.log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def read_log(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.log_file.exists():
            return []
        lines = self.log_file.read_text(encoding="utf-8").splitlines()
        out: list[dict[str, Any]] = []
        for line in lines[-limit:]:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    # ------------------------------------------------------------------ lifecycle
    def new_task(self, goal: str, workspace: str | Path) -> TaskState:
        if not goal.strip():
            raise ValueError("Task goal must not be empty")
        self.state = TaskState(
            task_id=f"task-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}",
            goal=goal.strip(),
            workspace=str(Path(workspace).resolve()),
        )
        self.save()
        self.log_event("task_created", goal=self.state.goal, workspace=self.state.workspace)
        return self.state

    def transition(self, target: Phase, reason: str = "") -> None:
        st = self._require()
        current = st.phase
        if target == current:
            return
        if target not in TRANSITIONS[current]:
            raise InvalidTransitionError(f"Illegal transition {current.value} -> {target.value}")
        st.phase = target
        self.save()
        self.log_event("transition", source=current.value, target=target.value, reason=reason)

    def set_plan(self, micro_tasks: list[MicroTask]) -> None:
        st = self._require()
        if st.phase != Phase.PLANNING:
            raise InvalidTransitionError("A plan can only be set during the PLANNING phase")
        if not micro_tasks:
            raise ValueError("A plan needs at least one micro-task")
        ids = [mt.id for mt in micro_tasks]
        if len(set(ids)) != len(ids):
            raise ValueError("Micro-task ids must be unique")
        st.micro_tasks = micro_tasks
        self.save()
        self.log_event("plan_set", count=len(micro_tasks), ids=ids)

    # ------------------------------------------------------------------ micro-tasks
    def get(self, task_id: str) -> MicroTask:
        for mt in self._require().micro_tasks:
            if mt.id == task_id:
                return mt
        raise KeyError(task_id)

    def next_pending(self) -> Optional[MicroTask]:
        for mt in self._require().micro_tasks:
            if mt.status in ("pending", "in_progress"):
                return mt
        return None

    def start_micro_task(self, task_id: str, snapshot: Optional[str] = None) -> MicroTask:
        mt = self.get(task_id)
        if mt.status == "done":
            raise IllegalCompletionError(f"{task_id} is already done")
        mt.status = "in_progress"
        mt.attempts += 1
        mt.files_written = []
        mt.relay_note = ""
        mt.inspected = False
        mt.validated = False
        mt.snapshot = snapshot
        mt.started_at = utc_now()
        self.save()
        self.log_event("micro_started", micro_id=task_id, attempt=mt.attempts)
        return mt

    def record_files(self, task_id: str, files: list[str], relay_note: str) -> None:
        mt = self.get(task_id)
        for f in files:
            if f not in mt.files_written:
                mt.files_written.append(f)
        if relay_note:
            mt.relay_note = relay_note
        mt.inspected = False
        mt.validated = False
        self.save()
        self.log_event("files_written", micro_id=task_id, files=files)

    def mark_inspected(self, task_id: str, approved: bool, notes: str = "") -> None:
        mt = self.get(task_id)
        mt.inspected = approved
        mt.inspection_notes = notes
        if not approved:
            mt.validated = False
            mt.last_error = notes
        self.save()
        self.log_event("inspected", micro_id=task_id, approved=approved, notes=notes[:500])

    def mark_validated(self, task_id: str, passed: bool, evidence: str) -> None:
        mt = self.get(task_id)
        if passed and not mt.inspected:
            raise IllegalCompletionError(f"{task_id} cannot be validated before inspection approval")
        mt.validated = passed
        mt.validation_evidence = evidence[-2000:]
        if not passed:
            mt.last_error = evidence[-4000:]
        self.save()
        self.log_event("validated", micro_id=task_id, passed=passed)

    def complete_micro_task(self, task_id: str) -> MicroTask:
        """Close a micro-task. Refuses to do so without real evidence."""
        mt = self.get(task_id)
        if mt.status != "in_progress":
            raise IllegalCompletionError(f"{task_id} is not in progress (status={mt.status})")
        if not mt.files_written:
            raise IllegalCompletionError(f"{task_id} produced no files")
        if not mt.inspected:
            raise IllegalCompletionError(f"{task_id} was not approved by the inspector")
        if not mt.validated:
            raise IllegalCompletionError(f"{task_id} has no passing validation evidence")
        mt.status = "done"
        mt.last_error = ""
        mt.finished_at = utc_now()
        self.save()
        self.log_event("micro_completed", micro_id=task_id, files=mt.files_written)
        return mt

    def fail_micro_task(self, task_id: str, error: str) -> MicroTask:
        mt = self.get(task_id)
        mt.status = "failed"
        mt.last_error = error[-4000:]
        mt.finished_at = utc_now()
        self.save()
        self.log_event("micro_failed", micro_id=task_id, error=error[-1000:])
        return mt

    def finish(self) -> TaskState:
        """Mark the task COMPLETED. Every micro-task must be verifiably done."""
        st = self._require()
        if not st.micro_tasks:
            raise IllegalCompletionError("Cannot complete a task without a plan")
        unfinished = [mt.id for mt in st.micro_tasks if mt.status != "done"]
        if unfinished:
            raise IllegalCompletionError(f"Unfinished micro-tasks: {', '.join(unfinished)}")
        unproven = [mt.id for mt in st.micro_tasks if not (mt.inspected and mt.validated)]
        if unproven:
            raise IllegalCompletionError(f"Micro-tasks without evidence: {', '.join(unproven)}")
        self.transition(Phase.COMPLETED, "all micro-tasks verified")
        return st

    def fail(self, error: str) -> TaskState:
        st = self._require()
        st.error = error
        if st.phase != Phase.FAILED:
            if Phase.FAILED not in TRANSITIONS[st.phase]:
                st.phase = Phase.PAUSED
            self.transition(Phase.FAILED, error[:200])
        self.save()
        return st

    def pause(self, reason: str) -> TaskState:
        st = self._require()
        if st.phase not in TERMINAL_PHASES or st.phase == Phase.FAILED:
            self.transition(Phase.PAUSED, reason)
        return st

    def prepare_resume(self) -> list[MicroTask]:
        """Put a crashed / paused / failed task back into a resumable state.

        Returns the micro-tasks that were interrupted (so the caller can roll
        their partial file changes back to their snapshots).
        """
        st = self._require()
        if st.phase == Phase.COMPLETED:
            raise IllegalCompletionError("Task already completed; nothing to resume")
        interrupted: list[MicroTask] = []
        for mt in st.micro_tasks:
            if mt.status in ("in_progress", "failed"):
                interrupted.append(mt.model_copy(deep=True))
                mt.status = "pending"
                mt.files_written = []
                mt.relay_note = ""
                mt.inspected = False
                mt.validated = False
                mt.medic_attempts = 0
        if st.phase != Phase.PAUSED:
            if Phase.PAUSED in TRANSITIONS[st.phase]:
                self.transition(Phase.PAUSED, "resume requested")
            else:
                st.phase = Phase.PAUSED
        st.error = ""
        self.save()
        self.log_event("resume_prepared", interrupted=[m.id for m in interrupted])
        return interrupted

    def _require(self) -> TaskState:
        if self.state is None:
            raise NoActiveTaskError("No active task loaded")
        return self.state


class TaskHistory:
    """Append-only record of finished tasks and their metrics."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        raw = self.path.read_text(encoding="utf-8").strip()
        if not raw:
            return []
        data = json.loads(raw)
        return data if isinstance(data, list) else []

    def append(self, state: TaskState) -> dict[str, Any]:
        counts = state.counts()
        record = {
            "task_id": state.task_id,
            "goal": state.goal,
            "workspace": state.workspace,
            "status": state.phase.value,
            "micro_tasks_total": counts["total"],
            "micro_tasks_done": counts["done"],
            "files": sorted({f for mt in state.micro_tasks for f in mt.files_written}),
            "metrics": state.metrics.model_dump(),
            "error": state.error,
            "duration_seconds": round(time.time() - state.started_epoch, 1),
            "created_at": state.created_at,
            "finished_at": utc_now(),
        }
        history = self.load()
        history.append(record)
        _atomic_write_text(self.path, json.dumps(history, indent=2, ensure_ascii=False) + "\n")
        return record

    def totals(self) -> dict[str, Any]:
        history = self.load()
        return {
            "tasks": len(history),
            "completed": sum(1 for h in history if h.get("status") == "completed"),
            "tokens_saved": sum(int(h.get("metrics", {}).get("tokens_saved", 0)) for h in history),
            "medic_runs": sum(int(h.get("metrics", {}).get("medic_runs", 0)) for h in history),
        }
