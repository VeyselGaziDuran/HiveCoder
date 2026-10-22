"""Unit tests for core/state_machine.py: deterministic transitions, persistence, and anti-cheat guards."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.state_machine import (
    IllegalCompletionError,
    InvalidTransitionError,
    MicroTask,
    Phase,
    StateMachine,
    TaskHistory,
)


def _sample_plan() -> list[MicroTask]:
    return [
        MicroTask(
            id="step-1",
            title="Implement math_ops.py",
            target_file="math_ops.py",
            kind="code",
            acceptance=["add(a, b) returns sum"],
        ),
        MicroTask(
            id="step-2",
            title="Write pytest for math_ops.py",
            target_file="tests/test_math_ops.py",
            kind="test",
            acceptance=["pytest passes"],
        ),
    ]


def test_illegal_phase_transition_is_rejected(tmp_path: Path) -> None:
    sm = StateMachine(tmp_path / "state.json")
    sm.new_task("Build calculator", tmp_path / "ws")
    with pytest.raises(InvalidTransitionError):
        sm.transition(Phase.COMPLETED)


def test_anti_cheat_blocks_unverified_micro_task_and_task_completion(tmp_path: Path) -> None:
    sm = StateMachine(tmp_path / "state.json")
    sm.new_task("Build calculator", tmp_path / "ws")
    sm.transition(Phase.PLANNING)
    sm.set_plan(_sample_plan())
    sm.transition(Phase.CODING)

    sm.start_micro_task("step-1", snapshot="abc1234")
    # No files written yet -> must refuse completion
    with pytest.raises(IllegalCompletionError):
        sm.complete_micro_task("step-1")

    sm.record_files("step-1", ["math_ops.py"], "Implemented add(a, b) -> int.")
    # Not inspected yet -> cannot validate or complete
    with pytest.raises(IllegalCompletionError):
        sm.mark_validated("step-1", True, "py_compile OK")
    with pytest.raises(IllegalCompletionError):
        sm.complete_micro_task("step-1")

    sm.transition(Phase.INSPECTING)
    sm.mark_inspected("step-1", True, "Approved")
    # Not validated yet -> still cannot complete
    with pytest.raises(IllegalCompletionError):
        sm.complete_micro_task("step-1")

    sm.transition(Phase.VALIDATING)
    sm.mark_validated("step-1", True, "py_compile OK")
    sm.complete_micro_task("step-1")

    # Whole task cannot finish while step-2 is still pending
    with pytest.raises(IllegalCompletionError):
        sm.finish()


def test_persistence_resume_and_history(tmp_path: Path) -> None:
    state_file = tmp_path / "state.json"
    log_file = tmp_path / "hive_log.jsonl"
    history_file = tmp_path / "task_history.json"

    sm = StateMachine(state_file, log_file)
    sm.new_task("Build service", tmp_path / "ws")
    sm.transition(Phase.PLANNING)
    sm.set_plan(_sample_plan())
    sm.transition(Phase.CODING)

    # Complete step-1
    sm.start_micro_task("step-1", snapshot="snap-1")
    sm.record_files("step-1", ["math_ops.py"], "Created math_ops.py")
    sm.transition(Phase.INSPECTING)
    sm.mark_inspected("step-1", True, "OK")
    sm.transition(Phase.VALIDATING)
    sm.mark_validated("step-1", True, "OK")
    sm.complete_micro_task("step-1")

    # Start step-2 and simulate mid-task interruption
    sm.transition(Phase.CODING)
    sm.start_micro_task("step-2", snapshot="snap-2")

    # Reload from disk in a fresh StateMachine instance
    sm2 = StateMachine(state_file, log_file)
    loaded = sm2.load()
    assert loaded is not None
    assert sm2.has_resumable()
    interrupted = sm2.prepare_resume()
    assert len(interrupted) == 1
    assert interrupted[0].id == "step-2"
    assert sm2.get("step-1").status == "done"
    assert sm2.get("step-2").status == "pending"

    # Now finish step-2 and close the task
    sm2.transition(Phase.CODING)
    sm2.start_micro_task("step-2", snapshot="snap-2b")
    sm2.record_files("step-2", ["tests/test_math_ops.py"], "Added unit tests")
    sm2.transition(Phase.INSPECTING)
    sm2.mark_inspected("step-2", True, "OK")
    sm2.transition(Phase.VALIDATING)
    sm2.mark_validated("step-2", True, "2 passed")
    sm2.complete_micro_task("step-2")

    final_state = sm2.finish()
    assert final_state.phase == Phase.COMPLETED
    assert not sm2.has_resumable()

    hist = TaskHistory(history_file)
    hist.append(final_state)
    totals = hist.totals()
    assert totals["tasks"] == 1
    assert totals["completed"] == 1
    assert len(sm2.read_log()) >= 5
