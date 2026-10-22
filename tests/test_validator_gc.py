"""Unit & End-to-End Integration tests for Validator, GarbageCollector, GitGuard, and HiveOrchestrator."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from core.config import DEFAULT_CONFIG
from core.garbage_collector import GarbageCollector
from core.git_guard import GitGuard
from core.llm_client import LLMClient
from core.orchestrator import HiveOrchestrator
from core.state_machine import Phase
from core.validator import Validator


def test_validator_injects_pythonpath_and_runs_pytest(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    (workspace / "tests").mkdir(parents=True)

    (workspace / "app.py").write_text(
        "def get_words() -> list[dict]:\n"
        "    return [{'word': 'hive', 'meaning': 'kovan'}]\n",
        encoding="utf-8",
    )
    (workspace / "tests" / "test_app.py").write_text(
        "from app import get_words\n\n"
        "def test_get_words_returns_list() -> None:\n"
        "    items = get_words()\n"
        "    assert isinstance(items, list)\n"
        "    assert items[0]['word'] == 'hive'\n",
        encoding="utf-8",
    )

    validator = Validator(workspace=workspace)
    result = validator.validate(["app.py", "tests/test_app.py"], run_tests=True)
    assert result.passed is True
    assert result.stage == "pytest"


def test_garbage_collector_cleans_caches_and_preserves_git_and_venv(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    (workspace / ".hive_tmp").mkdir(parents=True)
    (workspace / ".hive_tmp" / "scratch.txt").write_text("temp", encoding="utf-8")
    (workspace / ".pytest_cache").mkdir()
    (workspace / "__pycache__").mkdir()
    (workspace / "__pycache__" / "app.cpython-313.pyc").write_bytes(b"1234")
    (workspace / "orphan.pyc").write_bytes(b"5678")
    (workspace / ".coverage").write_text("cov", encoding="utf-8")

    # Protected directories must remain untouched
    (workspace / ".git").mkdir()
    (workspace / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (workspace / "venv").mkdir()
    (workspace / "venv" / "keep.pyc").write_bytes(b"keep")

    gc = GarbageCollector()
    report = gc.collect(workspace)
    assert report.count >= 5
    assert not (workspace / ".hive_tmp").exists()
    assert not (workspace / ".pytest_cache").exists()
    assert not (workspace / "__pycache__").exists()
    assert not (workspace / "orphan.pyc").exists()
    assert not (workspace / ".coverage").exists()
    assert (workspace / ".git" / "HEAD").is_file()
    assert (workspace / "venv" / "keep.pyc").is_file()


def test_git_guard_snapshot_and_rollback(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    store = tmp_path / "snapshots"
    workspace.mkdir()
    target = workspace / "service.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")

    guard = GitGuard(workspace=workspace, store_dir=store, enabled=True)
    snap = guard.snapshot("initial")
    assert snap is not None

    target.write_text("VALUE = 999\n", encoding="utf-8")
    (workspace / "extra.py").write_text("BROKEN = True\n", encoding="utf-8")

    assert guard.rollback(snap) is True
    assert target.read_text(encoding="utf-8") == "VALUE = 1\n"
    assert not (workspace / "extra.py").exists()


class ScriptedLLMClient(LLMClient):
    """Deterministic scripted LLM client for end-to-end orchestration testing."""

    def __init__(self, config: dict[str, Any], scripted_replies: list[str]) -> None:
        super().__init__(config)
        self.scripted_replies = list(scripted_replies)
        self.prompts_seen: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.prompts_seen.append((system, user))
        self.stats.calls += 1
        self.stats.prompt_chars += len(system) + len(user)
        if not self.scripted_replies:
            raise AssertionError("ScriptedLLMClient ran out of replies")
        reply = self.scripted_replies.pop(0)
        self.stats.response_chars += len(reply)
        return reply


def test_end_to_end_pipeline_with_infirmary_self_healing(tmp_path: Path) -> None:
    """Verify Architect -> Coder -> Blackboard Note -> Inspector -> Tester -> Validator -> Medic -> GC -> Completed."""
    workspace = tmp_path / "vocab_app"
    memory_dir = tmp_path / "memory"

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["memory_dir"] = str(memory_dir)
    cfg["default_workspace"] = str(workspace)
    cfg["thermal"]["profile"] = "turbo"
    cfg["thermal"]["profiles"]["turbo"]["micro_pause_seconds"] = 0

    # 1. Architect plan response
    architect_reply = json.dumps(
        {
            "micro_tasks": [
                {
                    "id": "step-1",
                    "title": "Create app.py with get_words()",
                    "target_file": "app.py",
                    "kind": "code",
                    "description": "Implement get_words() -> list[dict] returning vocabulary entries.",
                    "acceptance": ["get_words() returns a non-empty list of dicts"],
                },
                {
                    "id": "step-2",
                    "title": "Write pytest suite in tests/test_app.py",
                    "target_file": "tests/test_app.py",
                    "kind": "test",
                    "description": "Test get_words() from app.py.",
                    "acceptance": ["All pytest assertions pass"],
                },
            ]
        }
    )

    # 2. CoderAgent implements app.py and leaves a Relay Note
    coder_reply = """\
<<<FILE_PATH>>>
app.py
<<<END_FILE_PATH>>>
<<<CODE>>>
def get_words() -> list[dict]:
    return [
        {"en": "hive", "tr": "kovan"},
        {"en": "ember", "tr": "kor"},
    ]
<<<END_CODE>>>
<<<RELAY_NOTE>>>
Implemented get_words() -> list[dict] inside app.py. Test agent must import get_words from app.
<<<END_RELAY_NOTE>>>
"""

    # 3. InspectorAgent approves step-1
    inspector_approve = '{"approved": true, "issues": [], "summary": "Clean implementation."}'

    # 4. TesterAgent writes a test with a failing assertion first (triggers Validator failure -> Infirmary)
    tester_buggy_reply = """\
<<<FILE_PATH>>>
tests/test_app.py
<<<END_FILE_PATH>>>
<<<CODE>>>
from app import get_words


def test_get_words_count() -> None:
    words = get_words()
    assert len(words) == 999
<<<END_CODE>>>
<<<RELAY_NOTE>>>
Added unit test for get_words() in tests/test_app.py.
<<<END_RELAY_NOTE>>>
"""

    # 5. InspectorAgent approves syntax/imports of step-2 (so Validator runs pytest and catches the assertion bug)
    inspector_approve_2 = '{"approved": true, "issues": [], "summary": "Imports match Blackboard."}'

    # 6. MedicAgent (Infirmary) receives traceback + Blackboard and repairs tests/test_app.py
    medic_repair_reply = """\
<<<FILE_PATH>>>
tests/test_app.py
<<<END_FILE_PATH>>>
<<<CODE>>>
from app import get_words


def test_get_words_count() -> None:
    words = get_words()
    assert len(words) == 2
    assert words[0]["en"] == "hive"
<<<END_CODE>>>
<<<RELAY_NOTE>>>
Fixed assertion in tests/test_app.py to expect the 2 vocabulary items returned by get_words().
<<<END_RELAY_NOTE>>>
"""

    # 7. InspectorAgent approves repaired test
    inspector_approve_3 = '{"approved": true, "issues": [], "summary": "Repaired test looks great."}'

    scripted_llm = ScriptedLLMClient(
        cfg,
        [
            architect_reply,
            coder_reply,
            inspector_approve,
            tester_buggy_reply,
            inspector_approve_2,
            medic_repair_reply,
            inspector_approve_3,
        ],
    )

    events: list[str] = []
    orchestrator = HiveOrchestrator(
        config=cfg,
        llm_client=scripted_llm,
        on_event=lambda kind, _msg, _data: events.append(kind),
        sleeper=lambda _s: None,
    )

    state = orchestrator.run_task(
        goal="Create a vocabulary module app.py with get_words() and its pytest suite",
        workspace=workspace,
    )

    assert state.phase == Phase.COMPLETED
    assert state.counts()["done"] == 2
    assert state.metrics.medic_runs == 1
    assert state.metrics.medic_successes == 1
    assert state.metrics.validation_failures == 1
    assert "infirmary" in events
    assert "infirmary_healed" in events
    assert "task_completed" in events

    # Verify Blackboard Relay Note was injected into TesterAgent's prompt
    tester_prompt = scripted_llm.prompts_seen[3][1]
    assert "def get_words() -> list[dict]" in tester_prompt
    assert "Implemented get_words() -> list[dict] inside app.py" in tester_prompt

    # Verify garbage collector left no caches in workspace
    assert not (workspace / ".hive_tmp").exists()
    assert not (workspace / ".pytest_cache").exists()
    assert not (workspace / "__pycache__").exists()


def test_tui_slash_commands_and_resume_workflow(tmp_path: Path) -> None:
    """Verify TerminalUI slash commands and orchestrator.resume_task() recovery."""
    from rich.console import Console
    from ui.terminal_ui import TerminalUI

    workspace = tmp_path / "resume_ws"
    memory_dir = tmp_path / "memory"
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["memory_dir"] = str(memory_dir)
    cfg["default_workspace"] = str(workspace)
    cfg["_path"] = str(tmp_path / "config.json")
    cfg["agents"]["llm_inspector"] = False

    tester_reply = """\
<<<FILE_PATH>>>
tests/test_math.py
<<<END_FILE_PATH>>>
<<<CODE>>>
from math_mod import double


def test_double() -> None:
    assert double(4) == 8
<<<END_CODE>>>
<<<RELAY_NOTE>>>
Added pytest suite for double(x) -> int in tests/test_math.py.
<<<END_RELAY_NOTE>>>
"""
    scripted_llm = ScriptedLLMClient(cfg, [tester_reply])
    console = Console(record=True)
    orchestrator = HiveOrchestrator(config=cfg, llm_client=scripted_llm, console=console, sleeper=lambda _s: None)
    ui = TerminalUI(orchestrator=orchestrator, workspace=workspace, console=console)

    assert ui.dispatch_command("/help") is True
    assert ui.dispatch_command("/thermal balanced 4") is True
    assert orchestrator.thermal.profile.name == "balanced"
    assert orchestrator.thermal.profile.cooldown_seconds == 240
    assert ui.dispatch_command("/provider ollama") is True
    assert ui.dispatch_command("/model qwen3:8b") is True
    assert ui.dispatch_command("/status") is True
    assert ui.dispatch_command("/blackboard") is True
    assert ui.dispatch_command("/clean") is True
    assert ui.dispatch_command("/exit") is False

    # Prepare an interrupted task state with step-1 done and step-2 pending
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "math_mod.py").write_text("def double(x: int) -> int:\n    return x * 2\n", encoding="utf-8")
    orchestrator.blackboard.bind_workspace(workspace, reset_if_changed=False)
    orchestrator.blackboard.update_file_record("math_mod.py", agent_note="Implemented double(x: int) -> int.")

    sm = orchestrator.state_machine
    sm.new_task("Implement math_mod and tests", workspace)
    sm.transition(Phase.PLANNING)
    from core.state_machine import MicroTask

    sm.set_plan(
        [
            MicroTask(id="step-1", title="Write math_mod.py", target_file="math_mod.py", kind="code"),
            MicroTask(id="step-2", title="Write tests", target_file="tests/test_math.py", kind="test"),
        ]
    )
    sm.transition(Phase.CODING)
    sm.start_micro_task("step-1")
    sm.record_files("step-1", ["math_mod.py"], "Implemented double(x: int) -> int.")
    sm.transition(Phase.INSPECTING)
    sm.mark_inspected("step-1", True, "OK")
    sm.transition(Phase.VALIDATING)
    sm.mark_validated("step-1", True, "OK")
    sm.complete_micro_task("step-1")
    sm.pause("Simulated user interrupt before step-2")

    resumed_state = orchestrator.resume_task()
    assert resumed_state.phase == Phase.COMPLETED
    assert resumed_state.counts()["done"] == 2

