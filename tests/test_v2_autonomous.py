"""Unit and integration tests for HiveCoder v2 autonomous features:
- ASTPatcher surgical symbol preservation
- MemoryLifecycle orphan-note purge and post-task ephemeral flush
- ThermalSensor adaptive 5m/10m cooldown and automatic early wakeup
- RuntimeProbe, PackageRadarProbe, and SignatureProbe
- Infirmary loop-breaker temperature escalation
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Optional

from core.ast_patcher import ASTPatcher
from core.blackboard import Blackboard
from core.config import DEFAULT_CONFIG
from core.llm_client import LLMClient
from core.memory_lifecycle import MemoryLifecycle
from core.orchestrator import HiveOrchestrator
from core.state_machine import Phase
from core.thermal_guard import ThermalGuard
from core.thermal_sensor import ThermalReading, ThermalSensor
from probes.package_radar import PackageRadarProbe
from probes.runtime_probe import RuntimeProbe
from probes.signature_probe import SignatureProbe


def test_ast_patcher_preserves_verified_symbols_and_extracts_traceback_window() -> None:
    existing = (
        "import json\n\n"
        "def get_words() -> list[dict]:\n"
        "    return [{'en': 'hive', 'tr': 'kovan'}]\n"
    )
    # Small model emits only the new function and forgets get_words()
    incoming = (
        "def count_words(items: list[dict]) -> int:\n"
        "    return len(items)\n"
    )
    merged = ASTPatcher.merge_preserving_symbols(
        existing_source=existing,
        incoming_source=incoming,
        protected_symbols={"get_words"},
    )
    names = ASTPatcher.extract_top_level_names(merged)
    assert "get_words" in names
    assert "count_words" in names
    assert "import json" in merged

    window = ASTPatcher.extract_traceback_window(
        merged,
        'File "app.py", line 4, in get_words',
        radius=2,
    )
    assert ">>    4 |" in window


def test_memory_lifecycle_purges_orphan_notes_and_flushes_on_completion(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    memory_dir = tmp_path / "memory"
    workspace.mkdir()
    (workspace / "mod_a.py").write_text("def a() -> int:\n    return 1\n", encoding="utf-8")
    (workspace / "mod_b.py").write_text("def b() -> int:\n    return 2\n", encoding="utf-8")

    bb = Blackboard(memory_dir=memory_dir, workspace=workspace)
    bb.update_file_record("mod_a.py", agent_note="Verified note for step-1.", agent="CoderAgent", task_id="step-1")
    bb.update_file_record("mod_b.py", agent_note="Unfinished note for step-2.", agent="CoderAgent", task_id="step-2")
    assert len(bb.notes) == 2

    lifecycle = MemoryLifecycle()
    purged = lifecycle.purge_unverified_notes(bb, verified_task_ids=["step-1"])
    assert purged == 1
    assert len(bb.notes) == 1
    assert bb.notes[0]["task_id"] == "step-1"
    assert "Unfinished note" not in bb.notes_path.read_text(encoding="utf-8")

    cleared = lifecycle.clear_all_relay_notes(bb)
    assert cleared == 1
    assert len(bb.notes) == 0


def test_adaptive_thermal_sensor_zero_pause_when_cool_and_auto_early_wakeup() -> None:
    readings = [
        # During 4 tasks: machine is cool -> no cooldown
        ThermalReading(level="nominal", temp_celsius=52.0, reason="cool"),
        ThermalReading(level="nominal", temp_celsius=54.0, reason="cool"),
        ThermalReading(level="nominal", temp_celsius=56.0, reason="cool"),
        ThermalReading(level="nominal", temp_celsius=58.0, reason="cool"),
        # 5th check: machine heats up to heavy (84C -> 600s max cooldown)
        ThermalReading(
            level="heavy",
            temp_celsius=84.0,
            recommended_cooldown_seconds=600.0,
            reason="temperature 84.0C >= 82.0C",
        ),
        # Inside cooldown loop: tick 1 still warm, tick 2 cooled back to nominal!
        ThermalReading(
            level="moderate",
            temp_celsius=74.0,
            recommended_cooldown_seconds=300.0,
            reason="cooling",
        ),
        ThermalReading(level="nominal", temp_celsius=55.0, reason="cooled"),
    ]
    idx = 0

    def fake_probe() -> ThermalReading:
        nonlocal idx
        r = readings[min(idx, len(readings) - 1)]
        idx += 1
        return r

    now = 1000.0
    sleeps: list[float] = []

    def clock() -> float:
        return now

    def sleeper(sec: float) -> None:
        nonlocal now
        sleeps.append(sec)
        now += sec

    unloaded: list[bool] = []
    sensor = ThermalSensor(probe_fn=fake_probe)
    guard = ThermalGuard(
        settings={"profile": "adaptive", "early_wakeup": True, "unload_model_on_cooldown": True},
        unloader=lambda: (unloaded.append(True) or True),
        sleeper=sleeper,
        clock=clock,
        show_progress=False,
        sensor=sensor,
    )

    for _ in range(4):
        guard.record_task(30.0, response_chars=1200)
        assert guard.needs_cooldown() is False

    # 5th check sees 84C -> triggers cooldown with 600s cap, but wakes up after 2s when cooled!
    assert guard.needs_cooldown() is True
    assert guard.recommended_cooldown_seconds() == 600.0
    waited = guard.deep_cooldown()
    assert waited < 10.0
    assert len(unloaded) == 1
    assert guard.early_wakeups == 1


def test_probes_runtime_package_radar_and_signature_probe() -> None:
    rt = RuntimeProbe().inspect()
    md = rt.to_markdown()
    assert "Python" in md
    assert "pydantic" in rt.installed_packages

    class OfflineSession:
        def get(self, url: str, timeout: float = 2.0) -> Any:
            raise OSError("offline")

    radar = PackageRadarProbe(online_enabled=True, session=OfflineSession())  # type: ignore[arg-type]
    pkgs = radar.scan_goal_packages("Build a service using pydantic and pytest")
    names = {p.name for p in pkgs}
    assert "pydantic" in names
    assert "pytest" in names

    sig_probe = SignatureProbe()
    sigs = sig_probe.inspect_imports_in_source("from json import dumps\nimport pathlib\n")
    assert any("json.dumps" in s for s in sigs)


class TempTrackingLLMClient(LLMClient):
    """Records temperatures passed to complete() across scripted responses."""

    def __init__(self, config: dict[str, Any], replies: list[str]) -> None:
        super().__init__(config)
        self.replies = list(replies)
        self.temperatures: list[Optional[float]] = []

    def complete(self, system: str, user: str, temperature: Optional[float] = None) -> str:
        self.temperatures.append(temperature)
        self.stats.calls += 1
        return self.replies.pop(0)


def test_end_to_end_v2_surgical_merge_loop_breaker_and_zero_residue_cleanup(tmp_path: Path) -> None:
    """Verify:
    1. Multi-step edits to the same file preserve earlier verified functions via ASTPatcher.
    2. Infirmary escalates temperature on attempt 2 (loop-breaker).
    3. Post-task completion purges all relay notes, state.json, and snapshot repo.
    """
    workspace = tmp_path / "clean_app"
    memory_dir = tmp_path / "memory"

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["memory_dir"] = str(memory_dir)
    cfg["default_workspace"] = str(workspace)
    cfg["agents"]["llm_inspector"] = False
    cfg["probes"]["online_package_radar"] = False
    cfg["thermal"]["profile"] = "adaptive"

    plan_json = json.dumps(
        {
            "micro_tasks": [
                {
                    "id": "step-1",
                    "title": "Create store.py with init_store()",
                    "target_file": "store.py",
                    "kind": "code",
                    "description": "Define init_store() -> dict.",
                },
                {
                    "id": "step-2",
                    "title": "Add put_item(store, key, val) to store.py",
                    "target_file": "store.py",
                    "kind": "code",
                    "description": "Add put_item(store: dict, key: str, val: str) -> dict.",
                },
                {
                    "id": "step-3",
                    "title": "Write tests/test_store.py",
                    "target_file": "tests/test_store.py",
                    "kind": "test",
                    "description": "Test both init_store and put_item.",
                },
            ]
        }
    )

    step1_reply = """\
<<<FILE_PATH>>>
store.py
<<<END_FILE_PATH>>>
<<<CODE>>>
def init_store() -> dict:
    return {}
<<<END_CODE>>>
<<<RELAY_NOTE>>>
Defined init_store() -> dict in store.py.
<<<END_RELAY_NOTE>>>
"""

    # In step-2 the small model only emits put_item() and forgets init_store()!
    # ASTPatcher must automatically preserve init_store() from step-1.
    step2_partial_reply = """\
<<<FILE_PATH>>>
store.py
<<<END_FILE_PATH>>>
<<<CODE>>>
def put_item(store: dict, key: str, val: str) -> dict:
    store[key] = val
    return store
<<<END_CODE>>>
<<<RELAY_NOTE>>>
Added put_item(store, key, val) -> dict to store.py.
<<<END_RELAY_NOTE>>>
"""

    # In step-3 TesterAgent writes a broken assertion first
    step3_buggy_test = """\
<<<FILE_PATH>>>
tests/test_store.py
<<<END_FILE_PATH>>>
<<<CODE>>>
from store import init_store, put_item


def test_store_roundtrip() -> None:
    st = init_store()
    put_item(st, "k", "v")
    assert st["k"] == "wrong"
<<<END_CODE>>>
<<<RELAY_NOTE>>>
Created tests/test_store.py.
<<<END_RELAY_NOTE>>>
"""

    # Medic attempt 1 repeats the exact same buggy file (triggers loop-breaker!)
    medic_attempt_1_same = step3_buggy_test

    # Medic attempt 2 (with escalated temperature) fixes the assertion
    medic_attempt_2_fixed = """\
<<<FILE_PATH>>>
tests/test_store.py
<<<END_FILE_PATH>>>
<<<CODE>>>
from store import init_store, put_item


def test_store_roundtrip() -> None:
    st = init_store()
    put_item(st, "k", "v")
    assert st["k"] == "v"
<<<END_CODE>>>
<<<RELAY_NOTE>>>
Fixed assertion in tests/test_store.py.
<<<END_RELAY_NOTE>>>
"""

    llm = TempTrackingLLMClient(
        cfg,
        [
            plan_json,
            step1_reply,
            step2_partial_reply,
            step3_buggy_test,
            medic_attempt_1_same,
            medic_attempt_2_fixed,
        ],
    )

    events: list[str] = []
    sensor = ThermalSensor(probe_fn=lambda: ThermalReading(level="nominal", temp_celsius=50.0))
    orchestrator = HiveOrchestrator(
        config=cfg,
        llm_client=llm,
        on_event=lambda kind, _msg, _data: events.append(kind),
        sleeper=lambda _s: None,
        thermal_sensor=sensor,
    )

    state = orchestrator.run_task("Build key-value store module and tests", workspace=workspace)
    assert state.phase == Phase.COMPLETED
    assert "loop_breaker" in events

    # Verify Medic attempt 2 used an escalated temperature (> 0.2)
    assert llm.temperatures[-1] is not None and llm.temperatures[-1] > 0.2

    # Verify both init_store and put_item exist in store.py thanks to ASTPatcher
    store_code = (workspace / "store.py").read_text(encoding="utf-8")
    assert "def init_store()" in store_code
    assert "def put_item(" in store_code

    # Verify zero-residue cleanup: relay notes, state.json, and workspace caches are gone
    assert len(orchestrator.blackboard.notes) == 0
    assert not (memory_dir / "state.json").exists()
    assert not (workspace / ".hive_tmp").exists()
    assert not (workspace / "__pycache__").exists()


def test_ast_patcher_future_imports_and_class_level_methods() -> None:
    """Verify ASTPatcher keeps __future__ imports at line 1 and restores omitted class methods."""
    existing = (
        '"""User service module."""\n'
        "from __future__ import annotations\n"
        "import hashlib\n\n"
        "class UserService:\n"
        "    def __init__(self) -> None:\n"
        "        self.users: dict[str, str] = {}\n\n"
        "    def register(self, username: str, password: str) -> str:\n"
        "        digest = hashlib.sha256(password.encode()).hexdigest()\n"
        "        self.users[username] = digest\n"
        "        return digest\n\n"
        "def helper_version() -> str:\n"
        "    return '1.0'\n"
    )
    # Small model re-emits UserService with ONLY authenticate() and forgets __init__, register, and helper_version
    incoming = (
        '"""User service module."""\n'
        "from __future__ import annotations\n\n"
        "class UserService:\n"
        "    def authenticate(self, username: str, digest: str) -> bool:\n"
        "        return self.users.get(username) == digest\n\n"
        'if __name__ == "__main__":\n'
        "    svc = UserService()\n"
    )

    merged = ASTPatcher.merge_preserving_symbols(
        existing_source=existing,
        incoming_source=incoming,
        protected_symbols={"UserService", "UserService.__init__", "UserService.register", "helper_version"},
    )

    # Must compile cleanly without SyntaxError (from __future__ stays at top)
    code_obj = compile(merged, "<merged>", "exec")
    ns: dict[str, Any] = {}
    exec(code_obj, ns)
    svc = ns["UserService"]()
    token = svc.register("veysel", "secret")
    assert svc.authenticate("veysel", token) is True
    assert ns["helper_version"]() == "1.0"


def test_architect_retry_topological_sort_and_auto_test_injection(tmp_path: Path) -> None:
    """Verify ArchitectAgent retries invalid JSON, orders code before tests, and auto-injects missing tests."""
    from agents.architect_agent import ArchitectAgent
    from core.context_builder import ContextBuilder

    ws = tmp_path / "ws"
    ws.mkdir()
    bb = Blackboard(tmp_path / "mem", workspace=ws)
    ctx = ContextBuilder(bb, ws)

    # Reply 1 is broken non-JSON; Reply 2 puts a test before code; Reply 3 has code without tests
    reply_broken = "Sorry, I forgot the JSON format!"
    reply_reordered = json.dumps(
        {
            "micro_tasks": [
                {"id": "step-1", "title": "Write tests first", "target_file": "tests/test_core.py", "kind": "test"},
                {"id": "step-2", "title": "Write core.py", "target_file": "core.py", "kind": "code"},
            ]
        }
    )
    reply_no_test = json.dumps(
        {
            "micro_tasks": [
                {"id": "step-1", "title": "Implement engine.py", "target_file": "engine.py", "kind": "code"},
            ]
        }
    )

    llm = TempTrackingLLMClient(DEFAULT_CONFIG, [reply_broken, reply_reordered, reply_no_test])
    arch = ArchitectAgent(llm, bb, ctx, ws)

    plan1 = arch.plan("Build core module", max_retries=2)
    assert len(plan1) == 2
    assert plan1[0].kind == "code" and plan1[0].target_file == "core.py"
    assert plan1[1].kind == "test" and plan1[1].target_file == "tests/test_core.py"

    plan2 = arch.plan("Build engine module", max_retries=1)
    assert len(plan2) == 2
    assert plan2[0].kind == "code" and plan2[0].target_file == "engine.py"
    assert plan2[1].kind == "test" and plan2[1].target_file == "tests/test_engine.py"


def test_multilang_blackboard_indexing_and_brace_validator(tmp_path: Path) -> None:
    """Verify Blackboard indexes TypeScript/Dart symbols and Validator catches unclosed braces."""
    from core.validator import Validator

    ws = tmp_path / "mobile_app"
    ws.mkdir()
    bb = Blackboard(tmp_path / "mem", workspace=ws)

    tsx_file = ws / "App.tsx"
    tsx_file.write_text(
        "import React from 'react';\n"
        "export interface AppProps {\n"
        "  title: string;\n"
        "}\n"
        "export const HomeScreen = (props: AppProps) => {\n"
        "  return null;\n"
        "};\n",
        encoding="utf-8",
    )
    dart_file = ws / "main.dart"
    dart_file.write_text(
        "import 'package:flutter/material.dart';\n"
        "class CounterWidget extends StatelessWidget {\n"
        "  Widget build(BuildContext context) {\n"
        "    return Container();\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )

    bb.scan_workspace()
    compact = bb.get_compact_memory(target_files=["App.tsx", "main.dart"])
    assert "interface AppProps" in compact
    assert "const HomeScreen" in compact
    assert "class CounterWidget extends StatelessWidget" in compact
    assert "fn build(BuildContext context)" in compact

    validator = Validator(ws)
    assert validator.check_syntax(["App.tsx", "main.dart"]).passed is True

    # Introduce an unclosed brace in main.dart
    dart_file.write_text("class BrokenWidget {\n  void build() {\n", encoding="utf-8")
    res = validator.check_syntax(["main.dart"])
    assert res.passed is False
    assert "unclosed '{'" in res.output


def test_target_path_alignment_missing_pkg_alert_and_inspector_override(tmp_path: Path) -> None:
    """Verify subfolder path alignment, missing-package alert in MedicAgent, and false-rejection override."""
    ws = tmp_path / "app_ws"
    mem = tmp_path / "mem"
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["memory_dir"] = str(mem)
    cfg["default_workspace"] = str(ws)
    cfg["agents"]["llm_inspector"] = True
    cfg["thermal"]["profiles"]["balanced"]["micro_pause_seconds"] = 0

    plan_json = json.dumps(
        {
            "micro_tasks": [
                {"id": "step-1", "title": "Create src/calc.py", "target_file": "src/calc.py", "kind": "code"},
                {"id": "step-2", "title": "Create tests/test_calc.py", "target_file": "tests/test_calc.py", "kind": "test"},
            ]
        }
    )
    # Small model omits 'src/' prefix and writes 'calc.py' instead of 'src/calc.py'
    coder_reply = """\
<<<FILE_PATH>>>
calc.py
<<<END_FILE_PATH>>>
<<<CODE>>>
def add(a: int, b: int) -> int:
    return a + b
<<<END_CODE>>>
<<<RELAY_NOTE>>>
Implemented add(a: int, b: int) -> int in src/calc.py.
<<<END_RELAY_NOTE>>>
"""
    inspector_approve_1 = '{"approved": true, "issues": [], "summary": "OK"}'

    tester_reply = """\
<<<FILE_PATH>>>
tests/test_calc.py
<<<END_FILE_PATH>>>
<<<CODE>>>
from src.calc import add


def test_add() -> None:
    assert add(2, 3) == 5
<<<END_CODE>>>
<<<RELAY_NOTE>>>
Added test_add() in tests/test_calc.py.
<<<END_RELAY_NOTE>>>
"""
    # Inspector LLM hallucinated a false rejection even though AST, imports, and pytest all pass!
    inspector_false_reject_2 = '{"approved": false, "issues": ["Subjective style nitpick"], "summary": "Rejected"}'

    llm = TempTrackingLLMClient(
        cfg,
        [plan_json, coder_reply, inspector_approve_1, tester_reply, inspector_false_reject_2],
    )
    events: list[str] = []
    sensor = ThermalSensor(probe_fn=lambda: ThermalReading(level="nominal", temp_celsius=48.0))
    orchestrator = HiveOrchestrator(
        config=cfg,
        llm_client=llm,
        on_event=lambda kind, _msg, _data: events.append(kind),
        sleeper=lambda _s: None,
        thermal_sensor=sensor,
    )

    state = orchestrator.run_task("Create src/calc.py and tests", workspace=ws)
    assert state.phase == Phase.COMPLETED
    assert (ws / "src" / "calc.py").is_file()
    assert "inspection_override" in events

    # Also verify MedicAgent._detect_missing_package_alert directly
    from agents.medic_agent import MedicAgent
    from core.context_builder import ContextBuilder

    medic = MedicAgent(llm, orchestrator.blackboard, ContextBuilder(orchestrator.blackboard, ws), ws)
    alert = medic._detect_missing_package_alert("ModuleNotFoundError: No module named 'nonexistent_super_pkg_999'")
    assert "nonexistent_super_pkg_999" in alert
    assert "NOT installed" in alert


def test_deep_audit_base_class_ordering_relative_imports_and_stdlib_probe(tmp_path: Path) -> None:
    """Verify preserved base classes precede subclasses, root relative imports normalize, and stdlib probe works."""
    from core.llm_client import parse_nano_tags

    existing = (
        "from __future__ import annotations\n"
        "DEFAULT_MULTIPLIER = 10\n\n"
        "def _scale(x: int) -> int:\n"
        "    return x * DEFAULT_MULTIPLIER\n\n"
        "class BaseProcessor:\n"
        "    def run_base(self, val: int) -> int:\n"
        "        return _scale(val)\n"
    )
    # Incoming code subclasses BaseProcessor and uses DEFAULT_MULTIPLIER without re-defining them!
    incoming = (
        "from __future__ import annotations\n"
        "import math\n\n"
        "class AdvancedProcessor(BaseProcessor):\n"
        "    def compute(self, val: int = DEFAULT_MULTIPLIER) -> int:\n"
        "        return self.run_base(val) + int(math.sqrt(4))\n"
    )
    ws = tmp_path / "audit_ws"
    ws.mkdir()
    bb = Blackboard(tmp_path / "audit_mem", workspace=ws)
    (ws / "proc.py").write_text(existing, encoding="utf-8")
    bb.update_file_record("proc.py", agent_note="Base processor", task_id="step-1")
    bb.lock_verified_symbols(["proc.py"])

    protected = bb.get_verified_symbols("proc.py")
    assert "_scale" in protected
    assert "BaseProcessor" in protected
    assert "DEFAULT_MULTIPLIER" in protected

    merged = ASTPatcher.merge_preserving_symbols(existing, incoming, protected_symbols=protected)
    ns: dict[str, Any] = {}
    exec(compile(merged, "<audit_merged>", "exec"), ns)
    inst = ns["AdvancedProcessor"]()
    assert inst.compute(3) == 32

    # Root-level relative import normalization
    norm_code = ASTPatcher.normalize_root_relative_imports("from .utils import helper\n", "app.py")
    assert norm_code == "from utils import helper\n"

    # Markdown fence fallback for .tsx / .dart
    fenced_tsx = "Explanation:\n```\nnpm install\n```\n<<<FILE_PATH>>>\nApp.tsx\n<<<END_FILE_PATH>>>\n```tsx\nexport const App = () => null;\n```"
    parsed = parse_nano_tags(fenced_tsx)
    assert parsed.has_code is True
    assert "export const App" in parsed.files[0].code

    # SignatureProbe stdlib traceback inspection
    sp = SignatureProbe()
    sigs = sp.inspect_traceback(
        "import tempfile\n",
        "ImportError: cannot import name 'wrong_fn' from 'urllib.parse'",
    )
    assert any("urllib.parse." in s for s in sigs)

    # ThermalSensor ignores short TTFT-bound responses (< 200 chars)
    ts = ThermalSensor()
    ts.record_throughput(elapsed_seconds=1.0, response_chars=1000)
    assert ts.baseline_chars_per_sec == 1000.0
    ts.record_throughput(elapsed_seconds=2.0, response_chars=50)
    assert ts.latest_chars_per_sec == 1000.0


