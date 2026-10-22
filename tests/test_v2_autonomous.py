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


def test_cycle_1_2_3_tuple_assign_jsx_apostrophe_and_deepest_patient_selection(tmp_path: Path) -> None:
    """Verify tuple unpacking preservation, __main__ exclusion, JSX apostrophe balance, and deepest patient selection."""
    from agents.medic_agent import MedicAgent
    from core.context_builder import ContextBuilder
    from core.state_machine import MicroTask
    from core.validator import Validator

    ws = tmp_path / "cycle_ws"
    ws.mkdir()
    bb = Blackboard(tmp_path / "cycle_mem", workspace=ws)

    # 1. Tuple unpacking & class attribute preservation + __main__ exclusion
    old_code = (
        "MIN_BOUND, MAX_BOUND = 1, 99\n\n"
        "try:\n"
        "    def fallback_fn() -> int:\n"
        "        return MIN_BOUND\n"
        "except Exception:\n"
        "    def fallback_fn() -> int:\n"
        "        return 0\n\n"
        "class ConfigBox:\n"
        "    PREFIX: str = 'cfg_'\n"
        "    def label(self, name: str) -> str:\n"
        "        return self.PREFIX + name\n\n"
        'if __name__ == "__main__":\n'
        "    scratch_only_var = 42\n"
    )
    (ws / "box.py").write_text(old_code, encoding="utf-8")
    rec = bb.update_file_record("box.py", agent_note="Initial box", task_id="step-1")
    bb.lock_verified_symbols(["box.py"])

    # __main__ variable must NOT be indexed as an exportable module global
    from core.blackboard import defined_names

    exported = defined_names(rec)
    assert "MIN_BOUND" in exported and "MAX_BOUND" in exported
    assert "fallback_fn" in exported
    assert "scratch_only_var" not in exported

    # Incoming code re-emits ConfigBox without PREFIX or label(), and omits MIN_BOUND, MAX_BOUND
    new_code = (
        "class ConfigBox:\n"
        "    def bounds(self) -> tuple[int, int]:\n"
        "        return (MIN_BOUND, MAX_BOUND)\n"
    )
    merged = ASTPatcher.merge_preserving_symbols(
        existing_source=old_code,
        incoming_source=new_code,
        protected_symbols=bb.get_verified_symbols("box.py"),
    )
    ns: dict[str, Any] = {}
    exec(compile(merged, "<cycle_merged>", "exec"), ns)
    box = ns["ConfigBox"]()
    assert box.bounds() == (1, 99)
    assert box.label("prod") == "cfg_prod"

    # 2. JSX text with apostrophes ("User's Profile", "Don't") must pass brace balance
    jsx_path = ws / "Profile.tsx"
    jsx_path.write_text(
        "export function Profile() {\n"
        "  return <div>User's Profile — Don't worry!</div>;\n"
        "}\n",
        encoding="utf-8",
    )
    val = Validator(ws)
    assert val.check_syntax(["Profile.tsx"]).passed is True

    # 3. MedicAgent selects the deepest crashing non-test module on runtime TypeError
    (ws / "tests").mkdir()
    (ws / "tests" / "test_box.py").write_text("from box import ConfigBox\n", encoding="utf-8")
    llm = TempTrackingLLMClient(DEFAULT_CONFIG, [])
    medic = MedicAgent(llm, bb, ContextBuilder(bb, ws), ws)
    mt = MicroTask(id="step-2", title="Test box", target_file="tests/test_box.py", kind="test", files_written=["tests/test_box.py"])
    tb = (
        "tests/test_box.py:4: in test_box\n"
        "    box.label(123)\n"
        "box.py:12: in label\n"
        "    return self.PREFIX + name\n"
        "E   TypeError: can only concatenate str (not 'int') to str\n"
    )
    assert medic.select_patient_file(mt, tb) == "box.py"
    related = medic._extract_related_traceback_context("box.py", tb)
    assert "tests/test_box.py" in related


def test_cycle_4_5_import_dedup_subpackage_context_truncation_and_lifecycle(tmp_path: Path) -> None:
    """Verify ASTPatcher bound-import dedup, subpackage attribute check, per-file context truncation, and lifecycle cleanup."""
    from core.context_builder import ContextBuilder
    from core.state_machine import MicroTask

    ws = tmp_path / "cycle45_ws"
    ws.mkdir()
    bb = Blackboard(tmp_path / "cycle45_mem", workspace=ws)

    # 1. Bound import deduplication in ASTPatcher (e.g., 'import math as m' vs 'from math import sqrt as m')
    old_src = (
        "import math as m\n"
        "def old_fn(x: float) -> float:\n"
        "    return float(x)\n"
    )
    new_src = (
        "from math import sqrt as m\n"
        "def new_fn(x: float) -> float:\n"
        "    return m(x)\n"
    )
    merged = ASTPatcher.merge_preserving_symbols(old_src, new_src, protected_symbols={"old_fn"})
    # 'import math as m' must NOT be re-injected because 'm' is already bound in new_src
    assert "import math as m" not in merged
    ns: dict[str, Any] = {}
    exec(compile(merged, "<cycle4_merged>", "exec"), ns)
    assert ns["new_fn"](16.0) == 4.0
    assert ns["old_fn"](5.0) == 5.0

    # 2. Subpackage attribute access in Blackboard.check_import_consistency
    pkg_dir = ws / "pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
    (pkg_dir / "submod.py").write_text("def run_sub() -> int:\n    return 7\n", encoding="utf-8")
    (ws / "caller.py").write_text(
        "import pkg.submod\n"
        "def call_it() -> int:\n"
        "    return pkg.submod.run_sub()\n",
        encoding="utf-8",
    )
    bb.scan_workspace()
    assert bb.check_import_consistency("caller.py") == []

    # 3. ContextBuilder per-file truncation preserves headers & delimiters across multiple files
    (ws / "big_a.py").write_text("A_LINE = 1\n" * 600, encoding="utf-8")
    (ws / "big_b.py").write_text("B_LINE = 2\n" * 600, encoding="utf-8")
    ctx_builder = ContextBuilder(bb, ws, max_chars=3000)
    mt = MicroTask(id="step-x", title="Update big_a", target_file="big_a.py", kind="code")
    bundle = ctx_builder.build(mt, extra_files=["big_b.py", "big_a.py"])
    assert bundle.text.count("# CURRENT CONTENT OF big_a.py") == 1
    assert bundle.text.count("# CURRENT CONTENT OF big_b.py") == 1
    assert bundle.text.count("<<<CURRENT_FILE>>>") == 2
    assert bundle.text.count("<<<END_CURRENT_FILE>>>") == 2

    # 4. MemoryLifecycle.purge_unverified_notes removes verified_symbols for deleted files
    bb.lock_verified_symbols(["big_b.py"])
    assert "big_b.py" in bb.verified_symbols
    (ws / "big_b.py").unlink()
    lifecycle = MemoryLifecycle()
    lifecycle.purge_unverified_notes(bb, verified_task_ids=set())
    assert "big_b.py" not in bb.files
    assert "big_b.py" not in bb.verified_symbols


def test_cycle_6_7_code_toposort_init_ordering_and_git_guard_nested_memory(tmp_path: Path) -> None:
    """Verify code dependency topological sort, __init__.py ordering, leading-slash normalization, and nested memory GitGuard."""
    from agents.architect_agent import ArchitectAgent
    from core.context_builder import ContextBuilder
    from core.git_guard import GitGuard

    ws = tmp_path / "cycle67_ws"
    ws.mkdir()
    mem = ws / "memory"
    bb = Blackboard(mem, workspace=ws)
    ctx = ContextBuilder(bb, ws)

    # 1. Small model emits top-down plan: pkg/__init__.py -> main.py -> engine.py -> pkg/models.py
    raw_plan = json.dumps(
        {
            "micro_tasks": [
                {"id": "s1", "title": "Export package API", "target_file": "/pkg/__init__.py", "kind": "code", "description": "Re-export models"},
                {"id": "s2", "title": "CLI entrypoint", "target_file": "/main.py", "kind": "code", "description": "Import run_engine from engine.py"},
                {"id": "s3", "title": "Core engine", "target_file": "engine.py", "kind": "code", "description": "Use Item from pkg/models.py"},
                {"id": "s4", "title": "Data models", "target_file": "pkg/models.py", "kind": "code", "description": "Define Item dataclass"},
                {"id": "s5", "title": "Unit tests", "target_file": "tests/test_main.py", "kind": "test", "description": "Test main.py"},
            ]
        }
    )
    arch = ArchitectAgent(TempTrackingLLMClient(DEFAULT_CONFIG, [raw_plan]), bb, ctx, ws)
    plan = arch.plan("Build packaged CLI")
    ordered_targets = [mt.target_file for mt in plan]
    # pkg/models.py must precede both pkg/__init__.py and engine.py, and engine.py must precede main.py!
    assert ordered_targets.index("pkg/models.py") < ordered_targets.index("pkg/__init__.py")
    assert ordered_targets.index("pkg/models.py") < ordered_targets.index("engine.py")
    assert ordered_targets.index("engine.py") < ordered_targets.index("main.py")
    assert ordered_targets[-1] == "tests/test_main.py"

    # 2. GitGuard with memory/ nested inside workspace must NOT wipe memory/state.json on rollback
    (ws / "app.py").write_text("x = 1\n", encoding="utf-8")
    state_json = mem / "state.json"
    state_json.write_text('{"task_id": "t1", "phase": "coding"}\n', encoding="utf-8")
    guard = GitGuard(workspace=ws, store_dir=mem / "snapshots", enabled=True)
    snap = guard.snapshot("before-step")
    assert snap is not None
    # Modify both app.py (should roll back) and memory/state.json (must NOT roll back)
    (ws / "app.py").write_text("x = 999\n", encoding="utf-8")
    state_json.write_text('{"task_id": "t1", "phase": "paused"}\n', encoding="utf-8")
    assert guard.rollback(snap) is True
    assert (ws / "app.py").read_text(encoding="utf-8") == "x = 1\n"
    assert '"paused"' in state_json.read_text(encoding="utf-8")


def test_cycle_8_corrupt_state_recovery_resume_all_done_and_rich_markup_escape(tmp_path: Path) -> None:
    """Verify corrupt state/history quarantine, finish() from CODING after resume, and Rich markup escaping."""
    import io
    from rich.console import Console
    from core.state_machine import MicroTask, StateMachine, TaskHistory
    from ui.terminal_ui import TerminalUI

    # 1. Corrupt state.json and task_history.json are quarantined without crashing
    state_path = tmp_path / "state.json"
    state_path.write_text("{broken_json", encoding="utf-8")
    sm = StateMachine(state_path)
    assert sm.load() is None
    assert (tmp_path / "state.corrupt.json").is_file()

    hist_path = tmp_path / "task_history.json"
    hist_path.write_text("{not_valid", encoding="utf-8")
    th = TaskHistory(hist_path)
    assert th.load() == []
    assert (tmp_path / "task_history.corrupt.json").is_file()

    # 2. Resuming a task where all micro-tasks are already verified & done finishes cleanly from CODING phase
    ws = tmp_path / "ws"
    ws.mkdir()
    sm.new_task("Complete after resume", ws)
    sm.transition(Phase.PLANNING)
    sm.set_plan([MicroTask(id="step-1", title="Done step", target_file="a.py", kind="code")])
    sm.transition(Phase.CODING)
    sm.start_micro_task("step-1")
    sm.record_files("step-1", ["a.py"], "Done a.py")
    sm.transition(Phase.INSPECTING)
    sm.mark_inspected("step-1", True, "OK")
    sm.transition(Phase.VALIDATING)
    sm.mark_validated("step-1", True, "OK")
    sm.complete_micro_task("step-1")
    # Simulate resume transitioning to CODING when no pending micro-tasks remain
    sm.transition(Phase.CODING)
    finished = sm.finish()
    assert finished.phase == Phase.COMPLETED

    # 3. TerminalUI.handle_event escapes closing bracket tags '[/]' in tracebacks without MarkupError
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False)
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["memory_dir"] = str(tmp_path / "mem")
    cfg["default_workspace"] = str(ws)
    orch = HiveOrchestrator(config=cfg, console=console)
    tui = TerminalUI(orchestrator=orch, workspace=ws, console=console)
    tui.handle_event("validation_failed", "AssertionError: regex r'[/]' did not match [bold", {})
    assert "AssertionError" in buf.getvalue()






