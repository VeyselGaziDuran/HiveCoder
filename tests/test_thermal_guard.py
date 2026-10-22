"""Unit tests for core/thermal_guard.py: profiles, cooldown calculation, and Ollama RAM eviction."""

from __future__ import annotations

from typing import Any

from core.llm_client import unload_ollama_model
from core.thermal_guard import ThermalGuard, format_mmss


class FakeClock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakeHTTPSession:
    def __init__(self, status_code: int = 200) -> None:
        self.status_code = status_code
        self.calls: list[tuple[str, dict[str, Any], float]] = []

    def post(self, url: str, json: dict[str, Any], timeout: float = 15.0) -> Any:
        self.calls.append((url, json, timeout))

        class _Resp:
            def __init__(self, code: int) -> None:
                self.status_code = code

        return _Resp(self.status_code)


def test_profile_switching_and_custom_minutes() -> None:
    clock = FakeClock()
    guard = ThermalGuard(settings={"profile": "eco"}, sleeper=clock.sleep, clock=clock)
    assert guard.profile.name == "eco"
    assert guard.profile.cooldown_seconds == 300
    assert guard.profile.micro_pause_seconds == 10

    prof = guard.set_profile("eco", cooldown_minutes=10)
    assert prof.cooldown_seconds == 600

    turbo = guard.set_profile("turbo")
    assert turbo.micro_pause_seconds == 1
    assert not turbo.deep_cooldown_enabled


def test_cooldown_triggers_and_unloads_model() -> None:
    clock = FakeClock()
    unloaded: list[bool] = []

    def unloader() -> bool:
        unloaded.append(True)
        return True

    guard = ThermalGuard(
        settings={"profile": "eco", "unload_model_on_cooldown": True},
        unloader=unloader,
        sleeper=clock.sleep,
        clock=clock,
        show_progress=False,
    )

    guard.record_task(15.0)
    guard.record_task(20.0)
    assert not guard.needs_cooldown()
    guard.record_task(25.0)
    assert guard.needs_cooldown()

    waited = guard.deep_cooldown()
    assert waited == 300.0
    assert len(unloaded) == 1
    assert guard.last_unload_ok is True
    assert guard.tasks_since_cooldown == 0
    assert guard.total_cooldowns == 1
    assert not guard.needs_cooldown()


def test_continuous_work_duration_triggers_cooldown_and_state_resume() -> None:
    clock = FakeClock()
    guard = ThermalGuard(
        settings={"profile": "balanced"},
        sleeper=clock.sleep,
        clock=clock,
        show_progress=False,
    )
    # Balanced triggers after 5 tasks OR 900s continuous work
    guard.record_task(950.0)
    assert guard.needs_cooldown()

    # Simulate an interrupted cooldown saved to state.json with 45s remaining
    saved = guard.export_state()
    saved["cooldown_until"] = clock() + 45.0

    guard2 = ThermalGuard(
        settings={"profile": "balanced"},
        sleeper=clock.sleep,
        clock=clock,
        show_progress=False,
    )
    guard2.import_state(saved)
    assert guard2.remaining_cooldown() == 45.0
    resumed_wait = guard2.resume_pending_cooldown()
    assert resumed_wait == 45.0
    assert guard2.remaining_cooldown() == 0.0
    assert format_mmss(299) == "04:59"


def test_unload_ollama_model_sends_keep_alive_zero() -> None:
    session = FakeHTTPSession(status_code=200)
    ok = unload_ollama_model("http://localhost:11434", "qwen3:8b", session=session)
    assert ok is True
    assert len(session.calls) == 1
    url, payload, _timeout = session.calls[0]
    assert url == "http://localhost:11434/api/generate"
    assert payload == {"model": "qwen3:8b", "keep_alive": 0}
