"""Thermal Guard: adaptive hardware-aware and profile-based cooldown controller.

Supports two complementary modes:
1. **Sensor-driven Adaptive Mode (`adaptive` profile or attached :class:`ThermalSensor`)**:
   - Works continuously without artificial stops while the device is cool (`nominal`).
   - Triggers a 5-minute (`300s`) cooldown on `moderate` heat or throughput throttling,
     and a 10-minute (`600s`) cooldown on `heavy` heat.
   - Evicts the active Ollama model from RAM/VRAM (`keep_alive: 0`) at cooldown start.
   - Polls the sensor during cooldown and **wakes up immediately** as soon as the
     device returns to `nominal` temperature — no user prompt or wasted waiting.
2. **Deterministic Profile Mode (`eco`, `balanced`, `turbo`)**:
   - Task-count and continuous-work-duration thresholds with full state persistence.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from core.thermal_sensor import ThermalReading, ThermalSensor

DEFAULT_PROFILES: dict[str, dict[str, float]] = {
    "adaptive": {
        "micro_pause_seconds": 1,
        "cooldown_every_tasks": 0,
        "cooldown_seconds": 300,
        "max_continuous_work_seconds": 0,
    },
    "eco": {
        "micro_pause_seconds": 10,
        "cooldown_every_tasks": 3,
        "cooldown_seconds": 300,
        "max_continuous_work_seconds": 480,
    },
    "balanced": {
        "micro_pause_seconds": 5,
        "cooldown_every_tasks": 5,
        "cooldown_seconds": 180,
        "max_continuous_work_seconds": 900,
    },
    "turbo": {
        "micro_pause_seconds": 1,
        "cooldown_every_tasks": 0,
        "cooldown_seconds": 0,
        "max_continuous_work_seconds": 0,
    },
}
MAX_COOLDOWN_MINUTES = 60


@dataclass(frozen=True)
class ThermalProfile:
    name: str
    micro_pause_seconds: float
    cooldown_every_tasks: int
    cooldown_seconds: float
    max_continuous_work_seconds: float

    @property
    def deep_cooldown_enabled(self) -> bool:
        if self.name == "adaptive":
            return self.cooldown_seconds > 0
        return self.cooldown_seconds > 0 and (
            self.cooldown_every_tasks > 0 or self.max_continuous_work_seconds > 0
        )


def format_mmss(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    return f"{total // 60:02d}:{total % 60:02d}"


class ThermalGuard:
    """Decides when to breathe or cool down and executes the cooldown."""

    def __init__(
        self,
        settings: dict[str, Any],
        unloader: Optional[Callable[[], bool]] = None,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
        console: Optional[Console] = None,
        show_progress: bool = True,
        on_state_change: Optional[Callable[[dict[str, Any]], None]] = None,
        on_event: Optional[Callable[[str, str], None]] = None,
        sensor: Optional[ThermalSensor] = None,
    ) -> None:
        self.settings = settings
        self.settings.setdefault("profile", "eco")
        self.settings.setdefault("early_wakeup", True)
        profiles = self.settings.setdefault("profiles", {})
        for name, values in DEFAULT_PROFILES.items():
            merged = dict(values)
            merged.update(profiles.get(name, {}))
            profiles[name] = merged

        self.unloader = unloader
        self.sleeper = sleeper
        self.clock = clock
        self.console = console
        self.show_progress = show_progress
        self.on_state_change = on_state_change
        self.on_event = on_event
        self.sensor = sensor

        self.tasks_since_cooldown = 0
        self.work_seconds_since_cooldown = 0.0
        self.cooldown_until: Optional[float] = None
        self.total_cooldowns = 0
        self.early_wakeups = 0
        self.last_unload_ok: Optional[bool] = None
        self.last_reading: Optional[ThermalReading] = None

    @property
    def profile(self) -> ThermalProfile:
        name = str(self.settings.get("profile", "eco"))
        values = self.settings["profiles"].get(name)
        if values is None:
            raise ValueError(f"Unknown thermal profile: {name}")
        return ThermalProfile(
            name=name,
            micro_pause_seconds=float(values.get("micro_pause_seconds", 0)),
            cooldown_every_tasks=int(values.get("cooldown_every_tasks", 0)),
            cooldown_seconds=float(values.get("cooldown_seconds", 0)),
            max_continuous_work_seconds=float(values.get("max_continuous_work_seconds", 0)),
        )

    def set_profile(self, name: str, cooldown_minutes: Optional[float] = None) -> ThermalProfile:
        """Switch profile; optionally override its deep-cooldown duration in minutes."""
        name = name.strip().lower()
        if name not in self.settings["profiles"]:
            raise ValueError(f"Unknown thermal profile '{name}'. Choose: {', '.join(self.settings['profiles'])}")
        self.settings["profile"] = name
        if cooldown_minutes is not None:
            minutes = float(cooldown_minutes)
            if minutes < 0 or minutes > MAX_COOLDOWN_MINUTES:
                raise ValueError(f"Cooldown minutes must be between 0 and {MAX_COOLDOWN_MINUTES}")
            profile_values = self.settings["profiles"][name]
            profile_values["cooldown_seconds"] = int(minutes * 60)
            if minutes > 0 and name != "adaptive" and int(profile_values.get("cooldown_every_tasks", 0)) <= 0:
                profile_values["cooldown_every_tasks"] = 3
        self._notify()
        return self.profile

    def record_task(self, work_seconds: float, response_chars: int = 0) -> None:
        self.tasks_since_cooldown += 1
        self.work_seconds_since_cooldown += max(0.0, float(work_seconds))
        if self.sensor is not None and response_chars > 0:
            self.sensor.record_throughput(work_seconds, response_chars)
        self._notify()

    def needs_cooldown(self) -> bool:
        prof = self.profile
        if prof.name == "turbo":
            return False

        if self.sensor is not None:
            self.last_reading = self.sensor.read()
            if self.last_reading.is_hot:
                return True
            if prof.name == "adaptive":
                return False

        if not prof.deep_cooldown_enabled:
            return False
        if prof.cooldown_every_tasks > 0 and self.tasks_since_cooldown >= prof.cooldown_every_tasks:
            return True
        if (
            prof.max_continuous_work_seconds > 0
            and self.work_seconds_since_cooldown >= prof.max_continuous_work_seconds
        ):
            return True
        return False

    def recommended_cooldown_seconds(self) -> float:
        """Determine cooldown duration (5 min moderate / 10 min heavy from sensor, or profile default)."""
        if self.last_reading is not None and self.last_reading.is_hot:
            if self.last_reading.recommended_cooldown_seconds > 0:
                return float(self.last_reading.recommended_cooldown_seconds)
        return float(self.profile.cooldown_seconds)

    def remaining_cooldown(self) -> float:
        if self.cooldown_until is None:
            return 0.0
        return max(0.0, self.cooldown_until - self.clock())

    def micro_pause(self) -> float:
        seconds = self.profile.micro_pause_seconds
        if seconds > 0:
            self._event("thermal", f"micro-pause {seconds:.0f}s")
            self.sleeper(seconds)
        return seconds

    def deep_cooldown(self, duration: Optional[float] = None) -> float:
        """Unload the model and rest until cooled or timer expires. Returns seconds waited."""
        seconds = float(self.recommended_cooldown_seconds() if duration is None else duration)
        if seconds <= 0:
            self._reset_counters()
            return 0.0
        if self.unloader is not None and self.settings.get("unload_model_on_cooldown", True):
            try:
                self.last_unload_ok = bool(self.unloader())
            except Exception:  # noqa: BLE001
                self.last_unload_ok = False
            self._event(
                "thermal",
                "model evicted from RAM/VRAM (keep_alive=0)" if self.last_unload_ok else "model unload skipped/failed",
            )
        reason = f" ({self.last_reading.reason})" if (self.last_reading and self.last_reading.is_hot) else ""
        self.cooldown_until = self.clock() + seconds
        self._notify()
        self._event("thermal", f"deep cooldown max {format_mmss(seconds)}{reason}")
        waited = self._countdown()
        self.total_cooldowns += 1
        if self.sensor is not None:
            self.sensor.reset_throughput_after_cooldown()
        self.last_reading = None
        self._reset_counters()
        return waited

    def resume_pending_cooldown(self) -> float:
        """Finish a cooldown that was interrupted (crash / Ctrl+C / reboot)."""
        remaining = self.remaining_cooldown()
        if remaining <= 0:
            if self.cooldown_until is not None:
                self.cooldown_until = None
                self._notify()
            return 0.0
        if not self.profile.deep_cooldown_enabled:
            self._event("thermal", f"pending cooldown skipped ({self.profile.name} profile)")
            self._reset_counters()
            return 0.0
        max_cap = max(self.profile.cooldown_seconds, 600.0)
        remaining = min(remaining, max_cap)
        self._event("thermal", f"resuming interrupted cooldown {format_mmss(remaining)}")
        self.cooldown_until = self.clock() + remaining
        self._notify()
        waited = self._countdown()
        self.total_cooldowns += 1
        self._reset_counters()
        return waited

    def _countdown(self) -> float:
        assert self.cooldown_until is not None
        started = self.clock()
        total = max(0.001, self.cooldown_until - started)
        progress: Optional[Progress] = None
        task_id = None
        if self.console is not None and self.show_progress:
            progress = Progress(
                SpinnerColumn(style="cyan"),
                TextColumn("[bold cyan][THERMAL GUARD][/] Cooling down..."),
                BarColumn(bar_width=36, complete_style="cyan", finished_style="green"),
                TextColumn("[bold]Remaining: {task.fields[remaining]}[/]"),
                console=self.console,
                transient=False,
            )
            progress.start()
            task_id = progress.add_task("cooldown", total=total, remaining=format_mmss(total))
        ticks = 0
        early_wakeup_enabled = bool(self.settings.get("early_wakeup", True))
        try:
            while True:
                if self.profile.name == "turbo":
                    self._event("thermal", "cooldown cut short (turbo profile)")
                    break
                if early_wakeup_enabled and self.sensor is not None and ticks >= 1:
                    reading = self.sensor.read(polling_cooldown=True)
                    if not reading.is_hot:
                        self.early_wakeups += 1
                        self._event(
                            "thermal",
                            "device returned to nominal temperature; waking up immediately",
                        )
                        break
                remaining = self.remaining_cooldown()
                if progress is not None and task_id is not None:
                    progress.update(task_id, completed=total - remaining, remaining=format_mmss(remaining))
                if remaining <= 0:
                    break
                self.sleeper(min(1.0, remaining))
                ticks += 1
        finally:
            if progress is not None:
                progress.stop()
        return self.clock() - started

    def _reset_counters(self) -> None:
        self.tasks_since_cooldown = 0
        self.work_seconds_since_cooldown = 0.0
        self.cooldown_until = None
        self._notify()

    def export_state(self) -> dict[str, Any]:
        return {
            "tasks_since_cooldown": self.tasks_since_cooldown,
            "work_seconds_since_cooldown": round(self.work_seconds_since_cooldown, 3),
            "cooldown_until": self.cooldown_until,
            "total_cooldowns": self.total_cooldowns,
            "early_wakeups": self.early_wakeups,
        }

    def import_state(self, data: Optional[dict[str, Any]]) -> None:
        data = data or {}
        self.tasks_since_cooldown = int(data.get("tasks_since_cooldown", 0))
        self.work_seconds_since_cooldown = float(data.get("work_seconds_since_cooldown", 0.0))
        until = data.get("cooldown_until")
        self.cooldown_until = float(until) if until is not None else None
        self.total_cooldowns = int(data.get("total_cooldowns", 0))
        self.early_wakeups = int(data.get("early_wakeups", 0))

    def status(self) -> dict[str, Any]:
        prof = self.profile
        sensor_level = self.last_reading.level if self.last_reading else "nominal"
        return {
            "profile": asdict(prof),
            "sensor_level": sensor_level,
            "tasks_since_cooldown": self.tasks_since_cooldown,
            "work_seconds_since_cooldown": round(self.work_seconds_since_cooldown, 1),
            "cooldown_remaining": round(self.remaining_cooldown(), 1),
            "total_cooldowns": self.total_cooldowns,
            "early_wakeups": self.early_wakeups,
            "unload_model_on_cooldown": bool(self.settings.get("unload_model_on_cooldown", True)),
        }

    def _notify(self) -> None:
        if self.on_state_change is not None:
            self.on_state_change(self.export_state())

    def _event(self, kind: str, message: str) -> None:
        if self.on_event is not None:
            self.on_event(kind, message)
