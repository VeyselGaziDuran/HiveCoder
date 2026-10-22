"""Thermal Guard: keeps fanless laptops cool while local models grind.

Three mechanisms:

* **Micro-pause** - a short breath between every agent micro-task.
* **Deep cooldown** - after N micro-tasks *or* M seconds of continuous model
  work, the model is evicted from RAM/VRAM (Ollama ``keep_alive: 0``) and the
  machine rests for 5-10 minutes with a live Rich countdown.
* **Persistence** - counters and the cooldown deadline are exported so the
  orchestrator can store them in ``state.json``; an interrupted cooldown is
  resumed (or skipped, if the user switched to ``turbo``) after a restart.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

DEFAULT_PROFILES: dict[str, dict[str, float]] = {
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
        return self.cooldown_seconds > 0 and (
            self.cooldown_every_tasks > 0 or self.max_continuous_work_seconds > 0
        )


def format_mmss(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    return f"{total // 60:02d}:{total % 60:02d}"


class ThermalGuard:
    """Decides when to breathe / cool down and performs the cooldown."""

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
    ) -> None:
        self.settings = settings
        self.settings.setdefault("profile", "eco")
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
        self.tasks_since_cooldown = 0
        self.work_seconds_since_cooldown = 0.0
        self.cooldown_until: Optional[float] = None
        self.total_cooldowns = 0
        self.last_unload_ok: Optional[bool] = None

    # ------------------------------------------------------------------ profile
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
        """Switch profile; optionally override its deep-cooldown length in minutes."""
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
            if minutes > 0 and int(profile_values.get("cooldown_every_tasks", 0)) <= 0:
                profile_values["cooldown_every_tasks"] = 3
        self._notify()
        return self.profile

    # ------------------------------------------------------------------ accounting
    def record_task(self, work_seconds: float) -> None:
        self.tasks_since_cooldown += 1
        self.work_seconds_since_cooldown += max(0.0, float(work_seconds))
        self._notify()

    def needs_cooldown(self) -> bool:
        prof = self.profile
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

    def remaining_cooldown(self) -> float:
        if self.cooldown_until is None:
            return 0.0
        return max(0.0, self.cooldown_until - self.clock())

    # ------------------------------------------------------------------ actions
    def micro_pause(self) -> float:
        seconds = self.profile.micro_pause_seconds
        if seconds > 0:
            self._event("thermal", f"micro-pause {seconds:.0f}s")
            self.sleeper(seconds)
        return seconds

    def deep_cooldown(self, duration: Optional[float] = None) -> float:
        """Unload the model and rest. Returns seconds actually waited."""
        seconds = float(self.profile.cooldown_seconds if duration is None else duration)
        if seconds <= 0:
            self._reset_counters()
            return 0.0
        if self.unloader is not None and self.settings.get("unload_model_on_cooldown", True):
            try:
                self.last_unload_ok = bool(self.unloader())
            except Exception:  # noqa: BLE001 - unloading is best effort, cooling must still happen
                self.last_unload_ok = False
            self._event(
                "thermal",
                "model evicted from RAM/VRAM (keep_alive=0)" if self.last_unload_ok else "model unload skipped/failed",
            )
        self.cooldown_until = self.clock() + seconds
        self._notify()
        self._event("thermal", f"deep cooldown {format_mmss(seconds)}")
        waited = self._countdown()
        self.total_cooldowns += 1
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
        remaining = min(remaining, self.profile.cooldown_seconds)
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
                TextColumn("[bold cyan][TERMAL KORUMA][/] MacBook soğutuluyor..."),
                BarColumn(bar_width=36, complete_style="cyan", finished_style="green"),
                TextColumn("[bold]Kalan süre: {task.fields[remaining]}[/]"),
                console=self.console,
                transient=False,
            )
            progress.start()
            task_id = progress.add_task("cooldown", total=total, remaining=format_mmss(total))
        try:
            while True:
                if not self.profile.deep_cooldown_enabled:
                    self._event("thermal", "cooldown cut short (turbo profile)")
                    break
                remaining = self.remaining_cooldown()
                if progress is not None and task_id is not None:
                    progress.update(task_id, completed=total - remaining, remaining=format_mmss(remaining))
                if remaining <= 0:
                    break
                self.sleeper(min(1.0, remaining))
        finally:
            if progress is not None:
                progress.stop()
        return self.clock() - started

    def _reset_counters(self) -> None:
        self.tasks_since_cooldown = 0
        self.work_seconds_since_cooldown = 0.0
        self.cooldown_until = None
        self._notify()

    # ------------------------------------------------------------------ state
    def export_state(self) -> dict[str, Any]:
        return {
            "tasks_since_cooldown": self.tasks_since_cooldown,
            "work_seconds_since_cooldown": round(self.work_seconds_since_cooldown, 3),
            "cooldown_until": self.cooldown_until,
            "total_cooldowns": self.total_cooldowns,
        }

    def import_state(self, data: Optional[dict[str, Any]]) -> None:
        data = data or {}
        self.tasks_since_cooldown = int(data.get("tasks_since_cooldown", 0))
        self.work_seconds_since_cooldown = float(data.get("work_seconds_since_cooldown", 0.0))
        until = data.get("cooldown_until")
        self.cooldown_until = float(until) if until is not None else None
        self.total_cooldowns = int(data.get("total_cooldowns", 0))

    def status(self) -> dict[str, Any]:
        prof = self.profile
        return {
            "profile": asdict(prof),
            "tasks_since_cooldown": self.tasks_since_cooldown,
            "work_seconds_since_cooldown": round(self.work_seconds_since_cooldown, 1),
            "cooldown_remaining": round(self.remaining_cooldown(), 1),
            "total_cooldowns": self.total_cooldowns,
            "unload_model_on_cooldown": bool(self.settings.get("unload_model_on_cooldown", True)),
        }

    def _notify(self) -> None:
        if self.on_state_change is not None:
            self.on_state_change(self.export_state())

    def _event(self, kind: str, message: str) -> None:
        if self.on_event is not None:
            self.on_event(kind, message)
