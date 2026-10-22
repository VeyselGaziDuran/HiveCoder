"""Hardware thermal & inference-throughput sensor for adaptive cooldown control.

Instead of pausing blindly every N tasks while the machine is cool,
:class:`ThermalSensor` monitors three signals:

1. **OS / Hardware Temperature & Thermal Pressure** (Linux ``/sys/class/thermal``
   & ``hwmon`` °C sensors; macOS ``pmset -g therm`` / ``sysctl`` thermal levels).
2. **CPU Load Ratio** (``os.getloadavg()[0] / os.cpu_count()``).
3. **Inference Throughput Degradation** (characters/second drop compared to the
   rolling baseline of the first tasks — a direct indicator of GPU/CPU thermal
   throttling on fanless laptops).

Levels:
- ``nominal``: machine is cool -> continue working with zero cooldown.
- ``moderate``: machine is warming up -> trigger 5-minute (300s) max cooldown.
- ``heavy``: machine is hot -> trigger 10-minute (600s) max cooldown.

During a cooldown, the sensor is polled periodically; as soon as the reading
returns to ``nominal``, the orchestrator wakes up immediately without waiting
for the remaining timer or asking the user.
"""

from __future__ import annotations

import glob
import os
import platform
import re
import subprocess
from dataclasses import dataclass
from typing import Callable, Literal, Optional

ThermalLevel = Literal["nominal", "moderate", "heavy"]


@dataclass(frozen=True)
class ThermalReading:
    level: ThermalLevel
    temp_celsius: Optional[float] = None
    cpu_load_ratio: float = 0.0
    throughput_ratio: float = 1.0
    recommended_cooldown_seconds: float = 0.0
    reason: str = "nominal"

    @property
    def is_hot(self) -> bool:
        return self.level in ("moderate", "heavy")


class ThermalSensor:
    """Cross-platform thermal and throughput throttling detector."""

    def __init__(
        self,
        moderate_temp_c: float = 72.0,
        heavy_temp_c: float = 82.0,
        moderate_load_ratio: float = 0.85,
        heavy_load_ratio: float = 1.15,
        moderate_speed_ratio: float = 0.78,
        heavy_speed_ratio: float = 0.58,
        moderate_cooldown_seconds: float = 300.0,
        heavy_cooldown_seconds: float = 600.0,
        probe_fn: Optional[Callable[[], Optional[ThermalReading]]] = None,
    ) -> None:
        self.moderate_temp_c = moderate_temp_c
        self.heavy_temp_c = heavy_temp_c
        self.moderate_load_ratio = moderate_load_ratio
        self.heavy_load_ratio = heavy_load_ratio
        self.moderate_speed_ratio = moderate_speed_ratio
        self.heavy_speed_ratio = heavy_speed_ratio
        self.moderate_cooldown_seconds = moderate_cooldown_seconds
        self.heavy_cooldown_seconds = heavy_cooldown_seconds
        self.probe_fn = probe_fn

        self.baseline_chars_per_sec: Optional[float] = None
        self.latest_chars_per_sec: Optional[float] = None

    def record_throughput(self, elapsed_seconds: float, response_chars: int) -> None:
        """Record model generation speed to detect thermal throttling slowdowns."""
        if elapsed_seconds <= 0.05 or response_chars <= 20:
            return
        speed = float(response_chars) / float(elapsed_seconds)
        self.latest_chars_per_sec = speed
        if self.baseline_chars_per_sec is None:
            self.baseline_chars_per_sec = speed
        elif speed > self.baseline_chars_per_sec:
            self.baseline_chars_per_sec = 0.7 * self.baseline_chars_per_sec + 0.3 * speed

    def reset_throughput_after_cooldown(self) -> None:
        """Clear latest throttled speed after a cooldown finishes."""
        self.latest_chars_per_sec = self.baseline_chars_per_sec

    def read(self) -> ThermalReading:
        """Return the current :class:`ThermalReading`."""
        if self.probe_fn is not None:
            custom = self.probe_fn()
            if custom is not None:
                return custom

        temp_c = self._read_temperature_celsius()
        os_level = self._read_macos_thermal_pressure()
        load_ratio = self._read_cpu_load_ratio()
        speed_ratio = self._throughput_ratio()

        # 1. Check heavy conditions (10-minute max cooldown)
        if temp_c is not None and temp_c >= self.heavy_temp_c:
            return ThermalReading(
                level="heavy",
                temp_celsius=temp_c,
                cpu_load_ratio=load_ratio,
                throughput_ratio=speed_ratio,
                recommended_cooldown_seconds=self.heavy_cooldown_seconds,
                reason=f"temperature {temp_c:.1f}C >= {self.heavy_temp_c:.1f}C",
            )
        if os_level == "heavy":
            return ThermalReading(
                level="heavy",
                temp_celsius=temp_c,
                cpu_load_ratio=load_ratio,
                throughput_ratio=speed_ratio,
                recommended_cooldown_seconds=self.heavy_cooldown_seconds,
                reason="OS thermal pressure is heavy",
            )
        if speed_ratio <= self.heavy_speed_ratio and load_ratio >= self.moderate_load_ratio:
            return ThermalReading(
                level="heavy",
                temp_celsius=temp_c,
                cpu_load_ratio=load_ratio,
                throughput_ratio=speed_ratio,
                recommended_cooldown_seconds=self.heavy_cooldown_seconds,
                reason=f"severe throughput drop ({speed_ratio:.0%} of baseline) under high CPU load",
            )

        # 2. Check moderate conditions (5-minute max cooldown)
        if temp_c is not None and temp_c >= self.moderate_temp_c:
            return ThermalReading(
                level="moderate",
                temp_celsius=temp_c,
                cpu_load_ratio=load_ratio,
                throughput_ratio=speed_ratio,
                recommended_cooldown_seconds=self.moderate_cooldown_seconds,
                reason=f"temperature {temp_c:.1f}C >= {self.moderate_temp_c:.1f}C",
            )
        if os_level == "moderate":
            return ThermalReading(
                level="moderate",
                temp_celsius=temp_c,
                cpu_load_ratio=load_ratio,
                throughput_ratio=speed_ratio,
                recommended_cooldown_seconds=self.moderate_cooldown_seconds,
                reason="OS thermal pressure is moderate",
            )
        if speed_ratio <= self.moderate_speed_ratio:
            return ThermalReading(
                level="moderate",
                temp_celsius=temp_c,
                cpu_load_ratio=load_ratio,
                throughput_ratio=speed_ratio,
                recommended_cooldown_seconds=self.moderate_cooldown_seconds,
                reason=f"inference speed dropped to {speed_ratio:.0%} of baseline",
            )
        if load_ratio >= self.heavy_load_ratio:
            return ThermalReading(
                level="moderate",
                temp_celsius=temp_c,
                cpu_load_ratio=load_ratio,
                throughput_ratio=speed_ratio,
                recommended_cooldown_seconds=self.moderate_cooldown_seconds,
                reason=f"sustained CPU load ratio {load_ratio:.2f}",
            )

        return ThermalReading(
            level="nominal",
            temp_celsius=temp_c,
            cpu_load_ratio=load_ratio,
            throughput_ratio=speed_ratio,
            recommended_cooldown_seconds=0.0,
            reason="nominal",
        )

    def _throughput_ratio(self) -> float:
        if (
            self.baseline_chars_per_sec is None
            or self.latest_chars_per_sec is None
            or self.baseline_chars_per_sec <= 0
        ):
            return 1.0
        return min(1.0, self.latest_chars_per_sec / self.baseline_chars_per_sec)

    @staticmethod
    def _read_cpu_load_ratio() -> float:
        try:
            load1, _, _ = os.getloadavg()
            cores = max(1, os.cpu_count() or 1)
            return round(load1 / cores, 3)
        except (AttributeError, OSError):
            return 0.0

    @staticmethod
    def _read_temperature_celsius() -> Optional[float]:
        candidates = glob.glob("/sys/class/thermal/thermal_zone*/temp") + glob.glob(
            "/sys/class/hwmon/hwmon*/temp*_input"
        )
        temps: list[float] = []
        for path in candidates[:16]:
            try:
                raw = open(path, encoding="utf-8").read().strip()
                val = float(raw)
                if val > 1000:
                    val /= 1000.0
                if 15.0 <= val <= 125.0:
                    temps.append(val)
            except (OSError, ValueError):
                continue
        return max(temps) if temps else None

    @staticmethod
    def _read_macos_thermal_pressure() -> ThermalLevel:
        if platform.system() != "Darwin":
            return "nominal"
        try:
            proc = subprocess.run(
                ["pmset", "-g", "therm"],
                capture_output=True,
                text=True,
                timeout=1.0,
                check=False,
            )
            out = (proc.stdout or "").lower()
            speed_match = re.search(r"cpu_speed_limit\s*=\s*(\d+)", out)
            if speed_match:
                limit = int(speed_match.group(1))
                if limit <= 65:
                    return "heavy"
                if limit < 95:
                    return "moderate"
            if "heavy" in out or "trapping" in out or "sleeping" in out:
                return "heavy"
            if "moderate" in out or "fair" in out:
                return "moderate"
        except (OSError, subprocess.SubprocessError):
            return "nominal"
        return "nominal"
