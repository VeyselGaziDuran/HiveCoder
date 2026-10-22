"""RuntimeProbe: inspects active Python version, OS, and installed package versions.

Small local models often have stale training cutoffs and attempt to emit
deprecated APIs (e.g., Pydantic v1 ``.dict()`` / ``@validator`` when Pydantic v2
is installed, or pre-3.10 typing constructs).  :class:`RuntimeProbe` inspects
the local environment in milliseconds and generates deterministic version
constraints for the prompt context.
"""

from __future__ import annotations

import importlib.metadata as metadata
import platform
import sys
from dataclasses import dataclass, field


@dataclass
class RuntimeEnvironment:
    python_version: str
    os_name: str
    installed_packages: dict[str, str] = field(default_factory=dict)
    compatibility_hints: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            f"## RUNTIME ENVIRONMENT ({self.os_name}, Python {self.python_version})",
        ]
        if self.installed_packages:
            key_pkgs = ", ".join(
                f"{name}=={ver}"
                for name, ver in sorted(self.installed_packages.items())[:18]
            )
            lines.append(f"- Installed packages: {key_pkgs}")
        if self.compatibility_hints:
            lines.append("- Active version rules:")
            for hint in self.compatibility_hints:
                lines.append(f"  * {hint}")
        return "\n".join(lines)


class RuntimeProbe:
    """Introspects the local Python runtime and installed distribution versions."""

    TRACKED_PACKAGES: tuple[str, ...] = (
        "pydantic",
        "pytest",
        "rich",
        "requests",
        "fastapi",
        "flask",
        "sqlalchemy",
        "httpx",
        "aiohttp",
        "numpy",
        "pandas",
        "click",
        "typer",
    )

    def inspect(self) -> RuntimeEnvironment:
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        os_name = f"{platform.system()} {platform.machine()}".strip()
        installed: dict[str, str] = {}
        try:
            for dist in metadata.distributions():
                name = (dist.metadata["Name"] or "").strip().lower()
                ver = (dist.version or "").strip()
                if name and ver:
                    installed[name] = ver
        except Exception:  # noqa: BLE001
            installed = {}

        filtered = {
            k: installed[k]
            for k in self.TRACKED_PACKAGES
            if k in installed
        }
        hints = self._build_hints(installed)
        return RuntimeEnvironment(
            python_version=py_ver,
            os_name=os_name,
            installed_packages=filtered,
            compatibility_hints=hints,
        )

    @staticmethod
    def _build_hints(installed: dict[str, str]) -> list[str]:
        hints: list[str] = []
        if sys.version_info >= (3, 10):
            hints.append(
                f"Python {sys.version_info.major}.{sys.version_info.minor} active: "
                "built-in generic types (`list[dict]`, `dict[str, Any]`) and `X | Y` unions are supported."
            )
        pyd_ver = installed.get("pydantic", "")
        if pyd_ver.startswith("2."):
            hints.append(
                f"Pydantic {pyd_ver} (v2) installed: use `model_dump()`, `model_validate()`, "
                "`@field_validator`, and `ConfigDict` instead of deprecated v1 `.dict()`, `.parse_obj()`, or `@validator`."
            )
        pytest_ver = installed.get("pytest", "")
        if pytest_ver:
            hints.append(
                f"pytest {pytest_ver} installed: use plain `assert` statements and built-in `tmp_path` / `monkeypatch` fixtures."
            )
        return hints
