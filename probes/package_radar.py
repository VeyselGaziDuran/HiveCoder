"""PackageRadarProbe: live PyPI metadata lookup with instant offline fallback.

When internet connectivity is available, :class:`PackageRadarProbe` queries the
official PyPI JSON endpoint (``https://pypi.org/pypi/<package>/json``) with a
short timeout to retrieve the latest stable version, summary, and Python version
requirements.  If the machine is offline, it falls back seamlessly to local
``importlib.metadata`` without raising errors.
"""

from __future__ import annotations

import importlib.metadata as metadata
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Optional

import requests


@dataclass(frozen=True)
class PackageInfo:
    name: str
    installed_version: Optional[str]
    latest_version: Optional[str]
    requires_python: str = ""
    summary: str = ""
    source: str = "local"

    def format_brief(self) -> str:
        ver_part = f"installed={self.installed_version or 'no'}"
        if self.latest_version:
            ver_part += f", latest={self.latest_version}"
        req_part = f" (requires Python {self.requires_python})" if self.requires_python else ""
        sum_part = f" — {self.summary}" if self.summary else ""
        return f"{self.name} [{ver_part}]{req_part}{sum_part}"


class PackageRadarProbe:
    """Queries local distributions and optional PyPI JSON metadata."""

    PYPI_URL = "https://pypi.org/pypi/{package}/json"

    def __init__(
        self,
        online_enabled: bool = True,
        timeout: float = 2.0,
        session: Optional[requests.Session] = None,
        python_executable: Optional[str] = None,
    ) -> None:
        self.online_enabled = online_enabled
        self.timeout = timeout
        self.session = session or requests.Session()
        self.python = python_executable or sys.executable
        self._cache: dict[str, PackageInfo] = {}

    def lookup(self, package_name: str) -> PackageInfo:
        norm = re.sub(r"[^A-Za-z0-9_.-]+", "", package_name.strip().lower())
        if not norm:
            raise ValueError("Package name must not be empty")
        if norm in self._cache:
            return self._cache[norm]

        installed_ver: Optional[str] = None
        summary = ""
        requires_py = ""
        try:
            dist = metadata.distribution(norm)
            installed_ver = dist.version
            summary = (dist.metadata["Summary"] or "").strip()
            requires_py = (dist.metadata["Requires-Python"] or "").strip()
        except metadata.PackageNotFoundError:
            installed_ver = None

        latest_ver = installed_ver
        source = "local"
        if self.online_enabled:
            remote = self._fetch_pypi(norm)
            if remote is not None:
                info = remote.get("info", {})
                latest_ver = str(info.get("version") or installed_ver or "") or None
                summary = str(info.get("summary") or summary or "").strip()[:140]
                requires_py = str(info.get("requires_python") or requires_py or "").strip()
                source = "pypi"

        result = PackageInfo(
            name=norm,
            installed_version=installed_ver,
            latest_version=latest_ver,
            requires_python=requires_py,
            summary=summary,
            source=source,
        )
        self._cache[norm] = result
        return result

    def _fetch_pypi(self, package_name: str) -> Optional[dict[str, Any]]:
        url = self.PYPI_URL.format(package=package_name)
        try:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code != 200:
                return None
            data = resp.json()
            return data if isinstance(data, dict) else None
        except (requests.RequestException, OSError, ValueError):
            return None

    def scan_goal_packages(self, text: str) -> list[PackageInfo]:
        """Detect well-known package mentions in a task goal and return their version info."""
        candidates = (
            "pydantic",
            "fastapi",
            "flask",
            "pytest",
            "requests",
            "httpx",
            "rich",
            "sqlalchemy",
            "sqlite3",
            "aiohttp",
            "click",
            "typer",
        )
        lowered = text.lower()
        found: list[PackageInfo] = []
        for pkg in candidates:
            if pkg == "sqlite3":
                continue
            if re.search(rf"\b{re.escape(pkg)}\b", lowered):
                found.append(self.lookup(pkg))
        return found

    def install_package(self, package_spec: str, timeout: float = 60.0) -> bool:
        """Install a missing package into the active Python environment."""
        cleaned = package_spec.strip()
        if not re.match(r"^[A-Za-z0-9_.\-\[\]>=<!~]+$", cleaned):
            return False
        try:
            proc = subprocess.run(
                [self.python, "-m", "pip", "install", "-q", cleaned],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            return proc.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False
