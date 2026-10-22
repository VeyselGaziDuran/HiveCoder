"""SignatureProbe: extracts real callable signatures from installed libraries via ``inspect``.

When a small model imports standard-library or third-party modules, showing the
exact callable signatures from the active Python runtime prevents hallucinated
keyword arguments or methods.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import re
import sys
from typing import Iterable

_EXCLUDED_STDLIB: frozenset[str] = frozenset(
    {
        "antigravity",
        "this",
        "idlelib",
        "tkinter",
        "turtle",
        "turtledemo",
        "webbrowser",
        "curses",
        "ensurepip",
        "venv",
        "multiprocessing",
        "dbm",
    }
)
_STDLIB_NAMES: frozenset[str] = frozenset(getattr(sys, "stdlib_module_names", ())) - _EXCLUDED_STDLIB


class SignatureProbe:
    """Extracts live function and class constructor signatures from importable modules."""

    SAFE_MODULES: frozenset[str] = _STDLIB_NAMES | frozenset(
        {
            "json",
            "sqlite3",
            "pathlib",
            "datetime",
            "collections",
            "itertools",
            "functools",
            "hashlib",
            "re",
            "math",
            "statistics",
            "csv",
            "urllib.parse",
            "pydantic",
            "pytest",
            "requests",
            "rich",
            "fastapi",
            "flask",
            "sqlalchemy",
            "httpx",
            "click",
            "typer",
        }
    )

    def inspect_symbol(self, module_name: str, symbol_name: str = "") -> list[str]:
        """Return formatted signature strings for ``module_name`` (or ``module_name.symbol_name``)."""
        if module_name not in self.SAFE_MODULES and module_name.split(".")[0] not in self.SAFE_MODULES:
            return []
        try:
            mod = importlib.import_module(module_name)
        except Exception:  # noqa: BLE001
            return []

        if symbol_name:
            target = getattr(mod, symbol_name, None)
            if target is None:
                return []
            sig = self._format_callable(f"{module_name}.{symbol_name}", target)
            return [sig] if sig else []

        results: list[str] = []
        for attr in sorted(dir(mod)):
            if attr.startswith("_"):
                continue
            obj = getattr(mod, attr, None)
            if callable(obj):
                sig = self._format_callable(f"{module_name}.{attr}", obj)
                if sig:
                    results.append(sig)
            if len(results) >= 12:
                break
        return results

    def inspect_imports_in_source(self, source: str) -> list[str]:
        """Parse ``import`` statements in ``source`` and return live signatures of external symbols."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []
        collected: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                for alias in node.names:
                    if alias.name != "*":
                        collected.extend(self.inspect_symbol(node.module, alias.name))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    collected.extend(self.inspect_symbol(alias.name)[:4])
        seen: set[str] = set()
        unique: list[str] = []
        for item in collected:
            if item not in seen:
                seen.add(item)
                unique.append(item)
        return unique[:15]

    def inspect_traceback(self, source: str, error_output: str) -> list[str]:
        """Combine source import signatures with symbols referenced in ``error_output``."""
        collected = list(self.inspect_imports_in_source(source))
        for sym, mod in re.findall(r"cannot import name ['\"](\w+)['\"] from ['\"]([A-Za-z0-9_.]+)['\"]", error_output):
            collected.extend(self.inspect_symbol(mod)[:8])
        for mod, _attr in re.findall(r"module ['\"]([A-Za-z0-9_.]+)['\"] has no attribute ['\"](\w+)['\"]", error_output):
            collected.extend(self.inspect_symbol(mod)[:8])
        seen: set[str] = set()
        unique: list[str] = []
        for item in collected:
            if item not in seen:
                seen.add(item)
                unique.append(item)
        return unique[:15]

    @staticmethod
    def _format_callable(qualified_name: str, obj: object) -> str:
        try:
            sig = inspect.signature(obj)  # type: ignore[arg-type]
            return f"{qualified_name}{sig}"
        except (TypeError, ValueError):
            return ""


def format_probe_summary(lines: Iterable[str]) -> str:
    items = [line for line in lines if line.strip()]
    if not items:
        return ""
    return "## LIVE LIBRARY SIGNATURES\n" + "\n".join(f"- `{item}`" for item in items)
