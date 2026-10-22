"""Validator: real ``py_compile`` + ``pytest`` with automatic PYTHONPATH injection.

Generated tests usually do ``from app import get_words`` while living in
``tests/``.  Without the workspace on ``sys.path`` that import fails, so the
validator always injects the workspace (and ``src/`` when present) into
``PYTHONPATH`` and disables bytecode writing so no ``__pycache__`` leaks.
"""

from __future__ import annotations

import json
import os
import py_compile
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from core.blackboard import CODE_SUFFIXES, IGNORED_DIRS

HIVE_TMP = ".hive_tmp"
MAX_OUTPUT_CHARS = 6000
BRACE_CHECK_SUFFIXES: frozenset[str] = CODE_SUFFIXES | {".css", ".scss"}


def _check_brace_balance(source: str) -> Optional[str]:
    """Verify balanced ``{}``, ``()``, and ``[]`` outside strings and comments."""
    stack: list[tuple[str, int]] = []
    pairs = {"}": "{", ")": "(", "]": "["}
    expected_close = {"{": "}", "(": ")", "[": "]"}
    i = 0
    n = len(source)
    lineno = 1
    in_line_comment = False
    in_block_comment = False
    in_string: Optional[str] = None

    while i < n:
        ch = source[i]
        nxt = source[i + 1] if i + 1 < n else ""
        if ch == "\n":
            lineno += 1
            in_line_comment = False
            if in_string in ("'", '"'):
                in_string = None
            i += 1
            continue
        if in_line_comment:
            i += 1
            continue
        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
            else:
                i += 1
            continue
        if in_string is not None:
            if ch == "\\":
                i += 2
                continue
            if ch == in_string:
                in_string = None
            i += 1
            continue
        if ch == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue
        if ch in ("'", '"', "`"):
            prev = source[i - 1] if i > 0 else ""
            if ch == "'" and prev.isalnum() and nxt.isalpha():
                i += 1
                continue
            in_string = ch
            i += 1
            continue
        if ch in expected_close:
            stack.append((ch, lineno))
        elif ch in pairs:
            if not stack:
                return f"unmatched closing '{ch}' at line {lineno}"
            top, open_line = stack.pop()
            if top != pairs[ch]:
                return f"mismatched '{ch}' at line {lineno} (opened '{top}' at line {open_line})"
        i += 1

    if stack:
        top, open_line = stack[-1]
        return f"unclosed '{top}' opened at line {open_line} (expected '{expected_close[top]}')"
    return None


@dataclass
class ValidationResult:
    passed: bool
    stage: str
    output: str = ""
    files: list[str] = field(default_factory=list)
    returncode: Optional[int] = None

    def summary(self) -> str:
        state = "PASSED" if self.passed else "FAILED"
        return f"[{self.stage}] {state}" + (f"\n{self.output}" if self.output else "")


def _tail(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    return text if len(text) <= limit else "...(truncated)...\n" + text[-limit:]


def is_test_file(rel_path: str) -> bool:
    name = Path(rel_path).name
    return name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py"))


class Validator:
    def __init__(
        self,
        workspace: str | Path,
        python_executable: Optional[str] = None,
        pytest_timeout: float = 180.0,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.python = python_executable or sys.executable
        self.pytest_timeout = pytest_timeout

    # ------------------------------------------------------------------ environment
    def build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        paths = [str(self.workspace)]
        src = self.workspace / "src"
        if src.is_dir():
            paths.append(str(src))
        existing = env.get("PYTHONPATH", "")
        if existing:
            paths.extend(p for p in existing.split(os.pathsep) if p and p not in paths)
        env["PYTHONPATH"] = os.pathsep.join(paths)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPYCACHEPREFIX"] = str(self.workspace / HIVE_TMP / "pycache")
        env.pop("PYTEST_ADDOPTS", None)
        return env

    # ------------------------------------------------------------------ syntax
    def check_syntax(self, rel_paths: Iterable[str]) -> ValidationResult:
        files = list(rel_paths)
        errors: list[str] = []
        tmp_dir = self.workspace / HIVE_TMP / "compile"
        for rel in files:
            path = self.workspace / rel
            if not path.exists():
                errors.append(f"{rel}: file not found")
                continue
            suffix = path.suffix.lower()
            if suffix == ".py":
                tmp_dir.mkdir(parents=True, exist_ok=True)
                cfile = tmp_dir / (rel.replace("/", "__") + "c")
                try:
                    py_compile.compile(str(path), cfile=str(cfile), doraise=True)
                except py_compile.PyCompileError as exc:
                    errors.append(f"{rel}: {exc.msg.strip()}")
            elif suffix == ".json":
                try:
                    json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    errors.append(f"{rel}: invalid JSON at line {exc.lineno} col {exc.colno}: {exc.msg}")
            elif suffix in BRACE_CHECK_SUFFIXES:
                err = _check_brace_balance(path.read_text(encoding="utf-8", errors="replace"))
                if err:
                    errors.append(f"{rel}: syntax error ({err})")
        if errors:
            return ValidationResult(False, "syntax", "\n".join(errors), files)
        return ValidationResult(True, "syntax", f"py_compile OK ({len(files)} file(s))", files)

    # ------------------------------------------------------------------ tests
    def discover_tests(self) -> list[str]:
        found: list[str] = []
        for root, dirs, names in os.walk(self.workspace):
            dirs[:] = sorted(d for d in dirs if d not in IGNORED_DIRS and not d.startswith("."))
            for name in sorted(names):
                rel = (Path(root) / name).relative_to(self.workspace).as_posix()
                if is_test_file(rel):
                    found.append(rel)
        return found

    def run_pytest(self, targets: Optional[list[str]] = None) -> ValidationResult:
        targets = list(targets or [])
        cmd = [
            self.python, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider",
            "--rootdir", str(self.workspace), *targets,
        ]
        try:
            proc = subprocess.run(
                cmd,
                cwd=self.workspace,
                env=self.build_env(),
                capture_output=True,
                text=True,
                timeout=self.pytest_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            out = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            return ValidationResult(False, "pytest", _tail(f"pytest timed out after {self.pytest_timeout}s\n{out}"), targets)
        except FileNotFoundError as exc:
            return ValidationResult(False, "pytest", f"python executable not found: {exc}", targets)
        output = _tail((proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr.strip() else ""))
        if proc.returncode == 0:
            return ValidationResult(True, "pytest", output.strip(), targets, proc.returncode)
        if proc.returncode == 5 and not targets:
            return ValidationResult(True, "pytest", "no tests collected", targets, proc.returncode)
        return ValidationResult(False, "pytest", output.strip(), targets, proc.returncode)

    def validate(self, rel_paths: list[str], run_tests: bool = True) -> ValidationResult:
        """Syntax-check ``rel_paths`` then run the relevant tests.

        * If any written file is a test file, exactly those tests run.
        * Otherwise the whole suite runs (regression check) when tests exist.
        """
        syntax = self.check_syntax(rel_paths)
        if not syntax.passed or not run_tests:
            return syntax
        own_tests = [p for p in rel_paths if is_test_file(p)]
        if own_tests:
            result = self.run_pytest(own_tests)
        elif self.discover_tests():
            result = self.run_pytest()
        else:
            return ValidationResult(True, "syntax", syntax.output + "; no tests present yet", rel_paths)
        result.files = rel_paths
        if result.passed:
            result.output = f"{syntax.output}\n{result.output}".strip()
        return result
