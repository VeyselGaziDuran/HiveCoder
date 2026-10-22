"""Garbage collector: wipes agent/test by-products so the workspace stays pristine."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable

JUNK_DIRS: frozenset[str] = frozenset({".hive_tmp", ".pytest_cache", "__pycache__", ".mypy_cache", ".ruff_cache"})
JUNK_FILE_PATTERNS: tuple[str, ...] = ("*.pyc", "*.pyo", ".coverage", ".coverage.*")
PROTECTED_DIRS: frozenset[str] = frozenset({".git", "venv", ".venv", "env", "node_modules", "snapshots"})


@dataclass
class GCReport:
    removed: list[str] = field(default_factory=list)
    bytes_freed: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.removed)


def _size_of(path: Path) -> int:
    if path.is_file() or path.is_symlink():
        try:
            return path.lstat().st_size
        except OSError:
            return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).lstat().st_size
            except OSError:
                continue
    return total


class GarbageCollector:
    def __init__(
        self,
        junk_dirs: Iterable[str] = JUNK_DIRS,
        junk_patterns: Iterable[str] = JUNK_FILE_PATTERNS,
        protected_dirs: Iterable[str] = PROTECTED_DIRS,
    ) -> None:
        self.junk_dirs = frozenset(junk_dirs)
        self.junk_patterns = tuple(junk_patterns)
        self.protected_dirs = frozenset(protected_dirs)

    def scan(self, root: str | Path) -> list[Path]:
        """List junk paths below ``root`` without deleting anything."""
        base = Path(root).resolve()
        found: list[Path] = []
        if not base.exists():
            return found
        for current, dirs, files in os.walk(base, followlinks=False):
            keep: list[str] = []
            for d in dirs:
                if d in self.junk_dirs:
                    found.append(Path(current) / d)
                elif d not in self.protected_dirs:
                    keep.append(d)
            dirs[:] = keep
            for name in files:
                if any(fnmatch(name, pat) for pat in self.junk_patterns):
                    found.append(Path(current) / name)
        return found

    def collect(self, root: str | Path) -> GCReport:
        base = Path(root).resolve()
        report = GCReport()
        for path in self.scan(base):
            size = _size_of(path)
            try:
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            except FileNotFoundError:
                continue
            except OSError as exc:
                report.errors.append(f"{path}: {exc}")
                continue
            report.removed.append(path.relative_to(base).as_posix())
            report.bytes_freed += size
        return report
