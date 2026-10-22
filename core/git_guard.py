"""Git Guard: crash-safe snapshots and rollback without touching the user's repo.

Snapshots live in a *private* git directory under ``memory/snapshots/`` that
uses the workspace as its work-tree.  The user's own ``.git`` (if any), branch,
index and history are never modified.  Before every micro-task a snapshot
commit is taken; if a micro-task fails permanently or the run crashes, the
workspace is restored to that exact snapshot.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Optional

DEFAULT_EXCLUDES = (
    ".git/",
    "venv/",
    ".venv/",
    "__pycache__/",
    "*.pyc",
    ".pytest_cache/",
    ".hive_tmp/",
    ".mypy_cache/",
    "node_modules/",
    ".coverage",
    ".DS_Store",
)


class GitGuardError(RuntimeError):
    """Raised when a snapshot or rollback cannot be performed."""


class GitGuard:
    def __init__(self, workspace: str | Path, store_dir: str | Path, enabled: bool = True) -> None:
        self.workspace = Path(workspace).resolve()
        digest = hashlib.sha1(str(self.workspace).encode()).hexdigest()[:12]
        self.git_dir = Path(store_dir).resolve() / f"{self.workspace.name}-{digest}.git"
        self.enabled = enabled and shutil.which("git") is not None

    @property
    def available(self) -> bool:
        return self.enabled

    def _git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        cmd = [
            "git",
            f"--git-dir={self.git_dir}",
            f"--work-tree={self.workspace}",
            "-c", "user.name=HiveCoder",
            "-c", "user.email=hivecoder@localhost",
            "-c", "commit.gpgsign=false",
            "-c", "core.autocrlf=false",
            "-c", "core.quotepath=false",
            *args,
        ]
        proc = subprocess.run(cmd, cwd=self.workspace, capture_output=True, text=True)
        if check and proc.returncode != 0:
            raise GitGuardError(f"git {' '.join(args)} failed: {proc.stderr.strip() or proc.stdout.strip()}")
        return proc

    def ensure(self) -> bool:
        if not self.enabled:
            return False
        self.workspace.mkdir(parents=True, exist_ok=True)
        if not (self.git_dir / "HEAD").exists():
            self.git_dir.parent.mkdir(parents=True, exist_ok=True)
            self._git("init", "-q")
            self._git("config", "core.bare", "false")
        excludes = list(DEFAULT_EXCLUDES)
        if self.workspace in self.git_dir.parents:
            rel_store = self.git_dir.parent.relative_to(self.workspace).as_posix()
            excludes.append(f"/{rel_store}/")
            mem_parent = self.git_dir.parent.parent
            if self.workspace in mem_parent.parents:
                rel_mem = mem_parent.relative_to(self.workspace).as_posix()
                excludes.append(f"/{rel_mem}/")
        info = self.git_dir / "info"
        info.mkdir(parents=True, exist_ok=True)
        (info / "exclude").write_text("\n".join(excludes) + "\n", encoding="utf-8")
        return True

    def snapshot(self, label: str) -> Optional[str]:
        """Commit the full workspace state; returns the snapshot sha (or None if disabled)."""
        try:
            if not self.ensure():
                return None
            self._git("add", "-A")
            self._git("commit", "-q", "--allow-empty", "--no-verify", "-m", f"hive snapshot: {label}")
            return self._git("rev-parse", "HEAD").stdout.strip()
        except (GitGuardError, OSError):
            return None

    def rollback(self, sha: str) -> bool:
        """Restore the workspace to ``sha`` (tracked files reset, new files removed)."""
        if not sha:
            return False
        try:
            if not self.ensure():
                return False
            self._git("cat-file", "-e", f"{sha}^{{commit}}")
            self._git("add", "-A")
            self._git("reset", "-q", "--hard", sha)
            self._git("clean", "-fdq")
            return True
        except (GitGuardError, OSError):
            return False

    def list_snapshots(self, limit: int = 10) -> list[tuple[str, str]]:
        if not self.enabled or not (self.git_dir / "HEAD").exists():
            return []
        proc = self._git("log", f"-{limit}", "--format=%h%x09%s", check=False)
        if proc.returncode != 0:
            return []
        out: list[tuple[str, str]] = []
        for line in proc.stdout.splitlines():
            sha, _, subject = line.partition("\t")
            out.append((sha, subject))
        return out
