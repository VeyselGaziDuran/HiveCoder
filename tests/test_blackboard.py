"""Unit tests for core/blackboard.py: AST extraction, Relay Notes, and import consistency."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.blackboard import (
    Blackboard,
    UnsafePathError,
    extract_python_summary,
    normalize_relpath,
)


def test_extract_python_summary_classes_functions_and_globals() -> None:
    source = '''"""Vocabulary engine module."""

import json
from pathlib import Path
from typing import Optional

DEFAULT_LIMIT: int = 50
CACHE_NAME = "vocab.json"


def get_words(path: str, limit: int = DEFAULT_LIMIT) -> list[dict]:
    return []


async def fetch_remote(url: str) -> Optional[dict]:
    return None


class WordStore:
    version: int = 1

    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath

    @property
    def count(self) -> int:
        return 0
'''
    summary = extract_python_summary(source)
    assert summary["syntax_error"] is None
    assert summary["docstring"] == "Vocabulary engine module."
    assert "json" in summary["imports"]
    assert "from pathlib import Path" in summary["imports"]
    assert any("DEFAULT_LIMIT: int = 50" in g for g in summary["globals"])
    assert any("CACHE_NAME = 'vocab.json'" in g or 'CACHE_NAME = "vocab.json"' in g for g in summary["globals"])
    assert "def get_words(path: str, limit: int=DEFAULT_LIMIT) -> list[dict]" in summary["functions"]
    assert "async def fetch_remote(url: str) -> Optional[dict]" in summary["functions"]
    assert len(summary["classes"]) == 1
    cls = summary["classes"][0]
    assert cls["name"] == "WordStore"
    assert "version: int" in cls["attributes"]
    assert "def __init__(self, filepath: Path) -> None" in cls["methods"]
    assert "@property def count(self) -> int" in cls["methods"]


def test_update_file_record_and_relay_notes_persistence(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    memory_dir = tmp_path / "memory"
    workspace.mkdir()

    app_py = workspace / "app.py"
    app_py.write_text(
        "def get_words() -> list[dict]:\n"
        "    return [{'en': 'hive', 'tr': 'kovan'}]\n",
        encoding="utf-8",
    )
    vocab_json = workspace / "vocab.json"
    vocab_json.write_text(
        json.dumps([{"en": "hive", "tr": "kovan"}]),
        encoding="utf-8",
    )

    bb = Blackboard(memory_dir=memory_dir, workspace=workspace)
    note_1 = "vocab.json okuma fonksiyonunu app.py icinde get_words() adiyla yazdim, donus tipi list[dict]."
    rec = bb.update_file_record("app.py", agent_note=note_1, agent="CoderAgent", task_id="step-1")
    bb.update_file_record("vocab.json", agent_note="Added seed vocabulary list.", agent="CoderAgent", task_id="step-1")

    assert rec["kind"] == "python"
    assert "def get_words() -> list[dict]" in rec["summary"]["functions"]
    assert bb.json_path.is_file()
    assert bb.notes_path.is_file()

    persisted = json.loads(bb.json_path.read_text(encoding="utf-8"))
    assert "app.py" in persisted["files"]
    assert "vocab.json" in persisted["files"]
    assert len(persisted["notes"]) == 2

    notes_md = bb.notes_path.read_text(encoding="utf-8")
    assert "get_words()" in notes_md
    assert "CoderAgent" in notes_md

    compact = bb.get_compact_memory(target_files=["app.py"])
    assert "def get_words() -> list[dict]" in compact
    assert note_1 in compact
    assert "list[1] of {en: str, tr: str}" in compact


def test_check_import_consistency_catches_wrong_symbol_name(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    memory_dir = tmp_path / "memory"
    workspace.mkdir()
    (workspace / "tests").mkdir()

    (workspace / "app.py").write_text(
        "def get_words() -> list[dict]:\n    return []\n",
        encoding="utf-8",
    )
    (workspace / "tests" / "test_app.py").write_text(
        "from app import get_flashcards\n\n"
        "def test_items() -> None:\n"
        "    assert get_flashcards() == []\n",
        encoding="utf-8",
    )

    bb = Blackboard(memory_dir=memory_dir, workspace=workspace)
    bb.update_file_record("app.py", agent_note="Defined get_words() -> list[dict].")
    bb.update_file_record("tests/test_app.py", agent_note="Created test file.")

    issues = bb.check_import_consistency("tests/test_app.py")
    assert len(issues) == 1
    assert "get_flashcards" in issues[0].message
    assert "get_words" in issues[0].message


def test_normalize_relpath_blocks_workspace_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    assert normalize_relpath(workspace, "./pkg/mod.py") == "pkg/mod.py"
    with pytest.raises(UnsafePathError):
        normalize_relpath(workspace, "../outside.py")
    with pytest.raises(UnsafePathError):
        normalize_relpath(workspace, "/etc/hosts")
    with pytest.raises(UnsafePathError):
        normalize_relpath(workspace, "venv/lib/site.py")
