"""Ephemeral memory and relay-note lifecycle management.

Relay notes exist to pass symbol signatures and intent from one micro-agent to
the next during an active task.  They should never accumulate as permanent bloat:

1. When a micro-task is interrupted, rolled back, or resumed, any note produced
   by an unfinished micro-task is purged immediately.
2. When a task completes (100% verified), all temporary relay notes, active
   ``state.json`` files, isolated git snapshot repositories, and workspace caches
   are cleaned automatically so only the finished application remains.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from core.blackboard import Blackboard
from core.garbage_collector import GarbageCollector
from core.git_guard import GitGuard
from core.state_machine import StateMachine


@dataclass
class CleanupSummary:
    purged_notes: int = 0
    removed_stale_files: int = 0
    state_file_removed: bool = False
    snapshot_repo_removed: bool = False
    gc_items_removed: int = 0
    details: list[str] = field(default_factory=list)


class MemoryLifecycle:
    """Keeps Blackboard relay notes and runtime artifacts clean across task states."""

    def __init__(
        self,
        flush_notes_on_complete: bool = True,
        remove_state_on_complete: bool = True,
        remove_snapshots_on_complete: bool = True,
    ) -> None:
        self.flush_notes_on_complete = flush_notes_on_complete
        self.remove_state_on_complete = remove_state_on_complete
        self.remove_snapshots_on_complete = remove_snapshots_on_complete

    def purge_unverified_notes(
        self,
        blackboard: Blackboard,
        verified_task_ids: Iterable[str],
    ) -> int:
        """Delete relay notes that belong to incomplete, failed, or rolled-back micro-tasks."""
        verified = set(verified_task_ids)
        original_count = len(blackboard.notes)
        kept = [
            entry
            for entry in blackboard.notes
            if entry.get("task_id") in verified
        ]
        purged = original_count - len(kept)
        blackboard.data["notes"] = kept

        for rel, record in list(blackboard.files.items()):
            ws = blackboard.workspace
            if ws is not None and not (ws / rel).exists():
                del blackboard.files[rel]
                blackboard.verified_symbols.pop(rel, None)
                continue
            matching_notes = [n["note"] for n in kept if n.get("file") == rel]
            record["last_note"] = matching_notes[-1] if matching_notes else ""

        self._rewrite_notes_markdown(blackboard)
        blackboard.save()
        return purged

    def clear_all_relay_notes(self, blackboard: Blackboard) -> int:
        """Remove all temporary relay notes while keeping indexed file AST records synchronized."""
        count = len(blackboard.notes)
        blackboard.data["notes"] = []
        for record in blackboard.files.values():
            record["last_note"] = ""
        self._rewrite_notes_markdown(blackboard)
        blackboard.save()
        return count

    def finalize_completed_task(
        self,
        blackboard: Blackboard,
        state_machine: StateMachine,
        git_guard: Optional[GitGuard] = None,
        gc: Optional[GarbageCollector] = None,
        workspace: Optional[str | Path] = None,
    ) -> CleanupSummary:
        """Perform full post-task cleanup once all micro-tasks have passed verification."""
        summary = CleanupSummary()

        if workspace is not None and blackboard.workspace is not None:
            before_files = len(blackboard.files)
            blackboard.scan_workspace(agent="final_sync")
            summary.removed_stale_files = max(0, before_files - len(blackboard.files))

        if self.flush_notes_on_complete:
            summary.purged_notes = self.clear_all_relay_notes(blackboard)
            summary.details.append(f"purged {summary.purged_notes} ephemeral relay note(s)")

        if self.remove_state_on_complete and state_machine.state_file.exists():
            state_machine.state_file.unlink()
            summary.state_file_removed = True
            summary.details.append("removed completed state.json")

        if self.remove_snapshots_on_complete and git_guard is not None and git_guard.git_dir.exists():
            shutil.rmtree(git_guard.git_dir, ignore_errors=True)
            summary.snapshot_repo_removed = True
            summary.details.append("removed temporary snapshot store")

        if gc is not None and workspace is not None:
            rep = gc.collect(workspace)
            summary.gc_items_removed = rep.count

        return summary

    @staticmethod
    def _rewrite_notes_markdown(blackboard: Blackboard) -> None:
        ws = blackboard.data.get("workspace")
        lines = [
            "# HiveCoder Relay Notes",
            "",
            "Cumulative hand-off log written by micro-agents for the next micro-agent.",
            f"Workspace: `{ws}`",
        ]
        for entry in blackboard.notes:
            task_suffix = f" ({entry['task_id']})" if entry.get("task_id") else ""
            lines.extend(
                [
                    "",
                    f"### [{entry['ts']}] {entry['agent']} -> `{entry['file']}`{task_suffix}",
                    "",
                    entry["note"],
                ]
            )
        blackboard.notes_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
