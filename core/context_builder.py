"""Zero-bloat context builder.

An agent's prompt is assembled from exactly three ingredients:

1. the micro-task spec,
2. the Blackboard compact memory (AST map + relay notes),
3. the *current* content of the target file (if it exists).

Nothing else - no chat history, no unrelated files.  If the budget is
exceeded, other files collapse to name lists, then older notes are dropped,
and finally the target file is middle-truncated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from core.blackboard import Blackboard
from core.state_machine import MicroTask


@dataclass
class ContextBundle:
    text: str
    chars: int
    naive_chars: int

    @property
    def tokens(self) -> int:
        return max(1, self.chars // 4)

    @property
    def tokens_saved(self) -> int:
        return max(0, (self.naive_chars - self.chars) // 4)


def _middle_truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = max(200, limit // 2 - 40)
    return text[:half] + "\n# ... (middle of file truncated by HiveCoder context budget) ...\n" + text[-half:]


def format_task_spec(task: MicroTask) -> str:
    lines = [
        f"# MICRO-TASK {task.id}: {task.title}",
        f"Target file: {task.target_file}",
        f"Kind: {task.kind}",
    ]
    if task.description:
        lines.append(f"Description: {task.description}")
    if task.acceptance:
        lines.append("Acceptance criteria:")
        lines.extend(f"  - {item}" for item in task.acceptance)
    return "\n".join(lines)


class ContextBuilder:
    def __init__(self, blackboard: Blackboard, workspace: str | Path, max_chars: int = 14000, max_notes: int = 12) -> None:
        self.blackboard = blackboard
        self.workspace = Path(workspace).resolve()
        self.max_chars = max_chars
        self.max_notes = max_notes
        self.history_chars = 0

    def read_file(self, rel_path: str) -> Optional[str]:
        path = self.workspace / rel_path
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8", errors="replace")

    def naive_chars(self) -> int:
        """What a chat-history agent would carry: every file + every past exchange."""
        return self.blackboard.workspace_chars() + self.history_chars

    def register_exchange(self, prompt_chars: int, response_chars: int) -> None:
        self.history_chars += prompt_chars + response_chars

    def _memory(self, targets: list[str], budget: int) -> str:
        memory = self.blackboard.get_compact_memory(target_files=targets, max_notes=self.max_notes)
        if len(memory) <= budget:
            return memory
        memory = self.blackboard.get_compact_memory(
            target_files=targets, max_notes=self.max_notes, names_only_for_others=True
        )
        notes = self.max_notes
        while len(memory) > budget and notes > 2:
            notes = max(2, notes // 2)
            memory = self.blackboard.get_compact_memory(
                target_files=targets, max_notes=notes, names_only_for_others=True
            )
        return memory if len(memory) <= budget else memory[: max(0, budget - 40)] + "\n...(memory truncated)"

    def build(self, task: MicroTask, extra_files: Optional[list[str]] = None) -> ContextBundle:
        spec = format_task_spec(task)
        targets = list(dict.fromkeys([task.target_file, *(extra_files or [])]))
        file_budget = int(self.max_chars * 0.5)
        per_file_budget = max(400, file_budget // max(1, len(targets)) - 120)
        file_sections: list[str] = []
        for rel in targets:
            content = self.read_file(rel)
            if content is None:
                file_sections.append(f"# CURRENT CONTENT OF {rel}\n(file does not exist yet: create it)")
            else:
                if len(content) > per_file_budget:
                    content = _middle_truncate(content, per_file_budget)
                file_sections.append(f"# CURRENT CONTENT OF {rel}\n<<<CURRENT_FILE>>>\n{content}<<<END_CURRENT_FILE>>>")
        files_text = "\n\n".join(file_sections)
        memory_budget = max(800, self.max_chars - len(spec) - len(files_text) - 200)
        memory = self._memory(targets, memory_budget)
        text = f"{spec}\n\n{memory}\n\n{files_text}\n"
        return ContextBundle(text=text, chars=len(text), naive_chars=self.naive_chars() + len(spec))
