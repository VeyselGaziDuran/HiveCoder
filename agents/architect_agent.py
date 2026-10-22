"""ArchitectAgent: decomposes a task specification into ordered single-file micro-tasks."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from agents.base_agent import BaseAgent
from core.blackboard import Blackboard, normalize_relpath
from core.context_builder import ContextBuilder
from core.llm_client import LLMClient, LLMParseError, extract_json
from core.state_machine import MicroTask
from core.validator import is_test_file

ARCHITECT_SYSTEM_PROMPT = """\
Decompose the user's software engineering task into an ordered queue of atomic,
single-file micro-tasks.

Return ONLY a JSON object with a "micro_tasks" array:
{
  "micro_tasks": [
    {
      "id": "step-1",
      "title": "Short imperative title",
      "target_file": "relative/path.py",
      "kind": "code",
      "description": "Exact responsibilities, functions, classes, and signatures to implement.",
      "acceptance": ["Verifiable criterion 1", "Verifiable criterion 2"]
    },
    {
      "id": "step-2",
      "title": "Write pytest suite for relative/path.py",
      "target_file": "tests/test_path.py",
      "kind": "test",
      "description": "Unit tests covering the public functions and classes.",
      "acceptance": ["All pytest assertions pass"]
    }
  ]
}

Constraints:
1. Order tasks so dependencies are created before modules that import them.
2. Include at least one "test" micro-task targeting "tests/test_*.py" at the end.
3. Specify exact function and class names in "description" so downstream steps use consistent identifiers.
4. Keep the plan between 2 and 8 micro-tasks unless the specification requires more.
"""


class ArchitectAgent(BaseAgent):
    role = "ArchitectAgent"
    system_prompt = ARCHITECT_SYSTEM_PROMPT

    def __init__(
        self,
        llm: LLMClient,
        blackboard: Blackboard,
        context_builder: ContextBuilder,
        workspace: str | Path,
        max_micro_tasks: int = 30,
    ) -> None:
        super().__init__(llm, blackboard, context_builder, workspace)
        self.max_micro_tasks = max_micro_tasks

    def plan(self, goal: str) -> list[MicroTask]:
        """Decompose ``goal`` into validated :class:`MicroTask` items."""
        memory = self.blackboard.get_compact_memory(max_notes=8)
        user_prompt = (
            f"# PROJECT GOAL\n{goal.strip()}\n\n"
            f"{memory}\n\n"
            f"Produce at most {self.max_micro_tasks} atomic micro-tasks in JSON format."
        )
        raw, _elapsed = self._invoke_isolated(self.system_prompt, user_prompt)
        return self.parse_plan(raw, goal=goal)

    def parse_plan(self, raw: str, goal: str = "") -> list[MicroTask]:
        """Parse and sanitize the JSON plan produced by the LLM."""
        try:
            payload = extract_json(raw)
        except LLMParseError as exc:
            raise ValueError(f"ArchitectAgent could not parse JSON plan: {exc}") from exc

        items: list[Any]
        if isinstance(payload, dict):
            raw_list = (
                payload.get("micro_tasks")
                or payload.get("tasks")
                or payload.get("steps")
                or payload.get("plan")
            )
            if not isinstance(raw_list, list):
                raise ValueError("Architect JSON must contain a 'micro_tasks' list")
            items = raw_list
        elif isinstance(payload, list):
            items = payload
        else:
            raise ValueError("Architect JSON must be an object or array")

        tasks: list[MicroTask] = []
        seen_ids: set[str] = set()
        for idx, item in enumerate(items[: self.max_micro_tasks], start=1):
            if not isinstance(item, dict):
                continue
            raw_target = str(
                item.get("target_file")
                or item.get("file")
                or item.get("path")
                or f"module_{idx}.py"
            )
            target_file = normalize_relpath(self.workspace, raw_target)
            raw_id = str(item.get("id") or f"step-{idx}").strip()
            raw_id = re.sub(r"[^A-Za-z0-9_-]+", "-", raw_id).strip("-") or f"step-{idx}"
            if raw_id in seen_ids:
                raw_id = f"{raw_id}-{idx}"
            seen_ids.add(raw_id)

            kind_raw = str(item.get("kind") or ("test" if is_test_file(target_file) else "code")).lower()
            kind = "test" if kind_raw == "test" or is_test_file(target_file) else "code"

            title = str(item.get("title") or f"Implement {target_file}").strip()
            description = str(item.get("description") or goal or title).strip()
            acc_raw = item.get("acceptance") or item.get("acceptance_criteria") or []
            if isinstance(acc_raw, str):
                acceptance = [acc_raw]
            elif isinstance(acc_raw, list):
                acceptance = [str(a).strip() for a in acc_raw if str(a).strip()]
            else:
                acceptance = []

            tasks.append(
                MicroTask(
                    id=raw_id,
                    title=title,
                    description=description,
                    target_file=target_file,
                    kind=kind,
                    acceptance=acceptance,
                )
            )

        if not tasks:
            raise ValueError("ArchitectAgent produced an empty plan")
        return tasks
