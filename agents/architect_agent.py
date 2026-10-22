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

    def plan(self, goal: str, max_retries: int = 2) -> list[MicroTask]:
        """Decompose ``goal`` into validated :class:`MicroTask` items with parse retry."""
        memory = self.blackboard.get_compact_memory(max_notes=8)
        base_prompt = (
            f"# PROJECT GOAL\n{goal.strip()}\n\n"
            f"{memory}\n\n"
            f"Produce at most {self.max_micro_tasks} atomic micro-tasks in JSON format."
        )
        last_error: Exception | None = None
        prompt = base_prompt
        for attempt in range(max_retries + 1):
            raw, _elapsed = self._invoke_isolated(self.system_prompt, prompt)
            try:
                return self.parse_plan(raw, goal=goal)
            except ValueError as exc:
                last_error = exc
                prompt = (
                    f"{base_prompt}\n\n"
                    f"# PREVIOUS PLAN PARSE ERROR (Attempt {attempt + 1})\n"
                    f"{exc}\n"
                    "Return ONLY a valid JSON object containing a 'micro_tasks' array."
                )
        raise ValueError(str(last_error or "ArchitectAgent failed to produce a valid plan"))

    def parse_plan(self, raw: str, goal: str = "") -> list[MicroTask]:
        """Parse, deduplicate, order, and sanitize the JSON plan produced by the LLM."""
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
            ).strip()
            if raw_target.startswith(("/", "\\")):
                first_part = Path(raw_target.lstrip("/\\")).parts[0] if Path(raw_target.lstrip("/\\")).parts else ""
                if first_part not in {
                    "etc", "usr", "var", "tmp", "bin", "sbin", "dev", "proc",
                    "sys", "root", "Users", "home", "private", "Library", "System", "Windows",
                }:
                    raw_target = raw_target.lstrip("/\\")
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

            if tasks and tasks[-1].target_file == target_file and tasks[-1].title.lower() == title.lower():
                continue

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

        # Ensure implementation tasks execute in dependency order before test tasks
        tasks = self._topological_sort_tasks(tasks)

        # Auto-inject a pytest verification task if Python modules exist without any test micro-task
        has_test = any(mt.kind == "test" for mt in tasks)
        py_code_tasks = [mt for mt in tasks if mt.kind == "code" and mt.target_file.endswith(".py")]
        if not has_test and py_code_tasks and len(tasks) < self.max_micro_tasks:
            primary_stem = Path(py_code_tasks[0].target_file).stem
            test_target = f"tests/test_{primary_stem}.py"
            auto_id = f"step-{len(tasks) + 1}"
            while auto_id in seen_ids:
                auto_id += "-test"
            tasks.append(
                MicroTask(
                    id=auto_id,
                    title=f"Write pytest unit tests in {test_target}",
                    description=(
                        "Write deterministic pytest unit tests covering the public functions "
                        f"and classes implemented in {', '.join(t.target_file for t in py_code_tasks)}."
                    ),
                    target_file=test_target,
                    kind="test",
                    acceptance=["All pytest assertions pass"],
                )
            )

        return tasks

    @staticmethod
    def _topological_sort_tasks(tasks: list[MicroTask]) -> list[MicroTask]:
        """Order code micro-tasks so referenced dependency files are created before their importers."""
        code_tasks = [t for t in tasks if t.kind != "test"]
        test_tasks = [t for t in tasks if t.kind == "test"]
        n = len(code_tasks)
        if n <= 1:
            return code_tasks + test_tasks

        def _mentions_file(src: MicroTask, dep: MicroTask) -> bool:
            dep_name = Path(dep.target_file).name.lower()
            if dep_name == "__init__.py":
                return False
            haystack = f"{src.title} {src.description} {' '.join(src.acceptance)}".lower()
            if dep.target_file.lower() in haystack:
                return True
            return bool(re.search(rf"\b{re.escape(dep_name)}\b", haystack))

        deps: dict[int, set[int]] = {i: set() for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                if code_tasks[i].target_file == code_tasks[j].target_file:
                    deps[j].add(i)

        for i in range(n):
            i_path = Path(code_tasks[i].target_file)
            for j in range(n):
                if i == j or code_tasks[i].target_file == code_tasks[j].target_file:
                    continue
                j_path = Path(code_tasks[j].target_file)
                if (
                    i_path.name == "__init__.py"
                    and j_path.name != "__init__.py"
                    and (j_path.parent == i_path.parent or i_path.parent in j_path.parents)
                    and not _mentions_file(code_tasks[j], code_tasks[i])
                ):
                    deps[i].add(j)
                    continue
                if _mentions_file(code_tasks[i], code_tasks[j]) and not _mentions_file(code_tasks[j], code_tasks[i]):
                    deps[i].add(j)

        ordered: list[MicroTask] = []
        remaining = set(range(n))
        while remaining:
            ready = [idx for idx in sorted(remaining) if not (deps[idx] & remaining)]
            pick = ready[0] if ready else min(remaining)
            ordered.append(code_tasks[pick])
            remaining.remove(pick)
        return ordered + test_tasks
