"""InspectorAgent: static AST/Blackboard verification and optional LLM code review."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from agents.base_agent import BaseAgent
from core.blackboard import Blackboard
from core.context_builder import ContextBuilder, _middle_truncate, format_task_spec
from core.llm_client import LLMClient, LLMParseError, extract_json
from core.state_machine import MicroTask

INSPECTOR_SYSTEM_PROMPT = """\
Review the target file against:
1. The MICRO-TASK description and acceptance criteria.
2. The BLACKBOARD AST signatures and RELAY NOTES (verify imported and exported symbol names match).
3. Complete implementation without unfinished stubs.

Respond ONLY with a JSON object:
{
  "approved": true,
  "issues": [],
  "summary": "One sentence review summary."
}
If rejecting the file, set "approved": false and list specific defects in "issues".
"""


@dataclass
class InspectionVerdict:
    approved: bool
    issues: list[str] = field(default_factory=list)
    summary: str = ""
    deterministic_passed: bool = True

    def format_feedback(self) -> str:
        if self.approved:
            return self.summary or "Inspection approved."
        bullet_list = "\n".join(f"- {issue}" for issue in self.issues)
        header = self.summary or "Inspection rejected the file:"
        return f"{header}\n{bullet_list}".strip()


class InspectorAgent(BaseAgent):
    role = "InspectorAgent"
    system_prompt = INSPECTOR_SYSTEM_PROMPT

    def __init__(
        self,
        llm: LLMClient,
        blackboard: Blackboard,
        context_builder: ContextBuilder,
        workspace: str | Path,
        enable_llm_review: bool = True,
    ) -> None:
        super().__init__(llm, blackboard, context_builder, workspace)
        self.enable_llm_review = enable_llm_review

    def inspect(self, task: MicroTask, files_written: Optional[list[str]] = None) -> InspectionVerdict:
        """Run deterministic AST/Blackboard checks first, then optional LLM review."""
        targets = list(files_written or task.files_written or [task.target_file])
        deterministic_issues: list[str] = []

        for rel in targets:
            abs_path = self.workspace / rel
            if not abs_path.exists():
                deterministic_issues.append(f"{rel}: file was not created.")
                continue
            content = abs_path.read_text(encoding="utf-8", errors="replace")
            if not content.strip():
                deterministic_issues.append(f"{rel}: file is empty.")
                continue

            record = self.blackboard.files.get(rel)
            if record is None:
                record = self.blackboard.update_file_record(rel, agent_note="", agent=self.role, task_id=task.id)

            summary = record.get("summary", {})
            if record.get("kind") == "python" and summary.get("syntax_error"):
                deterministic_issues.append(f"{rel}: syntax error ({summary['syntax_error']}).")
            if record.get("kind") == "json" and summary.get("error"):
                deterministic_issues.append(f"{rel}: {summary['error']}.")

            for stub in self.blackboard.find_stub_markers(rel):
                deterministic_issues.append(str(stub))
            for mismatch in self.blackboard.check_import_consistency(rel):
                deterministic_issues.append(str(mismatch))

        if not task.relay_note.strip():
            deterministic_issues.append("Agent did not leave a Relay Note for the next agent.")

        if deterministic_issues:
            return InspectionVerdict(
                approved=False,
                issues=deterministic_issues,
                summary="Deterministic Blackboard/AST inspection failed.",
                deterministic_passed=False,
            )

        if not self.enable_llm_review:
            return InspectionVerdict(
                approved=True,
                issues=[],
                summary="Deterministic Blackboard/AST inspection approved.",
            )

        return self._llm_review(task, targets)

    def _llm_review(self, task: MicroTask, targets: list[str]) -> InspectionVerdict:
        spec = format_task_spec(task)
        memory = self.blackboard.get_compact_memory(target_files=targets, max_notes=8)
        per_file_budget = max(600, int(self.context_builder.max_chars * 0.5) // max(1, len(targets)))
        snippets: list[str] = []
        for rel in targets:
            text = (self.workspace / rel).read_text(encoding="utf-8", errors="replace")
            if len(text) > per_file_budget:
                text = _middle_truncate(text, per_file_budget)
            lang = Path(rel).suffix.lstrip(".") or "text"
            snippets.append(f"### FILE: {rel}\n```{lang}\n{text}\n```")
        user_prompt = (
            f"{spec}\n\n"
            f"Relay note left by agent: {task.relay_note}\n\n"
            f"{memory}\n\n"
            + "\n\n".join(snippets)
            + "\n\nReturn your JSON verdict."
        )
        raw, _elapsed = self._invoke_isolated(self.system_prompt, user_prompt)
        try:
            data = extract_json(raw, expect=dict)
        except LLMParseError:
            return InspectionVerdict(
                approved=True,
                issues=[],
                summary="Deterministic checks passed (unstructured inspector reply ignored).",
            )

        approved = bool(data.get("approved", True))
        raw_issues = data.get("issues") or []
        issues = [str(i).strip() for i in raw_issues if str(i).strip()] if isinstance(raw_issues, list) else []
        summary = str(data.get("summary") or ("Approved." if approved else "Rejected by InspectorAgent.")).strip()
        if not approved and not issues:
            issues = [summary]
        return InspectionVerdict(approved=approved, issues=issues, summary=summary)
