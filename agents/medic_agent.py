"""MedicAgent: isolated fault repair using traceback output and Blackboard signatures."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from agents.base_agent import AgentResult, BaseAgent, NANO_TAG_INSTRUCTIONS
from core.ast_patcher import ASTPatcher
from core.blackboard import normalize_relpath
from core.context_builder import format_task_spec
from core.llm_client import parse_nano_tags
from core.state_machine import MicroTask
from probes.runtime_probe import RuntimeProbe
from probes.signature_probe import SignatureProbe, format_probe_summary

MEDIC_SYSTEM_PROMPT = """\
Repair the target file that failed inspection or automated validation (`py_compile` / `pytest`).
You are given the broken file, the failure traceback or inspection report, and the BLACKBOARD
AST symbol table.

Constraints:
1. Identify the root cause from the traceback or inspection report.
2. Align any mismatched imports or function calls with the signatures on the BLACKBOARD.
3. Output the complete repaired file using the Nano-Tag format (`<<<FILE_PATH>>>`,
   `<<<CODE>>>`, `<<<RELAY_NOTE>>>`).
4. Include a 1-2 sentence <<<RELAY_NOTE>>> summarizing the fix.
"""


class MedicAgent(BaseAgent):
    role = "MedicAgent"
    system_prompt = MEDIC_SYSTEM_PROMPT

    def select_patient_file(self, task: MicroTask, error_output: str) -> str:
        """Pick which workspace file should be repaired based on the traceback."""
        frame_files: list[str] = []
        for match in re.findall(r"([A-Za-z0-9_./-]+\.py):\d+", error_output):
            try:
                norm = normalize_relpath(self.workspace, match)
            except ValueError:
                continue
            if (self.workspace / norm).is_file():
                frame_files.append(norm)

        # If the deepest frame in a non-AssertionError crash is a workspace module, repair that module
        if frame_files and "AssertionError" not in error_output:
            deepest = frame_files[-1]
            if not Path(deepest).name.startswith("test_"):
                return deepest

        candidates = list(dict.fromkeys([*task.files_written, task.target_file]))
        for rel in candidates:
            if rel in error_output:
                return rel
        if frame_files:
            return frame_files[-1]
        for match in re.findall(r"([A-Za-z0-9_./-]+\.py)", error_output):
            try:
                norm = normalize_relpath(self.workspace, match)
            except ValueError:
                continue
            if (self.workspace / norm).is_file():
                return norm
        return task.target_file

    def _extract_related_traceback_context(self, patient: str, error_output: str) -> str:
        """Extract line windows from other workspace files referenced in ``error_output``."""
        windows: list[str] = []
        seen = {patient}
        for match in re.findall(r"([A-Za-z0-9_./-]+\.py)", error_output):
            try:
                norm = normalize_relpath(self.workspace, match)
            except ValueError:
                continue
            if norm in seen or not (self.workspace / norm).is_file():
                continue
            seen.add(norm)
            other_src = (self.workspace / norm).read_text(encoding="utf-8", errors="replace")
            win = ASTPatcher.extract_traceback_window(other_src, error_output, filepath=norm)
            if win:
                windows.append(f"### {norm}\n```\n{win}\n```")
            elif other_src.strip() and len(other_src) <= 1200:
                windows.append(f"### {norm}\n```\n{other_src.strip()}\n```")
        return "\n\n".join(windows)

    def _detect_missing_package_alert(self, error_output: str) -> str:
        """Return a directive if ``error_output`` references an uninstalled third-party package."""
        matches = re.findall(r"No module named ['\"]([A-Za-z0-9_.-]+)['\"]", error_output)
        if not matches:
            return ""
        probe = RuntimeProbe()
        alerts: list[str] = []
        for raw_mod in dict.fromkeys(matches):
            root_mod = raw_mod.split(".")[0]
            if self.blackboard._module_record(raw_mod) or self.blackboard._module_record(root_mod):
                continue
            if not probe.is_installed(root_mod):
                alerts.append(
                    f"Package '{root_mod}' is NOT installed in this Python environment. "
                    f"Do NOT import '{root_mod}'; replace it with Python standard library modules "
                    "or packages listed under RUNTIME ENVIRONMENT."
                )
        return "\n".join(alerts)

    def repair(
        self,
        task: MicroTask,
        error_output: str,
        broken_file: Optional[str] = None,
        attempt: int = 1,
        temperature: Optional[float] = None,
        loop_breaker_hint: str = "",
    ) -> AgentResult:
        """Run an isolated Infirmary repair session for ``broken_file``."""
        patient = broken_file or self.select_patient_file(task, error_output)
        patient = normalize_relpath(self.workspace, patient)
        abs_path = self.workspace / patient
        current_code = (
            abs_path.read_text(encoding="utf-8", errors="replace")
            if abs_path.is_file()
            else "# (file does not exist yet)\n"
        )

        spec = format_task_spec(task)
        memory = self.blackboard.get_compact_memory(target_files=[patient], max_notes=10)
        naive_chars = self.context_builder.naive_chars() + len(spec)

        line_window = ASTPatcher.extract_traceback_window(current_code, error_output, filepath=patient)
        related_window = self._extract_related_traceback_context(patient, error_output)
        live_sigs = format_probe_summary(SignatureProbe().inspect_traceback(current_code, error_output))
        missing_pkg_alert = self._detect_missing_package_alert(error_output)

        extra_sections: list[str] = []
        if missing_pkg_alert:
            extra_sections.append(f"## MISSING PACKAGE ALERT\n{missing_pkg_alert}")
        if line_window:
            extra_sections.append(f"## FOCUS AREA AROUND ERROR LINE ({patient})\n```\n{line_window}\n```")
        if related_window:
            extra_sections.append(f"## RELATED WORKSPACE FILES IN TRACEBACK\n{related_window}")
        if live_sigs:
            extra_sections.append(live_sigs)
        if loop_breaker_hint.strip():
            extra_sections.append(f"## LOOP-BREAKER DIRECTIVE\n{loop_breaker_hint.strip()}")
        extra_text = ("\n\n".join(extra_sections) + "\n\n") if extra_sections else ""

        user_prompt = (
            f"# INFIRMARY REPAIR (Attempt {attempt})\n"
            f"Patient file: {patient}\n\n"
            f"{spec}\n\n"
            f"## FAILURE TRACEBACK / INSPECTION REPORT\n"
            f"```\n{error_output.strip()[-3500:]}\n```\n\n"
            f"{extra_text}"
            f"{memory}\n\n"
            f"## BROKEN FILE CONTENT ({patient})\n"
            f"<<<CURRENT_FILE>>>\n{current_code}<<<END_CURRENT_FILE>>>\n\n"
            f"{NANO_TAG_INSTRUCTIONS}"
        )

        raw, elapsed = self._invoke_isolated(
            self.system_prompt,
            user_prompt,
            temperature=temperature,
        )
        parsed = parse_nano_tags(raw)
        tokens_saved = max(0, (naive_chars - len(user_prompt)) // 4)
        if not parsed.has_code:
            return AgentResult(
                agent_name=self.role,
                success=False,
                raw_response=raw,
                prompt_chars=len(self.system_prompt) + len(user_prompt),
                response_chars=len(raw),
                tokens_saved=tokens_saved,
                elapsed_seconds=elapsed,
                error="MedicAgent response did not contain a valid <<<CODE>>> block.",
            )

        written, note = self._materialize_nano_response(
            parsed=parsed,
            default_target=patient,
            task_id=task.id,
        )
        return AgentResult(
            agent_name=self.role,
            success=True,
            files_written=written,
            relay_note=note,
            raw_response=raw,
            prompt_chars=len(self.system_prompt) + len(user_prompt),
            response_chars=len(raw),
            tokens_saved=tokens_saved,
            elapsed_seconds=elapsed,
        )
