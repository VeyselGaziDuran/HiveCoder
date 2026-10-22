"""Base class for stateless micro-agents.

Each agent invocation issues an isolated completion request to
:class:`~core.llm_client.LLMClient` without retaining conversation history.
State between steps is shared exclusively through :class:`~core.blackboard.Blackboard`.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from core.ast_patcher import ASTPatcher
from core.blackboard import Blackboard, normalize_relpath
from core.context_builder import ContextBuilder
from core.llm_client import LLMClient, NanoResponse, parse_nano_tags
from core.state_machine import MicroTask
from probes.signature_probe import SignatureProbe, format_probe_summary


@dataclass
class AgentResult:
    agent_name: str
    success: bool
    files_written: list[str] = field(default_factory=list)
    relay_note: str = ""
    raw_response: str = ""
    prompt_chars: int = 0
    response_chars: int = 0
    tokens_saved: int = 0
    elapsed_seconds: float = 0.0
    error: str = ""


NANO_TAG_INSTRUCTIONS = """\
OUTPUT FORMAT (STRICT NANO-TAG PROTOCOL — DO NOT USE JSON FOR CODE):
Return your file path, complete source code, and a 1-2 sentence technical relay note
using these exact block tags:

<<<FILE_PATH>>>
relative/path/to/file.py
<<<END_FILE_PATH>>>
<<<CODE>>>
# Complete, runnable source code here
<<<END_CODE>>>
<<<RELAY_NOTE>>>
Concise 1-2 sentence hand-off note explaining exact function/class names and signatures created or updated for the next agent.
<<<END_RELAY_NOTE>>>

CRITICAL RULES:
- Write COMPLETE, production-ready files. Never leave unfinished stubs or ellipses.
- Match existing function, class, and variable names shown on the BLACKBOARD exactly.
- Do not wrap the response in conversational prose outside the tags.
"""


class BaseAgent:
    """Stateless micro-agent that executes a single atomic step and self-destructs its context."""

    role: str = "base"
    system_prompt: str = "You are a disciplined software engineering micro-agent."

    def __init__(
        self,
        llm: LLMClient,
        blackboard: Blackboard,
        context_builder: ContextBuilder,
        workspace: str | Path,
    ) -> None:
        self.llm = llm
        self.blackboard = blackboard
        self.context_builder = context_builder
        self.workspace = Path(workspace).resolve()

    def _invoke_isolated(
        self,
        system: str,
        user_prompt: str,
        temperature: Optional[float] = None,
    ) -> tuple[str, float]:
        """Execute a single zero-history LLM call and register exchange size."""
        started = time.time()
        if temperature is not None:
            response = self.llm.complete(system=system, user=user_prompt, temperature=temperature)
        else:
            response = self.llm.complete(system=system, user=user_prompt)
        elapsed = time.time() - started
        self.context_builder.register_exchange(len(system) + len(user_prompt), len(response))
        return response, elapsed

    def _materialize_nano_response(
        self,
        parsed: NanoResponse,
        default_target: str,
        task_id: Optional[str] = None,
    ) -> tuple[list[str], str]:
        """Write extracted file blocks atomically to the workspace and update the Blackboard."""
        written: list[str] = []
        relay_note = parsed.relay_note.strip()
        if not relay_note and parsed.files:
            target_display = parsed.files[0].path or default_target
            relay_note = f"{self.role} updated {target_display}."

        default_norm = normalize_relpath(self.workspace, default_target)
        for idx, block in enumerate(parsed.files):
            raw_path = (block.path or default_norm).strip()
            if raw_path.startswith(("/", "\\")):
                first_part = Path(raw_path.lstrip("/\\")).parts[0] if Path(raw_path.lstrip("/\\")).parts else ""
                if first_part not in {
                    "etc", "usr", "var", "tmp", "bin", "sbin", "dev", "proc",
                    "sys", "root", "Users", "home", "private", "Library", "System", "Windows",
                }:
                    raw_path = raw_path.lstrip("/\\")
            try:
                rel_path = normalize_relpath(self.workspace, raw_path)
            except ValueError:
                rel_path = default_norm
            if len(parsed.files) == 1 and rel_path != default_norm:
                same_basename = Path(rel_path).name == Path(default_norm).name
                if (
                    same_basename
                    or not (self.workspace / default_norm).exists()
                    or not (self.workspace / rel_path).exists()
                ):
                    rel_path = default_norm
            abs_path = self.workspace / rel_path
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            final_code = block.code
            if rel_path.endswith(".py"):
                final_code = ASTPatcher.normalize_root_relative_imports(final_code, rel_path)
            if abs_path.is_file() and rel_path.endswith(".py"):
                protected = self.blackboard.get_verified_symbols(rel_path)
                if protected:
                    existing_code = abs_path.read_text(encoding="utf-8", errors="replace")
                    final_code = ASTPatcher.merge_preserving_symbols(
                        existing_source=existing_code,
                        incoming_source=final_code,
                        protected_symbols=protected,
                    )
            tmp_path = abs_path.with_name(f".{abs_path.name}.{os.getpid()}.tmp")
            tmp_path.write_text(final_code, encoding="utf-8")
            os.replace(tmp_path, abs_path)
            written.append(rel_path)
            note_for_file = relay_note if idx == len(parsed.files) - 1 else ""
            self.blackboard.update_file_record(
                filepath=rel_path,
                agent_note=note_for_file,
                agent=self.role,
                task_id=task_id,
            )
        return written, relay_note

    def execute_nano_task(
        self,
        task: MicroTask,
        extra_instructions: str = "",
        extra_files: Optional[list[str]] = None,
    ) -> AgentResult:
        """Build a zero-bloat context for ``task``, run the LLM, and persist output files."""
        bundle = self.context_builder.build(task, extra_files=extra_files)
        user_sections = [bundle.text]
        if task.target_file.endswith(".py"):
            existing_src = self.context_builder.read_file(task.target_file)
            if existing_src:
                live_sigs = format_probe_summary(SignatureProbe().inspect_imports_in_source(existing_src))
                if live_sigs:
                    user_sections.append(live_sigs)
        if extra_instructions.strip():
            user_sections.append(extra_instructions.strip())
        user_sections.append(NANO_TAG_INSTRUCTIONS)
        user_prompt = "\n\n".join(user_sections)

        raw, elapsed = self._invoke_isolated(self.system_prompt, user_prompt)
        parsed = parse_nano_tags(raw)
        if not parsed.has_code:
            return AgentResult(
                agent_name=self.role,
                success=False,
                raw_response=raw,
                prompt_chars=len(self.system_prompt) + len(user_prompt),
                response_chars=len(raw),
                tokens_saved=bundle.tokens_saved,
                elapsed_seconds=elapsed,
                error="Model response did not contain any valid <<<CODE>>> block.",
            )

        written, note = self._materialize_nano_response(
            parsed=parsed,
            default_target=task.target_file,
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
            tokens_saved=bundle.tokens_saved,
            elapsed_seconds=elapsed,
        )
