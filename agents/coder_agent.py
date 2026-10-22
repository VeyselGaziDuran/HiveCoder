"""CoderAgent: implements a single code micro-task and records a relay note."""

from __future__ import annotations

from agents.base_agent import AgentResult, BaseAgent
from core.state_machine import MicroTask

CODER_SYSTEM_PROMPT = """\
Implement the target module specified in the current MICRO-TASK.
Project context is provided via the BLACKBOARD (AST symbol signatures and relay notes)
and the current content of the target file.

Constraints:
1. Implement only the target file requested in the MICRO-TASK.
2. Preserve all function names, parameter lists, return types, and class names
   already published on the BLACKBOARD or specified in the task description.
3. Emit complete, type-annotated, runnable code without unfinished stubs.
4. Provide a 1-2 sentence <<<RELAY_NOTE>>> documenting the exact symbol names and
   signatures defined or updated for downstream tasks.
"""


class CoderAgent(BaseAgent):
    role = "CoderAgent"
    system_prompt = CODER_SYSTEM_PROMPT

    def run(self, task: MicroTask, feedback: str = "") -> AgentResult:
        extra = ""
        if feedback.strip():
            extra = (
                "# PREVIOUS ATTEMPT FEEDBACK (FIX THESE ISSUES)\n"
                f"{feedback.strip()}"
            )
        return self.execute_nano_task(task, extra_instructions=extra)
