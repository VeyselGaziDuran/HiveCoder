"""TesterAgent: generates pytest suites aligned with Blackboard AST signatures."""

from __future__ import annotations

from agents.base_agent import AgentResult, BaseAgent
from core.state_machine import MicroTask

TESTER_SYSTEM_PROMPT = """\
Write a deterministic `pytest` test module for the current MICRO-TASK using the
function and class signatures listed on the BLACKBOARD and in the RELAY NOTES.

Constraints:
1. Import only the exact symbol names published on the BLACKBOARD for workspace modules.
2. Cover normal execution paths, boundary cases, and error conditions with plain `pytest` assertions.
3. Keep tests self-contained (use `tmp_path` or `monkeypatch` for filesystem state).
4. Emit the complete test file and a concise <<<RELAY_NOTE>>> using the Nano-Tag format.
"""


class TesterAgent(BaseAgent):
    role = "TesterAgent"
    system_prompt = TESTER_SYSTEM_PROMPT

    def run(self, task: MicroTask, feedback: str = "") -> AgentResult:
        extra_lines = [
            "Ensure every imported symbol matches the BLACKBOARD AST signatures verbatim."
        ]
        if feedback.strip():
            extra_lines.append(f"# PREVIOUS TEST FEEDBACK\n{feedback.strip()}")
        return self.execute_nano_task(task, extra_instructions="\n\n".join(extra_lines))
