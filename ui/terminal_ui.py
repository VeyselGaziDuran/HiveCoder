"""Terminal UI (TUI) and interactive shell for HiveCoder."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from rich.box import ROUNDED
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from core.config import HIVE_ROOT, active_model
from core.orchestrator import HiveOrchestrator
from core.state_machine import NoActiveTaskError, TaskState

EVENT_STYLES: dict[str, tuple[str, str]] = {
    "planning": ("ARCHITECT", "bold magenta"),
    "plan_ready": ("PLAN", "bold cyan"),
    "micro_start": ("AGENT", "bold blue"),
    "relay_note": ("BLACKBOARD", "bold yellow"),
    "inspection_approved": ("INSPECTOR", "bold green"),
    "inspection_rejected": ("INSPECTOR", "bold red"),
    "inspection_override": ("INSPECTOR", "bold bright_green"),
    "validation_failed": ("VALIDATOR", "bold red"),
    "infirmary": ("INFIRMARY", "bold bright_red"),
    "infirmary_healed": ("MEDIC", "bold bright_green"),
    "loop_breaker": ("LOOP-BREAKER", "bold yellow"),
    "memory_gc": ("MEMORY-GC", "bold cyan"),
    "retry": ("RETRY", "bold yellow"),
    "paused": ("PAUSED", "bold yellow"),
    "micro_done": ("VERIFIED", "bold green"),
    "thermal": ("THERMAL", "bold cyan"),
    "rollback": ("GIT-GUARD", "bold yellow"),
    "task_completed": ("COMPLETE", "bold bright_green"),
    "task_failed": ("FAILED", "bold red"),
    "error": ("ERROR", "bold red"),
}


class TerminalUI:
    """Interactive shell and live Rich event renderer for :class:`HiveOrchestrator`."""

    def __init__(
        self,
        orchestrator: HiveOrchestrator,
        workspace: Optional[str | Path] = None,
        console: Optional[Console] = None,
    ) -> None:
        self.console = console or Console()
        self.orchestrator = orchestrator
        self.orchestrator.console = self.console
        self.orchestrator.thermal.console = self.console
        self.orchestrator.thermal.show_progress = True
        self.orchestrator.on_event = self.handle_event
        self.workspace = Path(
            workspace or orchestrator.config.get("default_workspace", "workspace")
        ).resolve()

    # ------------------------------------------------------------------ rendering
    def render_banner(self) -> None:
        cfg = self.orchestrator.config
        provider = cfg.get("provider", "ollama")
        model = active_model(cfg)
        prof = self.orchestrator.thermal.profile
        healthy, health_msg = self.orchestrator.llm.health()
        health_badge = f"[green]ONLINE ({health_msg})[/]" if healthy else f"[yellow]OFFLINE ({health_msg})[/]"

        grid = Table.grid(expand=True, padding=(0, 2))
        grid.add_column(justify="left")
        grid.add_column(justify="right")
        grid.add_row(
            "[bold bright_cyan]HIVECODER v2.0[/] — Stateless Micro-Agent Orchestrator",
            health_badge,
        )
        grid.add_row(
            f"[dim]Provider:[/] [bold]{provider}[/]   [dim]Model:[/] [bold]{model}[/]   "
            f"[dim]Workspace:[/] [bold]{self.workspace}[/]",
            f"[dim]Thermal:[/] [bold cyan]{prof.name.upper()}[/] "
            f"(pause {int(prof.micro_pause_seconds)}s | cooldown {int(prof.cooldown_seconds)}s)",
        )
        self.console.print(Panel(grid, box=ROUNDED, border_style="bright_cyan"))
        self.console.print(
            "[dim]Slash commands:[/] [bold]/run <goal>[/] | [bold]/resume[/] | [bold]/blackboard[/] | "
            "[bold]/thermal <adaptive|eco|balanced|turbo> [min][/] | [bold]/provider <ollama|api>[/] | "
            "[bold]/model <name>[/] | [bold]/status[/] | [bold]/clean[/] | [bold]/self-test[/] | [bold]/exit[/]\n"
        )

    def handle_event(self, kind: str, message: str, data: dict[str, Any]) -> None:
        tag, style = EVENT_STYLES.get(kind, (kind.upper(), "bold white"))
        if kind == "plan_ready" and "micro_tasks" in data:
            self.console.print(f"[{style}][{tag}][/] {message}")
            self.render_plan_table(data["micro_tasks"])
            return
        if kind == "relay_note":
            self.console.print(
                Panel(
                    Text(message),
                    title=f"[{style}]Relay Note -> Blackboard[/]",
                    border_style="yellow",
                    box=ROUNDED,
                )
            )
            return
        self.console.print(f"[{style}][{tag}][/] {message}")

    def render_plan_table(self, micro_tasks: list[dict[str, Any]]) -> None:
        table = Table(title="Micro-Task Queue", box=ROUNDED, header_style="bold cyan")
        table.add_column("ID", style="bold")
        table.add_column("Kind", style="magenta")
        table.add_column("Target File", style="green")
        table.add_column("Title")
        for mt in micro_tasks:
            table.add_row(
                str(mt.get("id", "")),
                str(mt.get("kind", "code")),
                str(mt.get("target_file", "")),
                str(mt.get("title", "")),
            )
        self.console.print(table)

    def render_blackboard(self) -> None:
        if self.orchestrator.blackboard.workspace is None:
            self.orchestrator.blackboard.bind_workspace(self.workspace, reset_if_changed=False)
            self.orchestrator.blackboard.scan_workspace()
        compact = self.orchestrator.blackboard.get_compact_memory(max_notes=20)
        self.console.print(
            Panel(
                Markdown(compact),
                title="[bold yellow]Blackboard AST Map & Relay Notes[/]",
                border_style="yellow",
                box=ROUNDED,
            )
        )

    def render_status(self) -> None:
        state = self.orchestrator.state_machine.state or self.orchestrator.state_machine.load()
        totals = self.orchestrator.history.totals()
        thermal = self.orchestrator.thermal.status()
        prof = thermal["profile"]

        table = Table(title="HiveCoder System & Task Status", box=ROUNDED, header_style="bold bright_cyan")
        table.add_column("Metric", style="bold")
        table.add_column("Value")

        table.add_row("Provider / Model", f"{self.orchestrator.config.get('provider')} / {active_model(self.orchestrator.config)}")
        table.add_row("Workspace", str(self.workspace))
        table.add_row(
            "Thermal Profile",
            f"{prof['name']} (micro-pause {prof['micro_pause_seconds']}s, "
            f"every {prof['cooldown_every_tasks']} tasks -> {prof['cooldown_seconds']}s cooldown)",
        )
        table.add_row("Cooldowns Triggered", str(thermal["total_cooldowns"]))
        table.add_row("Blackboard Indexed Files", str(len(self.orchestrator.blackboard.files)))
        table.add_row("Relay Notes Recorded", str(len(self.orchestrator.blackboard.notes)))

        if state is not None:
            counts = state.counts()
            table.add_row("Active Task ID", state.task_id)
            table.add_row("Active Goal", state.goal)
            table.add_row("Current Phase", state.phase.value.upper())
            table.add_row(
                "Micro-Tasks",
                f"{counts['done']}/{counts['total']} done "
                f"(pending: {counts['pending']}, in_progress: {counts['in_progress']}, failed: {counts['failed']})",
            )
            table.add_row(
                "Infirmary (Revir) Stats",
                f"{state.metrics.medic_runs} run(s), {state.metrics.medic_successes} healed, "
                f"{state.metrics.inspector_rejections} inspector rejection(s)",
            )
            table.add_row("Estimated Tokens Saved", str(state.metrics.tokens_saved))
            table.add_row("GC Junk Items Cleaned", str(state.metrics.gc_items_removed))
        else:
            table.add_row("Active Task", "None (idle)")

        table.add_row(
            "Historical Totals",
            f"{totals['completed']}/{totals['tasks']} tasks completed | "
            f"{totals['tokens_saved']} tokens saved | {totals['medic_runs']} medic runs",
        )
        self.console.print(table)

        if state and state.micro_tasks:
            mt_table = Table(title="Active Micro-Task Progress", box=ROUNDED)
            mt_table.add_column("ID", style="bold")
            mt_table.add_column("Status")
            mt_table.add_column("Target File", style="green")
            mt_table.add_column("Medic", justify="right")
            mt_table.add_column("Relay Note")
            for mt in state.micro_tasks:
                badge = {
                    "done": "[green]DONE[/]",
                    "in_progress": "[yellow]IN_PROGRESS[/]",
                    "pending": "[dim]PENDING[/]",
                    "failed": "[red]FAILED[/]",
                }.get(mt.status, mt.status)
                mt_table.add_row(
                    mt.id,
                    badge,
                    mt.target_file,
                    str(mt.medic_attempts),
                    (mt.relay_note[:70] + "...") if len(mt.relay_note) > 70 else mt.relay_note,
                )
            self.console.print(mt_table)

    # ------------------------------------------------------------------ commands
    def dispatch_command(self, raw_line: str) -> bool:
        """Execute a single REPL line or slash command. Returns False on ``/exit``."""
        line = raw_line.strip()
        if not line:
            return True
        if not line.startswith("/"):
            self._run_goal(line)
            return True

        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("/exit", "/quit"):
            self.console.print("[bold cyan]Session closed.[/]")
            return False

        if cmd == "/run":
            if not arg:
                self.console.print("[red]Usage: /run <task description>[/]")
            else:
                self._run_goal(arg)
            return True

        if cmd == "/resume":
            try:
                state = self.orchestrator.resume_task()
                self._render_completion_summary(state)
            except NoActiveTaskError as exc:
                self.console.print(f"[yellow]{exc}[/]")
            return True

        if cmd == "/blackboard":
            self.render_blackboard()
            return True

        if cmd == "/thermal":
            self._handle_thermal_command(arg)
            return True

        if cmd == "/provider":
            if not arg:
                self.console.print(f"Current provider: [bold]{self.orchestrator.config.get('provider')}[/]")
            else:
                try:
                    new_p = self.orchestrator.set_provider(arg)
                    self.console.print(
                        f"[green]Provider switched to[/] [bold]{new_p}[/] "
                        f"(active model: {active_model(self.orchestrator.config)})"
                    )
                except ValueError as exc:
                    self.console.print(f"[red]{exc}[/]")
            return True

        if cmd == "/model":
            if not arg:
                self.console.print(f"Current model: [bold]{active_model(self.orchestrator.config)}[/]")
            else:
                try:
                    new_m = self.orchestrator.set_model(arg)
                    self.console.print(f"[green]Active model set to[/] [bold]{new_m}[/]")
                except ValueError as exc:
                    self.console.print(f"[red]{exc}[/]")
            return True

        if cmd == "/status":
            self.render_status()
            return True

        if cmd == "/clean":
            rep_ws = self.orchestrator.gc.collect(self.workspace)
            rep_root = self.orchestrator.gc.collect(HIVE_ROOT)
            total_removed = rep_ws.count + rep_root.count
            total_bytes = rep_ws.bytes_freed + rep_root.bytes_freed
            self.console.print(
                f"[green]Garbage Collector removed {total_removed} item(s) ({total_bytes} bytes freed).[/]"
            )
            return True

        if cmd == "/self-test":
            self._run_self_test()
            return True

        if cmd in ("/help", "/?"):
            self.render_help()
            return True

        if cmd == "/workspace":
            if not arg:
                self.console.print(f"Current workspace: [bold]{self.workspace}[/]")
            else:
                self.workspace = Path(arg).expanduser().resolve()
                self.workspace.mkdir(parents=True, exist_ok=True)
                self.orchestrator.blackboard.bind_workspace(self.workspace, reset_if_changed=True)
                self.orchestrator.blackboard.scan_workspace()
                self.console.print(f"[green]Workspace switched to[/] [bold]{self.workspace}[/]")
            return True

        self.console.print(f"[red]Unknown command: {cmd}. Type /help for available commands.[/]")
        return True

    def render_help(self) -> None:
        table = Table(title="HiveCoder Slash Commands", box=ROUNDED, header_style="bold cyan")
        table.add_column("Command", style="bold green")
        table.add_column("Description")
        table.add_row("/run <goal>", "Start a new autonomous software engineering task")
        table.add_row("/resume", "Resume an interrupted task from state.json & blackboard.json")
        table.add_row("/blackboard", "Inspect the live AST project map and agent Relay Notes")
        table.add_row("/thermal <adaptive|eco|balanced|turbo> [min]", "Switch thermal profile and optional cooldown minutes")
        table.add_row("/provider <ollama|api>", "Switch between local Ollama and cloud API provider")
        table.add_row("/model <model_name>", "Change active model (e.g. qwen3:8b, gpt-4o-mini)")
        table.add_row("/workspace [path]", "Show or switch the active target project workspace")
        table.add_row("/status", "Display task progress, Infirmary stats, thermal status & saved tokens")
        table.add_row("/clean", "Run the Garbage Collector immediately across workspace & repo")
        table.add_row("/self-test", "Execute HiveCoder's internal pytest verification suite")
        table.add_row("/help", "Show this command reference")
        table.add_row("/exit", "Exit HiveCoder")
        self.console.print(table)

    def _run_goal(self, goal: str) -> TaskState:
        state = self.orchestrator.run_task(goal=goal, workspace=self.workspace)
        self._render_completion_summary(state)
        return state

    def _handle_thermal_command(self, arg: str) -> None:
        if not arg:
            st = self.orchestrator.thermal.status()
            prof = st["profile"]
            self.console.print(
                f"Active thermal profile: [bold cyan]{prof['name']}[/] "
                f"(micro-pause={prof['micro_pause_seconds']}s, "
                f"cooldown={prof['cooldown_seconds']}s every {prof['cooldown_every_tasks']} tasks)"
            )
            return
        tokens = arg.split()
        profile_name = tokens[0]
        minutes: Optional[float] = None
        if len(tokens) > 1:
            try:
                minutes = float(tokens[1])
            except ValueError:
                self.console.print("[red]Invalid minutes argument. Example: /thermal eco 10[/]")
                return
        try:
            status = self.orchestrator.set_thermal(profile_name, minutes=minutes)
            prof = status["profile"]
            self.console.print(
                f"[green]Thermal profile updated to[/] [bold cyan]{prof['name']}[/] "
                f"(micro-pause {prof['micro_pause_seconds']}s | cooldown {prof['cooldown_seconds']}s)"
            )
        except ValueError as exc:
            self.console.print(f"[red]{exc}[/]")

    def _run_self_test(self) -> int:
        import os

        self.console.print("[bold cyan]Running HiveCoder self-test suite (pytest tests/)...[/]")
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        proc = subprocess.run(
            [sys.executable, "-B", "-m", "pytest", "-v", str(HIVE_ROOT / "tests")],
            cwd=HIVE_ROOT,
            env=env,
        )
        self.orchestrator.gc.collect(HIVE_ROOT)
        return proc.returncode

    def _render_completion_summary(self, state: TaskState) -> None:
        counts = state.counts()
        border = "bright_green" if state.phase.value == "completed" else "red"
        body = (
            f"Task ID: {state.task_id}\n"
            f"Phase: {state.phase.value.upper()}\n"
            f"Micro-Tasks Completed: {counts['done']}/{counts['total']}\n"
            f"Infirmary Runs / Healed: {state.metrics.medic_runs} / {state.metrics.medic_successes}\n"
            f"Tokens Saved via Blackboard: ~{state.metrics.tokens_saved}\n"
            f"GC Items Cleaned: {state.metrics.gc_items_removed}"
        )
        if state.error:
            body += f"\nError: {state.error}"
        self.console.print(Panel(body, title="Task Summary", border_style=border, box=ROUNDED))

    def run_repl(self) -> None:
        self.render_banner()
        while True:
            try:
                line = self.console.input("[bold bright_cyan]hive>[/] ")
            except (EOFError, KeyboardInterrupt):
                self.console.print("\n[bold cyan]HiveCoder session closed.[/]")
                break
            if not self.dispatch_command(line):
                break
