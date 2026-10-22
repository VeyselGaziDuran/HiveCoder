#!/usr/bin/env python3
"""HiveCoder CLI & Interactive TUI entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from rich.console import Console

from core.config import load_config
from core.orchestrator import HiveOrchestrator
from core.state_machine import NoActiveTaskError, Phase
from ui.terminal_ui import TerminalUI


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hive.py",
        description="HiveCoder — Stateless micro-agent orchestrator with AST-indexed shared memory",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Run a single autonomous task non-interactively and exit.",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Target workspace directory for generated project files.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted task from state.json and blackboard.json.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config.json file.",
    )
    parser.add_argument(
        "--thermal",
        choices=["adaptive", "eco", "balanced", "turbo"],
        default=None,
        help="Override thermal profile for this session.",
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "api"],
        default=None,
        help="Override LLM provider (ollama or api).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override active model name.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run HiveCoder's internal pytest suite and exit.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.workspace:
        config["default_workspace"] = str(Path(args.workspace).resolve())

    console = Console()
    orchestrator = HiveOrchestrator(config=config, console=console)
    ui = TerminalUI(
        orchestrator=orchestrator,
        workspace=args.workspace or config.get("default_workspace", "workspace"),
        console=console,
    )

    if args.thermal:
        orchestrator.set_thermal(args.thermal, persist=False)
    if args.provider:
        orchestrator.set_provider(args.provider, persist=False)
    if args.model:
        orchestrator.set_model(args.model, persist=False)

    if args.self_test:
        return ui._run_self_test()

    if args.resume:
        try:
            state = orchestrator.resume_task()
        except NoActiveTaskError as exc:
            console.print(f"[red]{exc}[/]")
            return 1
        ui._render_completion_summary(state)
        return 0 if state.phase == Phase.COMPLETED else 1

    if args.task:
        state = orchestrator.run_task(goal=args.task, workspace=ui.workspace)
        ui._render_completion_summary(state)
        return 0 if state.phase == Phase.COMPLETED else 1

    ui.run_repl()
    return 0


if __name__ == "__main__":
    sys.exit(main())
