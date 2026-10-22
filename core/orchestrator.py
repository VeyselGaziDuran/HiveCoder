"""HiveOrchestrator: coordinates Architect, Coder, Inspector, Tester, Validator, Medic, Probes, MemoryLifecycle, ThermalGuard, GitGuard, and GC."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Callable, Optional

from agents.architect_agent import ArchitectAgent
from agents.base_agent import AgentResult
from agents.coder_agent import CoderAgent
from agents.inspector_agent import InspectorAgent
from agents.medic_agent import MedicAgent
from agents.tester_agent import TesterAgent
from core.blackboard import Blackboard
from core.config import HIVE_ROOT, resolve_memory_dir, save_config
from core.context_builder import ContextBuilder
from core.garbage_collector import GarbageCollector
from core.git_guard import GitGuard
from core.llm_client import LLMClient
from core.memory_lifecycle import MemoryLifecycle
from core.state_machine import (
    MicroTask,
    NoActiveTaskError,
    Phase,
    StateMachine,
    TaskHistory,
    TaskState,
)
from core.thermal_guard import ThermalGuard
from core.thermal_sensor import ThermalSensor
from core.validator import Validator
from probes.package_radar import PackageRadarProbe
from probes.runtime_probe import RuntimeProbe

EventCallback = Callable[[str, str, dict[str, Any]], None]


class HiveOrchestrator:
    """Coordinates planning, coding, inspection, validation, repair, memory lifecycle, and thermal pacing."""

    def __init__(
        self,
        config: dict[str, Any],
        llm_client: Optional[LLMClient] = None,
        on_event: Optional[EventCallback] = None,
        console: Any = None,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
        thermal_sensor: Optional[ThermalSensor] = None,
    ) -> None:
        self.config = config
        self.on_event = on_event
        self.console = console
        self.memory_dir = resolve_memory_dir(config)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        self.llm = llm_client or LLMClient(config)
        self.state_machine = StateMachine(
            state_file=self.memory_dir / "state.json",
            log_file=self.memory_dir / "hive_log.jsonl",
        )
        self.history = TaskHistory(self.memory_dir / "task_history.json")
        self.blackboard = Blackboard(self.memory_dir)
        self.gc = GarbageCollector()

        mem_cfg = self.config.get("memory_lifecycle", {})
        self.memory_lifecycle = MemoryLifecycle(
            flush_notes_on_complete=bool(mem_cfg.get("flush_notes_on_complete", True)),
            remove_state_on_complete=bool(mem_cfg.get("remove_state_on_complete", True)),
            remove_snapshots_on_complete=bool(mem_cfg.get("remove_snapshots_on_complete", True)),
        )

        probes_cfg = self.config.get("probes", {})
        self.runtime_probe = RuntimeProbe()
        self.package_radar = PackageRadarProbe(
            online_enabled=bool(probes_cfg.get("online_package_radar", True)),
            timeout=float(probes_cfg.get("radar_timeout_seconds", 2.0)),
        )

        thermal_cfg = self.config.setdefault("thermal", {})
        sensor = thermal_sensor
        if sensor is None and bool(thermal_cfg.get("sensor_enabled", True)):
            sensor = ThermalSensor()

        self.thermal = ThermalGuard(
            settings=thermal_cfg,
            unloader=self.llm.unload,
            sleeper=sleeper,
            clock=clock,
            console=console,
            show_progress=console is not None,
            on_state_change=self._sync_thermal_state,
            on_event=lambda kind, msg: self._emit(kind, msg),
            sensor=sensor,
        )

    def _emit(self, kind: str, message: str, **data: Any) -> None:
        self.state_machine.log_event(kind, message=message, **data)
        if self.on_event is not None:
            self.on_event(kind, message, data)

    def _sync_thermal_state(self, thermal_state: dict[str, Any]) -> None:
        if self.state_machine.state is not None:
            self.state_machine.state.thermal = thermal_state
            self.state_machine.save()

    def _resolve_workspace(self, workspace: str | Path | None) -> Path:
        raw = workspace or self.config.get("default_workspace", "workspace")
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (HIVE_ROOT / path).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _populate_runtime_context(self, goal: str) -> None:
        """Run RuntimeProbe and PackageRadarProbe to anchor language/library versions on the Blackboard."""
        probes_cfg = self.config.get("probes", {})
        if not bool(probes_cfg.get("enabled", True)):
            return
        env_info = self.runtime_probe.inspect()
        sections = [env_info.to_markdown()]
        pkg_infos = self.package_radar.scan_goal_packages(goal)
        if pkg_infos:
            pkg_lines = ["## PACKAGE RADAR"] + [f"- {p.format_brief()}" for p in pkg_infos]
            sections.append("\n".join(pkg_lines))
        self.blackboard.set_runtime_context("\n\n".join(sections))

    def _verified_task_ids(self) -> list[str]:
        if self.state_machine.state is None:
            return []
        return [mt.id for mt in self.state_machine.state.micro_tasks if mt.status == "done"]

    def _build_runtime_components(
        self, workspace: Path
    ) -> tuple[ContextBuilder, Validator, GitGuard, ArchitectAgent, CoderAgent, TesterAgent, InspectorAgent, MedicAgent]:
        limits = self.config.get("limits", {})
        ctx_builder = ContextBuilder(
            blackboard=self.blackboard,
            workspace=workspace,
            max_chars=int(limits.get("max_context_chars", 14000)),
            max_notes=int(limits.get("max_relay_notes_in_context", 12)),
        )
        validator = Validator(
            workspace=workspace,
            pytest_timeout=float(limits.get("pytest_timeout_seconds", 180)),
        )
        git_guard = GitGuard(
            workspace=workspace,
            store_dir=self.memory_dir / "snapshots",
            enabled=bool(self.config.get("git_guard", {}).get("enabled", True)),
        )
        architect = ArchitectAgent(
            llm=self.llm,
            blackboard=self.blackboard,
            context_builder=ctx_builder,
            workspace=workspace,
            max_micro_tasks=int(limits.get("max_micro_tasks", 30)),
        )
        coder = CoderAgent(self.llm, self.blackboard, ctx_builder, workspace)
        tester = TesterAgent(self.llm, self.blackboard, ctx_builder, workspace)
        inspector = InspectorAgent(
            llm=self.llm,
            blackboard=self.blackboard,
            context_builder=ctx_builder,
            workspace=workspace,
            enable_llm_review=bool(self.config.get("agents", {}).get("llm_inspector", True)),
        )
        medic = MedicAgent(self.llm, self.blackboard, ctx_builder, workspace)
        return ctx_builder, validator, git_guard, architect, coder, tester, inspector, medic

    def _record_agent_metrics(self, result: AgentResult) -> None:
        st = self.state_machine.state
        if st is None:
            return
        st.metrics.llm_calls += 1
        st.metrics.prompt_chars += result.prompt_chars
        st.metrics.response_chars += result.response_chars
        st.metrics.tokens_saved += result.tokens_saved
        st.metrics.work_seconds = round(st.metrics.work_seconds + result.elapsed_seconds, 3)
        self.state_machine.save()

    def run_task(self, goal: str, workspace: str | Path | None = None) -> TaskState:
        """Start and execute a brand-new autonomous software task."""
        ws = self._resolve_workspace(workspace)
        self.blackboard.bind_workspace(ws, reset_if_changed=True)
        self.blackboard.reset()
        self.blackboard.scan_workspace(agent="initial_scan")
        self._populate_runtime_context(goal)

        state = self.state_machine.new_task(goal=goal, workspace=ws)
        self._sync_thermal_state(self.thermal.export_state())

        (
            _ctx,
            validator,
            git_guard,
            architect,
            coder,
            tester,
            inspector,
            medic,
        ) = self._build_runtime_components(ws)

        state.pre_task_snapshot = git_guard.snapshot(f"pre-task-{state.task_id}")
        self.state_machine.save()

        self._emit("planning", f"ArchitectAgent decomposing task: {goal}")
        self.state_machine.transition(Phase.PLANNING, "starting architecture decomposition")
        try:
            plan = architect.plan(goal)
            state.metrics.llm_calls += 1
            self.state_machine.set_plan(plan)
        except Exception as exc:  # noqa: BLE001
            self._emit("error", f"Planning failed: {exc}")
            self.state_machine.fail(f"Planning failed: {exc}")
            self.history.append(state)
            return state

        self._emit(
            "plan_ready",
            f"ArchitectAgent created {len(plan)} micro-task(s).",
            micro_tasks=[mt.model_dump() for mt in plan],
        )
        self.state_machine.transition(Phase.CODING, "plan ready")
        return self._execute_loop(ws, validator, git_guard, coder, tester, inspector, medic)

    def resume_task(self) -> TaskState:
        """Resume an interrupted, paused, or failed task from ``state.json``."""
        state = self.state_machine.load()
        if state is None:
            raise NoActiveTaskError("No saved state.json found to resume.")
        ws = Path(state.workspace).resolve()
        ws.mkdir(parents=True, exist_ok=True)
        self.blackboard.bind_workspace(ws, reset_if_changed=False)
        self.blackboard.scan_workspace(agent="resume_scan")
        if not self.blackboard.data.get("runtime_context"):
            self._populate_runtime_context(state.goal)

        (
            _ctx,
            validator,
            git_guard,
            architect,
            coder,
            tester,
            inspector,
            medic,
        ) = self._build_runtime_components(ws)

        self.thermal.import_state(state.thermal)
        if self.thermal.remaining_cooldown() > 0:
            if state.phase != Phase.COOLING:
                self.state_machine.pause("resuming pending thermal cooldown")
                self.state_machine.transition(Phase.COOLING, "completing interrupted cooldown")
            self.thermal.resume_pending_cooldown()

        interrupted = self.state_machine.prepare_resume()
        for mt in interrupted:
            if mt.snapshot:
                git_guard.rollback(mt.snapshot)
                self._emit("rollback", f"Rolled back interrupted micro-task {mt.id} to snapshot {mt.snapshot[:8]}")
        if interrupted:
            self.blackboard.scan_workspace(agent="post_rollback_scan")

        # Purge any relay notes left by interrupted/unverified micro-tasks
        purged = self.memory_lifecycle.purge_unverified_notes(
            self.blackboard,
            verified_task_ids=self._verified_task_ids(),
        )
        if purged:
            self._emit("memory_gc", f"Purged {purged} unverified relay note(s) before resume.")

        if not state.micro_tasks:
            self.state_machine.transition(Phase.PLANNING, "resuming at planning phase")
            plan = architect.plan(state.goal)
            state.metrics.llm_calls += 1
            self.state_machine.set_plan(plan)
            self.state_machine.transition(Phase.CODING, "plan ready after resume")
        else:
            self.state_machine.transition(Phase.CODING, "resuming micro-task queue")

        return self._execute_loop(ws, validator, git_guard, coder, tester, inspector, medic)

    def _execute_loop(
        self,
        workspace: Path,
        validator: Validator,
        git_guard: GitGuard,
        coder: CoderAgent,
        tester: TesterAgent,
        inspector: InspectorAgent,
        medic: MedicAgent,
    ) -> TaskState:
        state = self.state_machine.state
        assert state is not None
        limits = self.config.get("limits", {})
        max_coder_attempts = max(1, int(limits.get("max_coder_attempts", 2)))
        max_medic_attempts = max(1, int(limits.get("max_medic_attempts", 3)))

        try:
            while True:
                mt = self.state_machine.next_pending()
                if mt is None:
                    break

                if state.phase != Phase.CODING:
                    self.state_machine.transition(Phase.CODING, f"starting {mt.id}")

                snap = git_guard.snapshot(f"before-{mt.id}")
                self.state_machine.start_micro_task(mt.id, snapshot=snap)
                step_started = time.time()

                worker = tester if mt.kind == "test" else coder
                self._emit(
                    "micro_start",
                    f"[{mt.id}] {worker.role} -> {mt.target_file}: {mt.title}",
                    micro_id=mt.id,
                    agent=worker.role,
                    target_file=mt.target_file,
                )

                agent_result: Optional[AgentResult] = None
                feedback = ""
                for attempt in range(1, max_coder_attempts + 1):
                    agent_result = worker.run(mt, feedback=feedback)
                    self._record_agent_metrics(agent_result)
                    if agent_result.success:
                        break
                    feedback = agent_result.error or "No valid <<<CODE>>> block found."
                    self._emit("retry", f"[{mt.id}] {worker.role} retry {attempt}/{max_coder_attempts}: {feedback}")

                assert agent_result is not None
                if not agent_result.success:
                    healed = self._enter_infirmary(
                        mt=mt,
                        failure_report=agent_result.error,
                        workspace=workspace,
                        validator=validator,
                        inspector=inspector,
                        medic=medic,
                        max_medic_attempts=max_medic_attempts,
                    )
                    if not healed:
                        return self._abort_on_micro_failure(mt, agent_result.error, workspace, git_guard)
                else:
                    self.state_machine.record_files(mt.id, agent_result.files_written, agent_result.relay_note)
                    self._emit(
                        "relay_note",
                        f"[{mt.id}] {worker.role} note: {agent_result.relay_note}",
                        micro_id=mt.id,
                        files=agent_result.files_written,
                        note=agent_result.relay_note,
                    )

                    self.state_machine.transition(Phase.INSPECTING, f"inspecting {mt.id}")
                    verdict = inspector.inspect(mt, agent_result.files_written)
                    if inspector.enable_llm_review:
                        state.metrics.llm_calls += 1
                    self.state_machine.mark_inspected(mt.id, verdict.approved, verdict.format_feedback())

                    if not verdict.approved:
                        state.metrics.inspector_rejections += 1
                        self._emit(
                            "inspection_rejected",
                            f"[{mt.id}] InspectorAgent rejected {mt.target_file}: {verdict.format_feedback()}",
                            micro_id=mt.id,
                        )
                        healed = self._enter_infirmary(
                            mt=mt,
                            failure_report=verdict.format_feedback(),
                            workspace=workspace,
                            validator=validator,
                            inspector=inspector,
                            medic=medic,
                            max_medic_attempts=max_medic_attempts,
                        )
                        if not healed:
                            return self._abort_on_micro_failure(
                                mt, mt.last_error or verdict.format_feedback(), workspace, git_guard
                            )
                    else:
                        self._emit("inspection_approved", f"[{mt.id}] InspectorAgent approved {mt.target_file}")
                        self.state_machine.transition(Phase.VALIDATING, f"validating {mt.id}")
                        val_res = validator.validate(mt.files_written, run_tests=True)
                        gc_rep = self.gc.collect(workspace)
                        state.metrics.gc_items_removed += gc_rep.count

                        if val_res.passed:
                            self.state_machine.mark_validated(mt.id, True, val_res.summary())
                            self.state_machine.complete_micro_task(mt.id)
                            self.blackboard.lock_verified_symbols(mt.files_written)
                            self._emit(
                                "micro_done",
                                f"[{mt.id}] Verified & completed ({val_res.stage}).",
                                micro_id=mt.id,
                            )
                        else:
                            state.metrics.validation_failures += 1
                            self.state_machine.mark_validated(mt.id, False, val_res.summary())
                            self._emit(
                                "validation_failed",
                                f"[{mt.id}] Validation failed ({val_res.stage}):\n{val_res.output}",
                                micro_id=mt.id,
                            )
                            healed = self._enter_infirmary(
                                mt=mt,
                                failure_report=val_res.summary(),
                                workspace=workspace,
                                validator=validator,
                                inspector=inspector,
                                medic=medic,
                                max_medic_attempts=max_medic_attempts,
                            )
                            if not healed:
                                return self._abort_on_micro_failure(
                                    mt, mt.last_error or val_res.summary(), workspace, git_guard
                                )

                step_elapsed = max(0.01, time.time() - step_started)
                self.thermal.record_task(step_elapsed, response_chars=agent_result.response_chars)
                if self.state_machine.next_pending() is not None:
                    if self.thermal.needs_cooldown():
                        self.state_machine.transition(Phase.COOLING, "deep thermal cooldown triggered")
                        self.thermal.deep_cooldown()
                        state.metrics.cooldowns += 1
                        self.state_machine.transition(Phase.CODING, "cooldown finished")
                    else:
                        self.thermal.micro_pause()
                        self.state_machine.transition(Phase.CODING, "next micro-task")

            final_gc = self.gc.collect(workspace)
            state.metrics.gc_items_removed += final_gc.count
            self.state_machine.finish()
            self.history.append(state)

            cleanup = self.memory_lifecycle.finalize_completed_task(
                blackboard=self.blackboard,
                state_machine=self.state_machine,
                git_guard=git_guard,
                gc=self.gc,
                workspace=workspace,
            )
            state.metrics.gc_items_removed += cleanup.gc_items_removed
            self._emit(
                "task_completed",
                f"Task {state.task_id} completed; {cleanup.purged_notes} ephemeral note(s) cleaned.",
            )
            return state
        except KeyboardInterrupt:
            self._emit("paused", "Interrupted by user (Ctrl+C). Cleaning unverified notes and saving state.")
            self.state_machine.pause("KeyboardInterrupt")
            self.memory_lifecycle.purge_unverified_notes(
                self.blackboard,
                verified_task_ids=self._verified_task_ids(),
            )
            return state

    def _enter_infirmary(
        self,
        mt: MicroTask,
        failure_report: str,
        workspace: Path,
        validator: Validator,
        inspector: InspectorAgent,
        medic: MedicAgent,
        max_medic_attempts: int,
    ) -> bool:
        """Run the Infirmary (MedicAgent -> InspectorAgent -> Validator) loop with stagnation loop-breaker."""
        state = self.state_machine.state
        assert state is not None
        current_report = failure_report
        seen_hashes: set[str] = set()
        base_temp = float(self.config.get("generation", {}).get("temperature", 0.2))

        for attempt in range(1, max_medic_attempts + 1):
            if state.phase != Phase.INFIRMARY:
                self.state_machine.transition(Phase.INFIRMARY, f"infirmary attempt {attempt} for {mt.id}")

            mt.medic_attempts += 1
            state.metrics.medic_runs += 1
            patient_file = medic.select_patient_file(mt, current_report)
            patient_abs = workspace / patient_file
            if patient_abs.is_file():
                seen_hashes.add(hashlib.sha1(patient_abs.read_bytes()).hexdigest())

            attempt_temp: Optional[float] = None
            loop_hint = ""
            if attempt > 1:
                attempt_temp = min(0.85, round(base_temp + 0.25 * (attempt - 1), 2))
                loop_hint = (
                    f"Previous repair attempt {attempt - 1} did not resolve the defect. "
                    "Do NOT repeat the previous implementation verbatim; change the faulty logic or import."
                )

            self._emit(
                "infirmary",
                f"[{mt.id}] MedicAgent attempt {attempt}/{max_medic_attempts} on {patient_file}",
                micro_id=mt.id,
                attempt=attempt,
                patient=patient_file,
            )

            med_result = medic.repair(
                mt,
                current_report,
                broken_file=patient_file,
                attempt=attempt,
                temperature=attempt_temp,
                loop_breaker_hint=loop_hint,
            )
            self._record_agent_metrics(med_result)
            if not med_result.success:
                current_report = med_result.error or current_report
                continue

            if patient_abs.is_file():
                new_hash = hashlib.sha1(patient_abs.read_bytes()).hexdigest()
                if new_hash in seen_hashes and attempt < max_medic_attempts:
                    self._emit(
                        "loop_breaker",
                        f"[{mt.id}] Identical file hash detected on attempt {attempt}; escalating temperature.",
                        micro_id=mt.id,
                    )
                seen_hashes.add(new_hash)

            self.state_machine.record_files(mt.id, med_result.files_written, med_result.relay_note)
            self._emit(
                "relay_note",
                f"[{mt.id}] MedicAgent note: {med_result.relay_note}",
                micro_id=mt.id,
                files=med_result.files_written,
                note=med_result.relay_note,
            )

            self.state_machine.transition(Phase.INSPECTING, f"re-inspecting {mt.id} after medic")
            verdict = inspector.inspect(mt, med_result.files_written)
            if inspector.enable_llm_review:
                state.metrics.llm_calls += 1
            self.state_machine.mark_inspected(mt.id, verdict.approved, verdict.format_feedback())

            if not verdict.approved:
                state.metrics.inspector_rejections += 1
                current_report = verdict.format_feedback()
                self._emit("inspection_rejected", f"[{mt.id}] Post-medic inspection failed: {current_report}")
                continue

            self.state_machine.transition(Phase.VALIDATING, f"re-validating {mt.id} after medic")
            val_res = validator.validate(mt.files_written, run_tests=True)
            gc_rep = self.gc.collect(workspace)
            state.metrics.gc_items_removed += gc_rep.count

            if val_res.passed:
                state.metrics.medic_successes += 1
                self.state_machine.mark_validated(mt.id, True, val_res.summary())
                self.state_machine.complete_micro_task(mt.id)
                self.blackboard.lock_verified_symbols(mt.files_written)
                self._emit(
                    "infirmary_healed",
                    f"[{mt.id}] MedicAgent repaired {patient_file} on attempt {attempt}.",
                    micro_id=mt.id,
                )
                return True

            state.metrics.validation_failures += 1
            self.state_machine.mark_validated(mt.id, False, val_res.summary())
            current_report = val_res.summary()
            self._emit("validation_failed", f"[{mt.id}] Post-medic validation failed:\n{val_res.output}")

        return False

    def _abort_on_micro_failure(
        self,
        mt: MicroTask,
        error: str,
        workspace: Path,
        git_guard: GitGuard,
    ) -> TaskState:
        state = self.state_machine.state
        assert state is not None
        if mt.snapshot:
            git_guard.rollback(mt.snapshot)
            self.blackboard.scan_workspace(agent="rollback_scan")
            self._emit("rollback", f"[{mt.id}] Rolled workspace back to snapshot {mt.snapshot[:8]}")
        self.memory_lifecycle.purge_unverified_notes(
            self.blackboard,
            verified_task_ids=self._verified_task_ids(),
        )
        gc_rep = self.gc.collect(workspace)
        state.metrics.gc_items_removed += gc_rep.count
        self.state_machine.fail_micro_task(mt.id, error)
        self.state_machine.fail(f"Micro-task {mt.id} failed after Infirmary attempts: {error[:300]}")
        self.history.append(state)
        self._emit("task_failed", f"Task {state.task_id} failed on micro-task {mt.id}.")
        return state

    def set_thermal(self, profile: str, minutes: Optional[float] = None) -> dict[str, Any]:
        self.thermal.set_profile(profile, cooldown_minutes=minutes)
        save_config(self.config)
        return self.thermal.status()

    def set_provider(self, provider: str) -> str:
        provider = provider.strip().lower()
        if provider not in ("ollama", "api"):
            raise ValueError("Provider must be 'ollama' or 'api'")
        self.config["provider"] = provider
        save_config(self.config)
        return provider

    def set_model(self, model_name: str) -> str:
        model_name = model_name.strip()
        if not model_name:
            raise ValueError("Model name cannot be empty")
        section = "ollama" if self.config.get("provider", "ollama") == "ollama" else "api"
        self.config.setdefault(section, {})["model"] = model_name
        save_config(self.config)
        return model_name
