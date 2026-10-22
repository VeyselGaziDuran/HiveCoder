"""HiveCoder core: deterministic engine pieces shared by every micro-agent."""

from core.ast_patcher import ASTPatcher
from core.blackboard import Blackboard, extract_file_summary, extract_python_summary
from core.config import DEFAULT_CONFIG, HIVE_ROOT, active_model, load_config, save_config
from core.context_builder import ContextBuilder
from core.garbage_collector import GarbageCollector, GCReport
from core.git_guard import GitGuard
from core.llm_client import LLMClient, NanoResponse, extract_json, parse_nano_tags, strip_think
from core.memory_lifecycle import CleanupSummary, MemoryLifecycle
from core.orchestrator import HiveOrchestrator
from core.state_machine import MicroTask, Phase, StateMachine, TaskHistory, TaskState
from core.thermal_guard import ThermalGuard, ThermalProfile
from core.thermal_sensor import ThermalReading, ThermalSensor
from core.validator import ValidationResult, Validator

__version__ = "2.0.0"

__all__ = [
    "__version__",
    "ASTPatcher",
    "Blackboard",
    "CleanupSummary",
    "ContextBuilder",
    "DEFAULT_CONFIG",
    "GCReport",
    "GarbageCollector",
    "GitGuard",
    "HIVE_ROOT",
    "HiveOrchestrator",
    "LLMClient",
    "MemoryLifecycle",
    "MicroTask",
    "NanoResponse",
    "Phase",
    "StateMachine",
    "TaskHistory",
    "TaskState",
    "ThermalGuard",
    "ThermalProfile",
    "ThermalReading",
    "ThermalSensor",
    "ValidationResult",
    "Validator",
    "active_model",
    "extract_file_summary",
    "extract_json",
    "extract_python_summary",
    "load_config",
    "parse_nano_tags",
    "save_config",
    "strip_think",
]
