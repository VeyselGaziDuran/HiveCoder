"""HiveCoder core: deterministic engine pieces shared by every micro-agent."""

from core.blackboard import Blackboard, extract_file_summary, extract_python_summary
from core.config import DEFAULT_CONFIG, HIVE_ROOT, active_model, load_config, save_config
from core.context_builder import ContextBuilder
from core.garbage_collector import GarbageCollector, GCReport
from core.git_guard import GitGuard
from core.llm_client import LLMClient, NanoResponse, extract_json, parse_nano_tags, strip_think
from core.orchestrator import HiveOrchestrator
from core.state_machine import MicroTask, Phase, StateMachine, TaskHistory, TaskState
from core.thermal_guard import ThermalGuard, ThermalProfile
from core.validator import ValidationResult, Validator

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "Blackboard",
    "ContextBuilder",
    "DEFAULT_CONFIG",
    "GCReport",
    "GarbageCollector",
    "GitGuard",
    "HIVE_ROOT",
    "HiveOrchestrator",
    "LLMClient",
    "MicroTask",
    "NanoResponse",
    "Phase",
    "StateMachine",
    "TaskHistory",
    "TaskState",
    "ThermalGuard",
    "ThermalProfile",
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
