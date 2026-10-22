"""Configuration loading with deep-merged defaults.

``config.json`` only needs to contain the values a user wants to override; every
missing key is filled from :data:`DEFAULT_CONFIG` so older config files keep
working after upgrades.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

HIVE_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH: Path = HIVE_ROOT / "config.json"

DEFAULT_CONFIG: dict[str, Any] = {
    "provider": "ollama",
    "ollama": {
        "base_url": "http://localhost:11434",
        "model": "qwen3:8b",
        "num_ctx": 8192,
        "think": False,
    },
    "api": {
        "style": "openai",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "HIVE_API_KEY",
        "model": "gpt-4o-mini",
        "max_tokens": 4096,
    },
    "generation": {
        "temperature": 0.2,
        "timeout_seconds": 600,
        "max_retries": 3,
    },
    "thermal": {
        "profile": "adaptive",
        "sensor_enabled": True,
        "early_wakeup": True,
        "unload_model_on_cooldown": True,
        "profiles": {
            "adaptive": {
                "micro_pause_seconds": 1,
                "cooldown_every_tasks": 0,
                "cooldown_seconds": 300,
                "max_continuous_work_seconds": 0,
            },
            "eco": {
                "micro_pause_seconds": 10,
                "cooldown_every_tasks": 3,
                "cooldown_seconds": 300,
                "max_continuous_work_seconds": 480,
            },
            "balanced": {
                "micro_pause_seconds": 5,
                "cooldown_every_tasks": 5,
                "cooldown_seconds": 180,
                "max_continuous_work_seconds": 900,
            },
            "turbo": {
                "micro_pause_seconds": 1,
                "cooldown_every_tasks": 0,
                "cooldown_seconds": 0,
                "max_continuous_work_seconds": 0,
            },
        },
    },
    "limits": {
        "max_micro_tasks": 30,
        "max_medic_attempts": 3,
        "max_coder_attempts": 2,
        "max_context_chars": 14000,
        "max_relay_notes_in_context": 12,
        "pytest_timeout_seconds": 180,
    },
    "memory_lifecycle": {
        "flush_notes_on_complete": True,
        "remove_state_on_complete": True,
        "remove_snapshots_on_complete": True,
    },
    "probes": {
        "enabled": True,
        "online_package_radar": True,
        "radar_timeout_seconds": 2.0,
    },
    "agents": {"llm_inspector": True},
    "git_guard": {"enabled": True},
    "memory_dir": "memory",
    "default_workspace": "workspace",
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict where ``override`` is recursively merged onto ``base``."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load ``config.json`` (or ``path``) merged on top of the defaults."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    user_cfg: dict[str, Any] = {}
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            raise ValueError(f"{cfg_path} must contain a JSON object")
        user_cfg = loaded
    config = deep_merge(DEFAULT_CONFIG, user_cfg)
    config["_path"] = str(cfg_path)
    return config


def save_config(config: dict[str, Any], path: str | Path | None = None) -> Path:
    """Atomically persist ``config`` (private ``_`` keys are stripped)."""
    cfg_path = Path(path or config.get("_path") or DEFAULT_CONFIG_PATH)
    clean = {k: v for k, v in config.items() if not k.startswith("_")}
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cfg_path.with_suffix(cfg_path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, cfg_path)
    return cfg_path


def resolve_memory_dir(config: dict[str, Any], hive_root: Path = HIVE_ROOT) -> Path:
    """Absolute path of the memory directory (relative paths are anchored at the repo)."""
    mem = Path(config.get("memory_dir", "memory")).expanduser()
    return mem if mem.is_absolute() else (hive_root / mem)


def active_model(config: dict[str, Any]) -> str:
    """Model name used by the currently selected provider."""
    provider = config.get("provider", "ollama")
    section = config.get("ollama" if provider == "ollama" else "api", {})
    return str(section.get("model", ""))
