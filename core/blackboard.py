"""Blackboard: deterministic AST project map + agent-to-agent relay notes.

Agents never receive chat history. Instead, every time a file is written the
blackboard re-extracts its *public surface* (classes, method and function
signatures, imports, module constants, JSON schema ...) with Python's ``ast``
module and stores it next to the 1-2 sentence relay note the agent left for
its successor.  The next agent only reads :meth:`Blackboard.get_compact_memory`
- a few hundred tokens instead of the whole repository - and still knows every
exact name in the project.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import re
import tokenize
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Optional

IGNORED_DIRS: frozenset[str] = frozenset(
    {
        ".git", "venv", ".venv", "env", "__pycache__", ".pytest_cache", ".hive_tmp",
        "node_modules", ".mypy_cache", ".ruff_cache", ".idea", ".vscode", "dist",
        "build", ".tox", ".eggs",
    }
)
TEXT_SUFFIXES: frozenset[str] = frozenset(
    {
        ".py", ".json", ".md", ".txt", ".toml", ".yaml", ".yml", ".cfg", ".ini",
        ".html", ".css", ".scss", ".js", ".jsx", ".ts", ".tsx", ".dart", ".kt",
        ".swift", ".vue", ".gradle", ".plist", ".sql", ".sh", ".csv",
        ".env.example", ".xml",
    }
)
CODE_SUFFIXES: frozenset[str] = frozenset(
    {".js", ".jsx", ".ts", ".tsx", ".dart", ".kt", ".swift"}
)
MAX_SCAN_BYTES = 512_000
STUB_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"#\s*T[O]DO\b", re.IGNORECASE),
    re.compile(r"#\s*F[I]XME\b", re.IGNORECASE),
    re.compile(r"raise\s+NotImplementedError"),
    re.compile(r"^\s*\.\.\.\s*$", re.MULTILINE),
)


class UnsafePathError(ValueError):
    """Raised when an agent tries to touch a path outside the workspace."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_relpath(workspace: str | Path, path: str) -> str:
    """Turn an agent-supplied path into a safe POSIX path relative to ``workspace``."""
    cleaned = path.strip().strip("`'\"").strip()
    cleaned = cleaned.replace("\\", "/")
    if not cleaned:
        raise UnsafePathError("Empty file path")
    ws = Path(workspace).resolve()
    candidate = Path(cleaned)
    if candidate.is_absolute():
        try:
            cleaned = candidate.resolve().relative_to(ws).as_posix()
        except ValueError as exc:
            raise UnsafePathError(f"Absolute path outside workspace: {path}") from exc
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    parts = PurePosixPath(cleaned).parts
    if not parts or any(p == ".." for p in parts):
        raise UnsafePathError(f"Path escapes workspace: {path}")
    if parts[0] in IGNORED_DIRS:
        raise UnsafePathError(f"Refusing to write inside protected directory: {path}")
    resolved = (ws / cleaned).resolve()
    if ws != resolved and ws not in resolved.parents:
        raise UnsafePathError(f"Path escapes workspace: {path}")
    return PurePosixPath(cleaned).as_posix()


# ---------------------------------------------------------------------- extraction
def _unparse(node: Optional[ast.AST]) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:  # noqa: BLE001 - ast.unparse never should fail, but stay safe
        return "?"


def format_function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    decorators = []
    for dec in node.decorator_list:
        name = _unparse(dec)
        if name in {"property", "staticmethod", "classmethod"} or name.endswith(".setter"):
            decorators.append(f"@{name} ")
    returns = f" -> {_unparse(node.returns)}" if node.returns is not None else ""
    return f"{''.join(decorators)}{prefix} {node.name}({_unparse(node.args)}){returns}"


def _assigned_names(target: ast.AST) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Starred):
        return _assigned_names(target.value)
    if isinstance(target, (ast.Tuple, ast.List)):
        names: list[str] = []
        for elt in target.elts:
            names.extend(_assigned_names(elt))
        return names
    return []


def _short_value(node: ast.AST, limit: int = 48) -> str:
    text = _unparse(node)
    return text if len(text) <= limit else ""


def _is_main_if(node: ast.If) -> bool:
    try:
        test_src = ast.unparse(node.test).replace("'", '"')
        return "__name__" in test_src and "__main__" in test_src
    except Exception:  # noqa: BLE001
        return False


def _iter_module_statements(body: list[ast.stmt]) -> list[ast.stmt]:
    """Yield module-level statements, including those inside top-level ``if`` and ``try`` blocks (excluding ``__main__``)."""
    out: list[ast.stmt] = []
    for node in body:
        if isinstance(node, ast.If):
            if _is_main_if(node):
                continue
            out.extend(_iter_module_statements(node.body))
            out.extend(_iter_module_statements(node.orelse))
        elif isinstance(node, ast.Try):
            out.extend(_iter_module_statements(node.body))
            for handler in node.handlers:
                out.extend(_iter_module_statements(handler.body))
            out.extend(_iter_module_statements(node.orelse))
            out.extend(_iter_module_statements(node.finalbody))
        elif isinstance(node, ast.With):
            out.extend(_iter_module_statements(node.body))
        else:
            out.append(node)
    return out


def extract_python_summary(source: str) -> dict[str, Any]:
    """Extract the public surface of a Python module deterministically."""
    summary: dict[str, Any] = {
        "docstring": "",
        "imports": [],
        "classes": [],
        "functions": [],
        "globals": [],
        "syntax_error": None,
    }
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        summary["syntax_error"] = f"line {exc.lineno}: {exc.msg}"
        summary["functions"] = [
            f"def {name}({args.strip()})"
            for name, args in re.findall(r"^\s*(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)", source, re.MULTILINE)
        ]
        summary["classes"] = [
            {"name": name, "bases": [], "methods": [], "attributes": []}
            for name in re.findall(r"^class\s+(\w+)", source, re.MULTILINE)
        ]
        return summary

    doc = ast.get_docstring(tree)
    if doc:
        summary["docstring"] = doc.strip().splitlines()[0][:160]

    for node in _iter_module_statements(tree.body):
        if isinstance(node, ast.Import):
            for alias in node.names:
                summary["imports"].append(alias.name + (f" as {alias.asname}" if alias.asname else ""))
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            names = ", ".join(a.name + (f" as {a.asname}" if a.asname else "") for a in node.names)
            summary["imports"].append(f"from {module} import {names}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            summary["functions"].append(format_function_signature(node))
        elif isinstance(node, ast.ClassDef):
            methods: list[str] = []
            attributes: list[str] = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append(format_function_signature(item))
                elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    attributes.append(f"{item.target.id}: {_unparse(item.annotation)}")
                elif isinstance(item, ast.Assign):
                    for tgt in item.targets:
                        attributes.extend(_assigned_names(tgt))
            summary["classes"].append(
                {
                    "name": node.name,
                    "bases": [_unparse(b) for b in node.bases],
                    "methods": methods,
                    "attributes": attributes,
                }
            )
        elif isinstance(node, ast.Assign):
            value = _short_value(node.value)
            for tgt in node.targets:
                for name in _assigned_names(tgt):
                    summary["globals"].append(f"{name} = {value}" if value else name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            value = _short_value(node.value) if node.value is not None else ""
            entry = f"{node.target.id}: {_unparse(node.annotation)}"
            summary["globals"].append(f"{entry} = {value}" if value else entry)
        elif hasattr(ast, "TypeAlias") and isinstance(node, getattr(ast, "TypeAlias")):
            alias_name = getattr(getattr(node, "name", None), "id", "")
            if alias_name:
                summary["globals"].append(f"{alias_name} = {_short_value(node.value) or 'TypeAlias'}")

    summary["imports"] = list(dict.fromkeys(summary["imports"]))
    summary["functions"] = list(dict.fromkeys(summary["functions"]))
    summary["globals"] = list(dict.fromkeys(summary["globals"]))
    seen_cls: set[str] = set()
    unique_cls: list[dict[str, Any]] = []
    for cls in summary["classes"]:
        if cls["name"] not in seen_cls:
            seen_cls.add(cls["name"])
            unique_cls.append(cls)
    summary["classes"] = unique_cls
    return summary


def describe_json_schema(value: Any, depth: int = 0, max_depth: int = 3) -> str:
    """Compact human/LLM readable schema of a JSON value."""
    if isinstance(value, dict):
        if depth >= max_depth:
            return "{...}"
        items = [f"{k}: {describe_json_schema(v, depth + 1, max_depth)}" for k, v in list(value.items())[:12]]
        more = ", ..." if len(value) > 12 else ""
        return "{" + ", ".join(items) + more + "}"
    if isinstance(value, list):
        if not value:
            return "list[0]"
        return f"list[{len(value)}] of {describe_json_schema(value[0], depth + 1, max_depth)}"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    return "null"


def extract_code_summary(content: str) -> dict[str, Any]:
    """Extract classes, interfaces, functions, and imports for JS/TS/React/Flutter/Kotlin/Swift files."""
    imports: list[str] = []
    classes: list[str] = []
    functions: list[str] = []

    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//") or stripped.startswith("/*"):
            continue
        if stripped.startswith("import "):
            imports.append(stripped.rstrip(";"))
            continue
        cls_match = re.match(
            r"^(?:export\s+)?(?:default\s+)?(?:abstract\s+|final\s+|sealed\s+|data\s+)?"
            r"(class|interface|type|enum|mixin|struct|extension)\s+([A-Za-z_]\w*(?:<[^>]+>)?(?:\s+(?:extends|implements|with)\s+[^{;=]+)?)",
            stripped,
        )
        if cls_match:
            classes.append(f"{cls_match.group(1)} {cls_match.group(2).strip()}")
            continue
        fn_match = re.match(
            r"^(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function|fun|func)\s+([A-Za-z_]\w*\s*(?:<[^>]+>)?\s*\([^)]*\)(?:\s*(?:->|:)\s*[^{;=]+)?)",
            stripped,
        )
        if fn_match:
            functions.append(f"fn {fn_match.group(1).strip()}")
            continue
        arrow_match = re.match(
            r"^(?:export\s+)?(?:const|let|final)\s+([A-Za-z_]\w*)\s*(?::\s*[^=]+)?=\s*(?:async\s*)?(\([^)]*\)\s*(?::\s*[^=]+)?=>)",
            stripped,
        )
        if arrow_match:
            functions.append(f"const {arrow_match.group(1)} = {arrow_match.group(2).strip()}")
            continue
        dart_fn = re.match(
            r"^(?:Future<[^>]+>|Stream<[^>]+>|Widget|void|int|double|bool|String|List<[^>]+>|Map<[^>]+>|[A-Z]\w*)\s+([a-z_]\w*\s*\([^)]*\))(?:\s*async)?\s*\{",
            stripped,
        )
        if dart_fn:
            functions.append(f"fn {dart_fn.group(1).strip()}")

    return {
        "imports": imports[:20],
        "classes": classes[:30],
        "functions": functions[:40],
        "preview": _preview(content),
    }


def extract_file_summary(rel_path: str, content: str) -> tuple[str, dict[str, Any]]:
    suffix = PurePosixPath(rel_path).suffix.lower()
    if suffix == ".py":
        return "python", extract_python_summary(content)
    if suffix == ".json":
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            return "json", {"schema": None, "error": f"invalid JSON: {exc.msg}", "preview": _preview(content)}
        return "json", {"schema": describe_json_schema(data)[:400], "error": None}
    if suffix in CODE_SUFFIXES:
        return "code", extract_code_summary(content)
    return "text", {"preview": _preview(content)}


def _preview(content: str, limit: int = 200) -> str:
    return re.sub(r"\s+", " ", content).strip()[:limit]


def defined_names(record: dict[str, Any]) -> set[str]:
    """Every name a Python module exposes (functions, classes, globals, imports)."""
    summary = record.get("summary", {})
    names: set[str] = set()
    for sig in summary.get("functions", []):
        match = re.search(r"def\s+(\w+)", sig)
        if match:
            names.add(match.group(1))
    for cls in summary.get("classes", []):
        names.add(cls["name"])
    for glob in summary.get("globals", []):
        match = re.match(r"\s*(\w+)", glob)
        if match:
            names.add(match.group(1))
    for imp in summary.get("imports", []):
        if imp.startswith("from "):
            imported = imp.split(" import ", 1)[1]
            for part in imported.split(","):
                part = part.strip()
                names.add(part.split(" as ")[-1].strip())
        else:
            first = imp.split(" as ")
            names.add(first[-1].strip().split(".")[0])
    return names


@dataclass
class ConsistencyIssue:
    file: str
    line: int
    message: str

    def __str__(self) -> str:
        return f"{self.file}:{self.line}: {self.message}"


class Blackboard:
    """Persistent shared memory between short-lived micro-agents."""

    def __init__(self, memory_dir: str | Path, workspace: str | Path | None = None) -> None:
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.memory_dir / "blackboard.json"
        self.notes_path = self.memory_dir / "relay_notes.md"
        self.data: dict[str, Any] = self._load()
        if workspace is not None:
            self.bind_workspace(workspace)

    # ------------------------------------------------------------------ storage
    def _empty(self, workspace: Optional[str] = None) -> dict[str, Any]:
        return {
            "workspace": workspace,
            "updated_at": None,
            "files": {},
            "notes": [],
            "verified_symbols": {},
            "runtime_context": "",
        }

    def _load(self) -> dict[str, Any]:
        if self.json_path.exists():
            raw = self.json_path.read_text(encoding="utf-8").strip()
            if raw:
                try:
                    data = json.loads(raw)
                    if isinstance(data, dict):
                        base = self._empty()
                        base.update(data)
                        return base
                except json.JSONDecodeError:
                    corrupt = self.json_path.with_suffix(".corrupt.json")
                    os.replace(self.json_path, corrupt)
        return self._empty()

    def save(self) -> None:
        self.data["updated_at"] = utc_now()
        tmp = self.json_path.with_name(f".{self.json_path.name}.tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, self.json_path)

    @property
    def workspace(self) -> Optional[Path]:
        ws = self.data.get("workspace")
        return Path(ws) if ws else None

    @property
    def files(self) -> dict[str, dict[str, Any]]:
        return self.data["files"]

    @property
    def notes(self) -> list[dict[str, Any]]:
        return self.data["notes"]

    @property
    def verified_symbols(self) -> dict[str, list[str]]:
        return self.data.setdefault("verified_symbols", {})

    def set_runtime_context(self, context_markdown: str) -> None:
        """Store runtime version and package radar constraints on the Blackboard."""
        self.data["runtime_context"] = context_markdown.strip()
        self.save()

    def lock_verified_symbols(self, filepaths: list[str]) -> None:
        """Mark top-level symbols, helpers, and class methods in ``filepaths`` as verified."""
        ws = self._require_workspace()
        verified_map: dict[str, list[str]] = self.data.setdefault("verified_symbols", {})
        for filepath in filepaths:
            rel = normalize_relpath(ws, filepath)
            record = self.files.get(rel)
            if not record and (ws / rel).is_file():
                record = self.update_file_record(rel)
            if not record or record.get("kind") != "python":
                continue
            own_names = {
                n
                for n in (defined_names(record) - self._imported_names(record))
                if not (n.startswith("__") and n.endswith("__"))
            }
            for cls in record.get("summary", {}).get("classes", []):
                cls_name = cls.get("name", "")
                if not cls_name:
                    continue
                for attr_sig in cls.get("attributes", []):
                    attr_name = attr_sig.split(":", 1)[0].strip()
                    if attr_name and not (attr_name.startswith("__") and attr_name.endswith("__")):
                        own_names.add(f"{cls_name}.{attr_name}")
                for m_sig in cls.get("methods", []):
                    match = re.search(r"def\s+(\w+)", m_sig)
                    if match:
                        m_name = match.group(1)
                        if not (m_name.startswith("__") and m_name.endswith("__") and m_name != "__init__"):
                            own_names.add(f"{cls_name}.{m_name}")
            existing = set(verified_map.get(rel, []))
            verified_map[rel] = sorted(existing | own_names)
        self.save()

    def get_verified_symbols(self, filepath: str) -> set[str]:
        ws = self._require_workspace()
        rel = normalize_relpath(ws, filepath)
        verified_map: dict[str, list[str]] = self.data.get("verified_symbols", {})
        return set(verified_map.get(rel, []))

    def bind_workspace(self, workspace: str | Path, reset_if_changed: bool = True) -> bool:
        """Attach the blackboard to a workspace. Returns True if memory was reset."""
        ws = str(Path(workspace).resolve())
        if self.data.get("workspace") == ws:
            return False
        reset = reset_if_changed and self.data.get("workspace") is not None
        if reset:
            self.data = self._empty(ws)
            self._write_notes_header(ws)
        else:
            self.data["workspace"] = ws
            if not self.notes_path.exists():
                self._write_notes_header(ws)
        self.save()
        return reset

    def reset(self) -> None:
        ws = self.data.get("workspace")
        self.data = self._empty(ws)
        self._write_notes_header(ws)
        self.save()

    def _write_notes_header(self, workspace: Optional[str]) -> None:
        header = (
            "# HiveCoder Relay Notes\n\n"
            "Cumulative hand-off log written by micro-agents for the next micro-agent.\n"
            f"Workspace: `{workspace}`\n"
        )
        self.notes_path.write_text(header, encoding="utf-8")

    def _require_workspace(self) -> Path:
        ws = self.workspace
        if ws is None:
            raise RuntimeError("Blackboard is not bound to a workspace")
        return ws

    # ------------------------------------------------------------------ updates
    def update_file_record(
        self,
        filepath: str,
        agent_note: str = "",
        agent: str = "agent",
        task_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Re-extract a file's AST/schema signature and store the agent's relay note."""
        ws = self._require_workspace()
        rel = normalize_relpath(ws, filepath)
        abs_path = ws / rel
        if not abs_path.exists():
            self.remove_file_record(rel)
            raise FileNotFoundError(abs_path)
        content = abs_path.read_text(encoding="utf-8", errors="replace")
        kind, summary = extract_file_summary(rel, content)
        record = {
            "path": rel,
            "kind": kind,
            "summary": summary,
            "size": len(content),
            "sha1": hashlib.sha1(content.encode("utf-8", errors="replace")).hexdigest(),
            "updated_at": utc_now(),
            "last_agent": agent,
            "last_note": agent_note.strip(),
        }
        self.files[rel] = record
        note = agent_note.strip()
        if note:
            entry = {"ts": utc_now(), "agent": agent, "file": rel, "task_id": task_id, "note": note}
            self.notes.append(entry)
            self._append_note_markdown(entry, record)
        self.save()
        return record

    def _append_note_markdown(self, entry: dict[str, Any], record: dict[str, Any]) -> None:
        sigs = self._signature_lines(record, include_imports=False)
        lines = [
            "",
            f"### [{entry['ts']}] {entry['agent']} -> `{entry['file']}`"
            + (f" ({entry['task_id']})" if entry.get("task_id") else ""),
            "",
            entry["note"],
        ]
        if sigs:
            lines.append("")
            lines.extend(f"    {s}" for s in sigs)
        if not self.notes_path.exists():
            self._write_notes_header(self.data.get("workspace"))
        with self.notes_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

    def remove_file_record(self, filepath: str) -> bool:
        ws = self._require_workspace()
        rel = normalize_relpath(ws, filepath)
        if rel in self.files:
            del self.files[rel]
            self.verified_symbols.pop(rel, None)
            self.save()
            return True
        return False

    def scan_workspace(self, agent: str = "scanner") -> list[str]:
        """Index every text file in the workspace (no notes). Drops stale records."""
        ws = self._require_workspace()
        seen: list[str] = []
        if ws.exists():
            for root, dirs, files in os.walk(ws):
                dirs[:] = sorted(d for d in dirs if d not in IGNORED_DIRS and not d.startswith("."))
                for name in sorted(files):
                    path = Path(root) / name
                    rel = path.relative_to(ws).as_posix()
                    if path.suffix.lower() not in TEXT_SUFFIXES:
                        continue
                    try:
                        if path.stat().st_size > MAX_SCAN_BYTES:
                            continue
                    except OSError:
                        continue
                    seen.append(rel)
                    content = path.read_text(encoding="utf-8", errors="replace")
                    digest = hashlib.sha1(content.encode("utf-8", errors="replace")).hexdigest()
                    existing = self.files.get(rel)
                    if existing and existing.get("sha1") == digest:
                        continue
                    kind, summary = extract_file_summary(rel, content)
                    self.files[rel] = {
                        "path": rel,
                        "kind": kind,
                        "summary": summary,
                        "size": len(content),
                        "sha1": digest,
                        "updated_at": utc_now(),
                        "last_agent": agent,
                        "last_note": existing.get("last_note", "") if existing else "",
                    }
        for stale in [p for p in self.files if p not in seen]:
            del self.files[stale]
            self.verified_symbols.pop(stale, None)
        self.save()
        return seen

    # ------------------------------------------------------------------ reading
    def _signature_lines(self, record: dict[str, Any], include_imports: bool = True) -> list[str]:
        summary = record.get("summary", {})
        lines: list[str] = []
        if record.get("kind") == "python":
            if summary.get("syntax_error"):
                lines.append(f"!! SYNTAX ERROR {summary['syntax_error']}")
            if include_imports and summary.get("imports"):
                lines.append("imports: " + "; ".join(summary["imports"]))
            for glob in summary.get("globals", []):
                lines.append(f"const {glob}")
            for sig in summary.get("functions", []):
                lines.append(sig)
            for cls in summary.get("classes", []):
                bases = f"({', '.join(cls['bases'])})" if cls.get("bases") else ""
                lines.append(f"class {cls['name']}{bases}")
                for attr in cls.get("attributes", []):
                    lines.append(f"    attr {attr}")
                for method in cls.get("methods", []):
                    lines.append(f"    {method}")
        elif record.get("kind") == "json":
            if summary.get("error"):
                lines.append(f"!! {summary['error']}")
            else:
                lines.append(f"schema: {summary.get('schema')}")
        elif record.get("kind") == "code":
            if include_imports and summary.get("imports"):
                lines.append("imports: " + "; ".join(summary["imports"]))
            for cls_sig in summary.get("classes", []):
                lines.append(cls_sig)
            for fn_sig in summary.get("functions", []):
                lines.append(fn_sig)
            if not lines and summary.get("preview"):
                lines.append(f"preview: {summary['preview']}")
        else:
            preview = summary.get("preview", "")
            if preview:
                lines.append(f"preview: {preview}")
        return lines

    def get_compact_memory(
        self,
        target_files: Optional[list[str]] = None,
        max_notes: int = 12,
        names_only_for_others: bool = False,
    ) -> str:
        """Compact Markdown memory: AST map of the project + latest relay notes."""
        targets = set(target_files or [])
        out: list[str] = []
        ws = self.workspace
        out.append(f"## BLACKBOARD: project map ({ws.name if ws else 'unbound'})")
        if not self.files:
            out.append("(empty project: no files yet)")
        for rel in sorted(self.files):
            record = self.files[rel]
            marker = "  <- TARGET" if rel in targets else ""
            out.append(f"### {rel} [{record.get('kind')}]{marker}")
            if names_only_for_others and targets and rel not in targets and record.get("kind") == "python":
                names = sorted(n for n in defined_names(record) if not n.startswith("_"))
                own = [n for n in names if n not in self._imported_names(record)]
                out.append("- names: " + (", ".join(own) if own else "(none)"))
                continue
            for line in self._signature_lines(record, include_imports=rel in targets):
                out.append(f"- {line}")
        notes = self.notes[-max_notes:] if max_notes > 0 else []
        out.append("")
        out.append("## RELAY NOTES (oldest -> newest)")
        if not notes:
            out.append("(no relay notes yet)")
        for note in notes:
            out.append(f"- [{note['agent']} -> {note['file']}] {note['note']}")
        runtime_ctx = str(self.data.get("runtime_context") or "").strip()
        if runtime_ctx:
            out.append("")
            out.append(runtime_ctx)
        return "\n".join(out)

    @staticmethod
    def _imported_names(record: dict[str, Any]) -> set[str]:
        names: set[str] = set()
        for imp in record.get("summary", {}).get("imports", []):
            if imp.startswith("from "):
                for part in imp.split(" import ", 1)[1].split(","):
                    names.add(part.strip().split(" as ")[-1].strip())
            else:
                names.add(imp.split(" as ")[-1].strip().split(".")[0])
        return names

    def workspace_chars(self) -> int:
        """Total characters of indexed files: what a naive agent would have to read."""
        return sum(int(r.get("size", 0)) for r in self.files.values())

    def file_list(self) -> list[str]:
        return sorted(self.files)

    # ------------------------------------------------------------------ deterministic checks
    def _module_record(self, module: str) -> Optional[tuple[str, dict[str, Any]]]:
        base = module.replace(".", "/")
        for candidate in (f"{base}.py", f"{base}/__init__.py", f"src/{base}.py", f"src/{base}/__init__.py"):
            if candidate in self.files:
                return candidate, self.files[candidate]
        return None

    def _is_package(self, module: str) -> bool:
        base = module.replace(".", "/")
        return any(p.startswith(base + "/") for p in self.files) or any(
            p.startswith(f"src/{base}/") for p in self.files
        )

    def check_import_consistency(self, filepath: str) -> list[ConsistencyIssue]:
        """Detect hallucinated names imported from *workspace* modules.

        ``from app import get_flashcards`` when ``app.py`` only defines
        ``get_words`` is reported deterministically, before any LLM review.
        """
        ws = self._require_workspace()
        rel = normalize_relpath(ws, filepath)
        if not rel.endswith(".py"):
            return []
        abs_path = ws / rel
        if not abs_path.exists():
            return [ConsistencyIssue(rel, 0, "file does not exist")]
        try:
            tree = ast.parse(abs_path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            return []
        issues: list[ConsistencyIssue] = []
        module_aliases: dict[str, str] = {}
        package_parts = list(PurePosixPath(rel).parent.parts)

        guarded_imports: set[int] = set()
        guarded_attrs: set[int] = set()
        for tnode in ast.walk(tree):
            if isinstance(tnode, ast.Try):
                catches_import_err = False
                catches_attr_err = False
                for h in tnode.handlers:
                    if h.type is None:
                        continue
                    htype = _unparse(h.type)
                    if any(err in htype for err in ("ImportError", "ModuleNotFoundError")):
                        catches_import_err = True
                    if "AttributeError" in htype:
                        catches_attr_err = True
                if catches_import_err:
                    for sub in tnode.body:
                        for child in ast.walk(sub):
                            if isinstance(child, (ast.Import, ast.ImportFrom)):
                                guarded_imports.add(id(child))
                if catches_attr_err:
                    for sub in tnode.body:
                        for child in ast.walk(sub):
                            if isinstance(child, ast.Attribute):
                                guarded_attrs.add(id(child))
            elif isinstance(tnode, ast.With):
                ctx_str = " ".join(_unparse(item.context_expr) for item in tnode.items)
                if "suppress" in ctx_str and any(e in ctx_str for e in ("ImportError", "ModuleNotFoundError")):
                    for sub in tnode.body:
                        for child in ast.walk(sub):
                            if isinstance(child, (ast.Import, ast.ImportFrom)):
                                guarded_imports.add(id(child))

        for node in ast.walk(tree):
            if id(node) in guarded_imports:
                continue
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    effective_parts = (
                        package_parts[1:]
                        if (package_parts and package_parts[0] == "src")
                        else package_parts
                    )
                    if not effective_parts or node.level > len(effective_parts):
                        issues.append(
                            ConsistencyIssue(
                                rel,
                                node.lineno,
                                f"Relative import 'from {'.' * node.level}{module} import ...' goes beyond top-level package in '{rel}'; use absolute import.",
                            )
                        )
                        continue
                    keep_depth = len( package_parts) - (node.level - 1)
                    base = package_parts[:keep_depth]
                    module = ".".join([*base, module] if module else base)
                found = self._module_record(module) if module else None
                if found is None:
                    for alias in node.names:
                        sub = f"{module}.{alias.name}" if module else alias.name
                        if alias.name != "*" and self._module_record(sub):
                            module_aliases[alias.asname or alias.name] = sub
                    continue
                mod_path, record = found
                if record.get("summary", {}).get("syntax_error"):
                    continue
                available = defined_names(record)
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    sub_module = f"{module}.{alias.name}"
                    if alias.name in available or self._module_record(sub_module) or self._is_package(sub_module):
                        if self._module_record(sub_module):
                            module_aliases[alias.asname or alias.name] = sub_module
                        continue
                    hint = ", ".join(sorted(n for n in available if not n.startswith("_"))[:15])
                    issues.append(
                        ConsistencyIssue(
                            rel,
                            node.lineno,
                            f"'{alias.name}' is not defined in {mod_path}. Available names: {hint or '(none)'}",
                        )
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if self._module_record(alias.name):
                        module_aliases[alias.asname or alias.name] = alias.name

        module_dunders = {
            "__name__", "__doc__", "__file__", "__package__", "__loader__",
            "__spec__", "__path__", "__dict__", "__cached__", "__all__", "__builtins__", "__annotations__",
        }
        for node in ast.walk(tree):
            if id(node) in guarded_attrs:
                continue
            if isinstance(node, ast.Attribute) and isinstance(node.value, (ast.Name, ast.Attribute)):
                if node.attr in module_dunders:
                    continue
                expr_key = node.value.id if isinstance(node.value, ast.Name) else _unparse(node.value)
                module = module_aliases.get(expr_key)
                if not module:
                    continue
                found = self._module_record(module)
                if not found:
                    continue
                mod_path, record = found
                if record.get("summary", {}).get("syntax_error"):
                    continue
                available = defined_names(record)
                sub_attr = f"{module}.{node.attr}"
                if node.attr in available or self._module_record(sub_attr) or self._is_package(sub_attr):
                    continue
                if isinstance(node.ctx, ast.Store):
                    continue
                hint = ", ".join(sorted(n for n in available if not n.startswith("_"))[:15])
                issues.append(
                    ConsistencyIssue(
                        rel,
                        node.lineno,
                        f"'{expr_key}.{node.attr}' does not exist in {mod_path}. Available names: {hint or '(none)'}",
                    )
                )
        unique: dict[str, ConsistencyIssue] = {}
        for issue in issues:
            unique.setdefault(f"{issue.line}:{issue.message}", issue)
        return list(unique.values())

    def find_stub_markers(self, filepath: str) -> list[ConsistencyIssue]:
        """Flag unfinished-code markers while permitting Protocol, @overload, @abstractmethod, and docstrings."""
        ws = self._require_workspace()
        rel = normalize_relpath(ws, filepath)
        abs_path = ws / rel
        if not abs_path.exists() or abs_path.suffix != ".py":
            return []
        source = abs_path.read_text(encoding="utf-8", errors="replace")
        lines = source.splitlines()
        issues: list[ConsistencyIssue] = []
        comment_patterns = STUB_PATTERNS[:2]

        try:
            tree = ast.parse(source)
        except SyntaxError:
            for lineno, line in enumerate(lines, 1):
                for pattern in STUB_PATTERNS:
                    if pattern.search(line):
                        issues.append(ConsistencyIssue(rel, lineno, f"unfinished code marker: {line.strip()[:80]}"))
                        break
            return issues

        try:
            for tok in tokenize.generate_tokens(io.StringIO(source).readline):
                if tok.type == tokenize.COMMENT:
                    lineno = tok.start[0]
                    for pattern in comment_patterns:
                        if pattern.search(tok.string):
                            issues.append(
                                ConsistencyIssue(rel, lineno, f"unfinished code marker: {lines[lineno - 1].strip()[:80]}")
                            )
                            break
        except tokenize.TokenError:
            for lineno, line in enumerate(lines, 1):
                for pattern in comment_patterns:
                    if pattern.search(line):
                        issues.append(ConsistencyIssue(rel, lineno, f"unfinished code marker: {line.strip()[:80]}"))
                        break

        def _is_abstract_or_overload(fn_node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
            for dec in fn_node.decorator_list:
                dec_src = _unparse(dec).lstrip("@")
                base_dec = dec_src.split("(")[0].strip()
                if base_dec in {"abstractmethod", "abc.abstractmethod", "overload", "typing.overload"}:
                    return True
            return False

        def _check_function_body(fn_node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            if _is_abstract_or_overload(fn_node):
                return
            stmts = list(fn_node.body)
            if (
                len(stmts) > 1
                and isinstance(stmts[0], ast.Expr)
                and isinstance(getattr(stmts[0], "value", None), ast.Constant)
                and isinstance(stmts[0].value.value, str)
            ):
                stmts = stmts[1:]
            if len(stmts) == 1:
                only = stmts[0]
                if (
                    isinstance(only, ast.Expr)
                    and isinstance(getattr(only, "value", None), ast.Constant)
                    and only.value.value is Ellipsis
                ):
                    ln = only.lineno
                    issues.append(ConsistencyIssue(rel, ln, f"unfinished code marker: {lines[ln - 1].strip()[:80]}"))
                elif isinstance(only, ast.Raise) and only.exc is not None:
                    exc_src = _unparse(only.exc)
                    if exc_src == "NotImplementedError" or exc_src.startswith("NotImplementedError("):
                        ln = only.lineno
                        issues.append(ConsistencyIssue(rel, ln, f"unfinished code marker: {lines[ln - 1].strip()[:80]}"))

        for node in _iter_module_statements(tree.body):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                _check_function_body(node)
            elif isinstance(node, ast.ClassDef):
                bases = {_unparse(b) for b in node.bases}
                if "Protocol" in bases or "typing.Protocol" in bases:
                    continue
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        _check_function_body(item)

        unique: dict[int, ConsistencyIssue] = {}
        for issue in sorted(issues, key=lambda x: x.line):
            unique.setdefault(issue.line, issue)
        return list(unique.values())
