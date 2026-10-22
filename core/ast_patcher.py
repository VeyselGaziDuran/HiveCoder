"""AST-guided surgical code merger and line-window extractor for small local models.

When a small model updates an existing Python module in a multi-step plan, it may
only emit the newly requested function/class and omit functions that were already
implemented and verified in earlier micro-tasks.  :class:`ASTPatcher` parses both
versions and preserves previously verified top-level symbols while applying all
new or updated definitions from the incoming source.
"""

from __future__ import annotations

import ast
import re
from typing import Iterable, Optional


def _top_level_symbol_name(node: ast.stmt) -> Optional[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                return target.id
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def _import_key(node: ast.stmt) -> Optional[str]:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        try:
            return ast.unparse(node).strip()
        except Exception:  # noqa: BLE001
            return None
    return None


class ASTPatcher:
    """Safely merges incoming Python code with existing verified symbols."""

    @staticmethod
    def extract_top_level_names(source: str) -> set[str]:
        """Return all top-level function, class, and global variable names defined in ``source``."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return set()
        names: set[str] = set()
        for node in tree.body:
            name = _top_level_symbol_name(node)
            if name:
                names.add(name)
        return names

    @staticmethod
    def merge_preserving_symbols(
        existing_source: str,
        incoming_source: str,
        protected_symbols: Optional[Iterable[str]] = None,
    ) -> str:
        """Merge ``incoming_source`` onto ``existing_source``, retaining missing ``protected_symbols``.

        * If ``incoming_source`` has a syntax error or ``existing_source`` is empty,
          ``incoming_source`` is returned as-is so the validator/Infirmary can handle it.
        * Any symbol in ``protected_symbols`` (or any existing top-level function/class if
          ``protected_symbols`` is None) that is missing from ``incoming_source`` is
          preserved alongside its required imports.
        """
        if not existing_source.strip():
            return incoming_source
        try:
            old_tree = ast.parse(existing_source)
            new_tree = ast.parse(incoming_source)
        except SyntaxError:
            return incoming_source

        protected: set[str] = (
            set(protected_symbols)
            if protected_symbols is not None
            else ASTPatcher.extract_top_level_names(existing_source)
        )
        if not protected:
            return incoming_source

        new_names = ASTPatcher.extract_top_level_names(incoming_source)
        missing = protected - new_names
        if not missing:
            return incoming_source

        old_lines = existing_source.splitlines()
        new_import_keys = {k for node in new_tree.body if (k := _import_key(node)) is not None}

        preserved_imports: list[str] = []
        preserved_blocks: list[str] = []

        for node in old_tree.body:
            imp_key = _import_key(node)
            if imp_key is not None and imp_key not in new_import_keys:
                preserved_imports.append(imp_key)
                continue
            sym = _top_level_symbol_name(node)
            if sym in missing and hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                start = max(0, int(node.lineno) - 1)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.decorator_list:
                    first_dec = min(
                        (int(d.lineno) - 1 for d in node.decorator_list if hasattr(d, "lineno")),
                        default=start,
                    )
                    start = max(0, first_dec)
                end = int(node.end_lineno or node.lineno)
                snippet = "\n".join(old_lines[start:end]).rstrip()
                if snippet:
                    preserved_blocks.append(snippet)

        if not preserved_blocks:
            return incoming_source

        merged_parts: list[str] = []
        if preserved_imports:
            merged_parts.append("\n".join(preserved_imports))
        merged_parts.extend(preserved_blocks)
        merged_parts.append(incoming_source.strip())
        return "\n\n\n".join(part for part in merged_parts if part.strip()) + "\n"

    @staticmethod
    def extract_traceback_window(source: str, traceback_text: str, radius: int = 5) -> str:
        """Extract ``±radius`` lines around line numbers referenced in ``traceback_text``."""
        if not source.strip() or not traceback_text.strip():
            return ""
        line_numbers: list[int] = []
        for match in re.finditer(r"(?:line\s+|:)(\d+)\b", traceback_text):
            lineno = int(match.group(1))
            if lineno not in line_numbers:
                line_numbers.append(lineno)

        lines = source.splitlines()
        if not lines or not line_numbers:
            return ""

        windows: list[str] = []
        for lineno in line_numbers[:3]:
            if 1 <= lineno <= len(lines):
                start = max(1, lineno - radius)
                end = min(len(lines), lineno + radius)
                snippet_lines = [
                    f"{'>>' if idx == lineno else '  '} {idx:4d} | {lines[idx - 1]}"
                    for idx in range(start, end + 1)
                ]
                windows.append(f"Lines {start}-{end} (around line {lineno}):\n" + "\n".join(snippet_lines))
        return "\n\n".join(windows)
