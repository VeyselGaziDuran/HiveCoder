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
import textwrap
from typing import Iterable, Optional


def _assigned_target_names(target: ast.AST) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Starred):
        return _assigned_target_names(target.value)
    if isinstance(target, (ast.Tuple, ast.List)):
        names: list[str] = []
        for elt in target.elts:
            names.extend(_assigned_target_names(elt))
        return names
    return []


def _node_symbol_names(node: ast.stmt) -> list[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.Assign):
        names: list[str] = []
        for target in node.targets:
            names.extend(_assigned_target_names(target))
        return names
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id]
    if hasattr(ast, "TypeAlias") and isinstance(node, getattr(ast, "TypeAlias")):
        alias_id = getattr(getattr(node, "name", None), "id", None)
        return [alias_id] if alias_id else []
    if isinstance(node, ast.If) and not _is_main_guard(node):
        cond_names: list[str] = []
        for sub in (*node.body, *node.orelse):
            cond_names.extend(_node_symbol_names(sub))
        return list(dict.fromkeys(cond_names))
    if isinstance(node, ast.Try):
        try_names: list[str] = []
        for sub in (
            *node.body,
            *(s for h in node.handlers for s in h.body),
            *node.orelse,
            *node.finalbody,
        ):
            try_names.extend(_node_symbol_names(sub))
        return list(dict.fromkeys(try_names))
    if isinstance(node, ast.With):
        with_names: list[str] = []
        for sub in node.body:
            with_names.extend(_node_symbol_names(sub))
        return list(dict.fromkeys(with_names))
    return []


def _top_level_symbol_name(node: ast.stmt) -> Optional[str]:
    names = _node_symbol_names(node)
    return names[0] if names else None


def _raw_imported_bound_names(node: ast.stmt) -> set[str]:
    if isinstance(node, ast.Import):
        return {(a.asname or a.name.split(".")[0]) for a in node.names}
    if isinstance(node, ast.ImportFrom):
        return {(a.asname or a.name) for a in node.names if a.name != "*"}
    if isinstance(node, ast.If) and not _is_main_guard(node):
        if_bound: set[str] = set()
        for sub in (*node.body, *node.orelse):
            if_bound.update(_raw_imported_bound_names(sub))
        return if_bound
    if isinstance(node, ast.Try):
        bound: set[str] = set()
        for sub in (
            *node.body,
            *(s for h in node.handlers for s in h.body),
            *node.orelse,
            *node.finalbody,
        ):
            bound.update(_raw_imported_bound_names(sub))
        return bound
    if isinstance(node, ast.With):
        with_bound: set[str] = set()
        for sub in node.body:
            with_bound.update(_raw_imported_bound_names(sub))
        return with_bound
    return set()


def _is_import_guard_block(node: ast.stmt) -> bool:
    if not (
        isinstance(node, (ast.Try, ast.With))
        or (isinstance(node, ast.If) and not _is_main_guard(node))
    ):
        return False
    imp_bound = _raw_imported_bound_names(node)
    if not imp_bound:
        return False
    for child in ast.walk(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return False
    return set(_node_symbol_names(node)).issubset(imp_bound)


def _imported_bound_names(node: ast.stmt) -> set[str]:
    if isinstance(node, (ast.Import, ast.ImportFrom)) or _is_import_guard_block(node):
        return _raw_imported_bound_names(node)
    return set()


def _import_key(node: ast.stmt) -> Optional[str]:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        try:
            return ast.unparse(node).strip()
        except Exception:  # noqa: BLE001
            return None
    return None


def _node_line_range(node: ast.stmt) -> tuple[int, int]:
    start = max(0, int(getattr(node, "lineno", 1)) - 1)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.decorator_list:
        first_dec = min(
            (int(d.lineno) - 1 for d in node.decorator_list if hasattr(d, "lineno")),
            default=start,
        )
        start = max(0, first_dec)
    end = int(getattr(node, "end_lineno", None) or getattr(node, "lineno", 1))
    return start, end


def _is_main_guard(node: ast.stmt) -> bool:
    if not isinstance(node, ast.If):
        return False
    try:
        test_str = ast.unparse(node.test).replace("'", '"')
        return '__name__' in test_str and '__main__' in test_str
    except Exception:  # noqa: BLE001
        return False


class ASTPatcher:
    """Safely merges incoming Python code with existing verified symbols and class methods."""

    @staticmethod
    def extract_top_level_names(source: str) -> set[str]:
        """Return all top-level function, class, and global variable names defined in ``source``."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return set()
        names: set[str] = set()
        for node in tree.body:
            names.update(_node_symbol_names(node))
        return names

    @staticmethod
    def extract_all_symbols(source: str) -> set[str]:
        """Return top-level symbol names plus ``ClassName.method_name`` entries."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return set()
        names: set[str] = set()
        for node in tree.body:
            names.update(_node_symbol_names(node))
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    for m_name in _node_symbol_names(item):
                        names.add(f"{node.name}.{m_name}")
        return names

    @staticmethod
    def merge_preserving_symbols(
        existing_source: str,
        incoming_source: str,
        protected_symbols: Optional[Iterable[str]] = None,
    ) -> str:
        """Merge ``incoming_source`` onto ``existing_source``, retaining missing ``protected_symbols``.

        * Preserves missing verified top-level functions, classes, and constants.
        * Preserves missing verified methods inside existing classes (``ClassName.method``).
        * Keeps module docstrings and ``from __future__ import ...`` at the top of the file,
          places preserved imports after the header, and appends preserved top-level blocks
          before ``if __name__ == "__main__":`` (or at EOF).
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
            else ASTPatcher.extract_all_symbols(existing_source)
        )
        if not protected:
            return incoming_source

        old_lines = existing_source.splitlines()
        working_lines = incoming_source.splitlines()

        # 1. Surgical Class-Level Method Preservation (processed bottom-up to keep line numbers stable)
        old_classes: dict[str, ast.ClassDef] = {
            node.name: node for node in old_tree.body if isinstance(node, ast.ClassDef)
        }
        new_classes: list[ast.ClassDef] = [
            node for node in new_tree.body if isinstance(node, ast.ClassDef) and node.name in old_classes
        ]
        methods_restored = False
        for new_cls in sorted(new_classes, key=lambda c: int(c.end_lineno or c.lineno), reverse=True):
            old_cls = old_classes[new_cls.name]
            new_member_names = {
                name
                for item in new_cls.body
                for name in _node_symbol_names(item)
            }
            raw_indent = (
                int(getattr(new_cls.body[0], "col_offset", new_cls.col_offset + 4))
                if new_cls.body
                else (new_cls.col_offset + 4)
            )
            body_indent = raw_indent if raw_indent > new_cls.col_offset else (new_cls.col_offset + 4)
            indent_prefix = " " * body_indent
            missing_attr_snippets: list[str] = []
            missing_method_snippets: list[str] = []
            for item in old_cls.body:
                member_names = _node_symbol_names(item)
                if not member_names:
                    continue
                is_protected = any(
                    f"{new_cls.name}.{m_name}" in protected
                    or (
                        new_cls.name in protected
                        and not (m_name.startswith("__") and m_name.endswith("__") and m_name != "__init__")
                    )
                    for m_name in member_names
                )
                if is_protected and not (set(member_names) & new_member_names):
                    m_start, m_end = _node_line_range(item)
                    raw_snippet = "\n".join(old_lines[m_start:m_end]).rstrip()
                    if raw_snippet:
                        reindented = textwrap.indent(textwrap.dedent(raw_snippet), indent_prefix)
                        if isinstance(item, (ast.Assign, ast.AnnAssign)):
                            missing_attr_snippets.append(reindented)
                        else:
                            missing_method_snippets.append(reindented)
            if missing_attr_snippets or missing_method_snippets:
                if missing_method_snippets:
                    insert_idx = int(new_cls.end_lineno or new_cls.lineno)
                    injected = [""] + missing_method_snippets
                    working_lines[insert_idx:insert_idx] = "\n".join(injected).splitlines()
                if missing_attr_snippets:
                    if not new_cls.body:
                        cls_header_end = int(new_cls.lineno)
                    elif (
                        isinstance(new_cls.body[0], ast.Expr)
                        and isinstance(getattr(new_cls.body[0], "value", None), ast.Constant)
                        and isinstance(new_cls.body[0].value.value, str)
                    ):
                        cls_header_end = int(new_cls.body[0].end_lineno or new_cls.body[0].lineno)
                    else:
                        cls_header_end = _node_line_range(new_cls.body[0])[0]
                    working_lines[cls_header_end:cls_header_end] = "\n".join(missing_attr_snippets).splitlines()
                methods_restored = True

        working_source = "\n".join(working_lines) + ("\n" if incoming_source.endswith("\n") else "")
        try:
            working_tree = ast.parse(working_source)
        except SyntaxError:
            working_source = incoming_source
            working_tree = new_tree
            working_lines = incoming_source.splitlines()
            methods_restored = False

        # 2. Top-Level Symbol & Import Preservation
        top_protected = {s for s in protected if "." not in s}
        new_top_names = ASTPatcher.extract_top_level_names(working_source)
        missing_top = top_protected - new_top_names

        if not missing_top and not methods_restored:
            return incoming_source

        new_import_keys = {k for node in working_tree.body if (k := _import_key(node)) is not None}
        existing_working_names = new_top_names | {
            name for node in working_tree.body for name in _imported_bound_names(node)
        }
        preserved_future_imports: list[str] = []
        preserved_imports: list[str] = []
        preserved_blocks: list[str] = []

        for node in old_tree.body:
            imp_key = _import_key(node)
            if imp_key is not None and imp_key not in new_import_keys:
                if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                    preserved_future_imports.append(imp_key)
                else:
                    bound = _imported_bound_names(node)
                    if not bound or not bound.issubset(existing_working_names):
                        preserved_imports.append(imp_key)
                continue
            if _is_import_guard_block(node):
                bound = _imported_bound_names(node)
                if bound and not bound.issubset(existing_working_names):
                    start, end = _node_line_range(node)
                    snippet = "\n".join(old_lines[start:end]).rstrip()
                    if snippet:
                        preserved_imports.append(snippet)
                continue
            node_syms = _node_symbol_names(node)
            if node_syms and any(s in missing_top for s in node_syms):
                start, end = _node_line_range(node)
                snippet = "\n".join(old_lines[start:end]).rstrip()
                if snippet:
                    preserved_blocks.append(snippet)

        if not preserved_blocks and not (missing_top and preserved_imports) and not methods_restored:
            return incoming_source

        # Find safe header boundary (docstring + __future__) and leading imports in working_tree
        future_end_line = 0
        imports_end_line = 0
        in_future_phase = True
        for idx, node in enumerate(working_tree.body):
            if (
                idx == 0
                and isinstance(node, ast.Expr)
                and isinstance(getattr(node, "value", None), ast.Constant)
                and isinstance(node.value.value, str)
            ):
                future_end_line = int(node.end_lineno or node.lineno)
                imports_end_line = future_end_line
                continue
            if in_future_phase and isinstance(node, ast.ImportFrom) and node.module == "__future__":
                future_end_line = int(node.end_lineno or node.lineno)
                imports_end_line = future_end_line
                continue
            in_future_phase = False
            if isinstance(node, (ast.Import, ast.ImportFrom)) or _is_import_guard_block(node):
                imports_end_line = int(node.end_lineno or node.lineno)
                continue
            break

        # Find optional `if __name__ == "__main__":` guard at the end of working_tree
        main_guard_start: Optional[int] = None
        if working_tree.body and _is_main_guard(working_tree.body[-1]):
            main_guard_start, _ = _node_line_range(working_tree.body[-1])

        future_part = "\n".join(working_lines[:future_end_line]).strip()
        working_imports_part = "\n".join(working_lines[future_end_line:imports_end_line]).strip()
        if main_guard_start is not None and main_guard_start >= imports_end_line:
            body_part = "\n".join(working_lines[imports_end_line:main_guard_start]).strip()
            footer_part = "\n".join(working_lines[main_guard_start:]).strip()
        else:
            body_part = "\n".join(working_lines[imports_end_line:]).strip()
            footer_part = ""

        sections: list[str] = []
        if future_part:
            sections.append(future_part)
        if preserved_future_imports:
            sections.append("\n".join(preserved_future_imports))
        if preserved_imports:
            sections.append("\n".join(preserved_imports))
        if working_imports_part:
            sections.append(working_imports_part)
        if preserved_blocks:
            sections.extend(preserved_blocks)
        if body_part:
            sections.append(body_part)
        if footer_part:
            sections.append(footer_part)

        merged = "\n\n\n".join(sec for sec in sections if sec.strip()) + "\n"
        try:
            ast.parse(merged)
            return merged
        except SyntaxError:
            return working_source if methods_restored else incoming_source

    @staticmethod
    def normalize_root_relative_imports(source: str, rel_path: str) -> str:
        """Convert invalid root-level or beyond-top-level relative imports to absolute imports."""
        if not rel_path.endswith(".py"):
            return source
        depth = rel_path.count("/")
        if depth == 0:
            return re.sub(
                r"^(\s*)from\s+\.+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s+import\s+",
                r"\1from \2 import ",
                source,
                flags=re.MULTILINE,
            )
        if depth == 1:
            return re.sub(
                r"^(\s*)from\s+\.{2,}([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s+import\s+",
                r"\1from \2 import ",
                source,
                flags=re.MULTILINE,
            )
        return source

    @staticmethod
    def extract_traceback_window(
        source: str,
        traceback_text: str,
        radius: int = 5,
        filepath: str = "",
    ) -> str:
        """Extract ``±radius`` lines around line numbers referenced in ``traceback_text``."""
        if not source.strip() or not traceback_text.strip():
            return ""
        line_numbers: list[int] = []
        if filepath:
            basename = filepath.rsplit("/", 1)[-1]
            for pat in (filepath, basename):
                for match in re.finditer(rf"{re.escape(pat)}['\", )]*?(?:,\s*line\s+|:)(\d+)\b", traceback_text):
                    lineno = int(match.group(1))
                    if lineno not in line_numbers:
                        line_numbers.append(lineno)
        if not line_numbers:
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
