from __future__ import annotations

import ast
from typing import Any


def _statement_source(generator: Any, node: ast.AST) -> str:
    source = None
    if hasattr(node, "lineno"):
        try:
            source = ast.get_source_segment(generator.jit_fn.src, node)
        except Exception:
            source = None
    if source is None and hasattr(ast, "unparse"):
        try:
            source = ast.unparse(node)
        except Exception:
            source = None
    return "" if source is None else " ".join(source.strip().split())


def _statement_id(generator: Any, node: ast.AST) -> int:
    line = int(generator.begin_line + getattr(node, "lineno", 0))
    col = int(getattr(node, "col_offset", 0))
    return max(0, min(line * 1000 + min(col, 999), (1 << 31) - 1))


def _operation_handle(generator: Any, value: Any):
    from triton.language import tensor

    if not isinstance(value, tensor):
        return None
    handle = getattr(value, "handle", None)
    if handle is not None:
        return handle
    get_last_op = getattr(generator.builder, "get_last_op", None)
    return get_last_op() if callable(get_last_op) else None


def _annotate_value(
    generator: Any,
    node: ast.AST,
    value: Any,
    source: str,
    name: str | None = None,
) -> None:
    handle = _operation_handle(generator, value)
    if handle is None or not hasattr(handle, "set_attr"):
        return
    if source:
        handle.set_attr(
            "flagtree.debug.triton_statement",
            generator.builder.get_string_attr(source),
        )
    if name is not None:
        handle.set_attr(
            "flagtree.debug.statement_result_name",
            generator.builder.get_string_attr(str(name)),
        )
    handle.set_attr(
        "flagtree.debug.statement_id",
        generator.builder.get_int32_attr(_statement_id(generator, node)),
    )


def _annotate_target(
    generator: Any,
    node: ast.AST,
    target: ast.AST,
    value: Any,
    source: str,
) -> None:
    if isinstance(target, ast.Name):
        _annotate_value(generator, node, value, source, target.id)
        return
    if isinstance(target, ast.Tuple):
        values = getattr(value, "values", ())
        for child, child_value in zip(target.elts, values):
            _annotate_target(generator, node, child, child_value, source)


def annotate_statement(
    kind: str,
    generator: Any,
    node: ast.AST,
    target: ast.AST | None,
    value: Any,
) -> None:
    source = _statement_source(generator, node)
    if kind == "assignment":
        if target is not None:
            _annotate_target(generator, node, target, value, source)
        return
    if kind == "expression":
        _annotate_value(generator, node, value, source)
        return
    raise ValueError(f"unsupported statement metadata event: {kind}")
