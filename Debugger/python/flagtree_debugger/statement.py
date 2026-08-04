from __future__ import annotations

from typing import Any

from .native import compiler_binding


def _annotate_value(
    builder: Any,
    value: Any,
    source: str,
    statement_id: int,
    name: str | None = None,
) -> None:
    from triton.language import tensor

    if not isinstance(value, tensor):
        return
    handle = getattr(value, "handle", None)
    if handle is None:
        if type(builder).__name__ == "InterpreterBuilder":
            return
        binding = compiler_binding()
        annotate_operation = (
            getattr(binding, "annotate_statement_operation", None)
            if binding is not None
            else None
        )
        if callable(annotate_operation):
            annotate_operation(
                builder,
                source,
                name,
                statement_id,
            )
        return
    if not hasattr(handle, "set_attr"):
        return
    if source:
        handle.set_attr(
            "flagtree.debug.triton_statement",
            builder.get_string_attr(source),
        )
    if name is not None:
        handle.set_attr(
            "flagtree.debug.statement_result_name",
            builder.get_string_attr(str(name)),
        )
    handle.set_attr(
        "flagtree.debug.statement_id",
        builder.get_int32_attr(statement_id),
    )


def annotate_statement(event: Any) -> None:
    if event.kind not in {"assignment", "expression"}:
        raise ValueError(f"unsupported statement metadata event: {event.kind}")
    for result in event.results:
        _annotate_value(
            event.builder,
            result.value,
            event.source,
            event.statement_id,
            result.name,
        )
