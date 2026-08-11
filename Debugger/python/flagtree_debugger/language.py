from __future__ import annotations

from typing import Any

from .native import compiler_binding


def _constexpr_value(value: Any) -> Any:
    from triton.language import constexpr

    return value.value if isinstance(value, constexpr) else value


def _is_interpreter_builder(builder: Any) -> bool:
    return (
        type(builder).__name__ == "InterpreterBuilder"
        and type(builder).__module__.endswith(".runtime.interpreter")
    )


def debug_collect_start(semantic: Any, level: Any, addr_level: Any):
    from triton.language import void

    level_value = _constexpr_value(level)
    addr_level_value = _constexpr_value(addr_level)
    if not isinstance(level_value, int):
        raise TypeError("ftl.debug_collect_start: level must be an integer")
    if addr_level_value is not None and not isinstance(addr_level_value, int):
        raise TypeError("ftl.debug_collect_start: addr_level must be an integer")
    if addr_level_value is not None and not 0 <= addr_level_value <= 2:
        raise ValueError("ftl.debug_collect_start: addr_level must be 0, 1, or 2")
    if _is_interpreter_builder(semantic.builder):
        return semantic.tensor(None, void)
    handle = compiler_binding().create_debug_collect_begin(
        semantic.builder,
        int(level_value),
        -1 if addr_level_value is None else int(addr_level_value),
    )
    return semantic.tensor(handle, void)


def debug_collect_end(semantic: Any):
    from triton.language import void

    if _is_interpreter_builder(semantic.builder):
        return semantic.tensor(None, void)
    handle = compiler_binding().create_debug_collect_end(semantic.builder)
    return semantic.tensor(handle, void)
