from __future__ import annotations

import importlib
from types import ModuleType


def _optional_module(name: str) -> ModuleType | None:
    try:
        return importlib.import_module(name)
    except (ImportError, OSError):
        return None


def runtime_binding() -> ModuleType | None:
    # Import core first so libtriton is loaded before the component resolves
    # its ABI symbols and pybind cross-module types.
    libtriton = _optional_module("triton._C.libtriton")
    native = _optional_module("flagtree_debugger._native")
    if native is not None:
        return native

    # Transitional compatibility with FlagTree builds that still embed the
    # native implementation in libtriton.
    return getattr(libtriton, "debugger", None) if libtriton is not None else None


def compiler_binding() -> ModuleType | None:
    libtriton = _optional_module("triton._C.libtriton")
    native = _optional_module("flagtree_debugger._native")
    if native is not None:
        return native

    # Transitional support for an early split-wheel layout that emitted a
    # second compiler-only extension.
    native = _optional_module("flagtree_debugger._compiler")
    if native is not None:
        return native

    passes = getattr(libtriton, "passes", None) if libtriton is not None else None
    return getattr(passes, "flagtree_debug", None)
