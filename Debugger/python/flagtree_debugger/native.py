from __future__ import annotations

import importlib
from types import ModuleType


def _optional_module(name: str) -> ModuleType | None:
    try:
        return importlib.import_module(name)
    except (ImportError, OSError):
        return None


def runtime_binding() -> ModuleType | None:
    return _optional_module("flagtree.debugger._native")


def compiler_binding() -> ModuleType | None:
    libtriton = _optional_module("triton._C.libtriton")
    return getattr(libtriton, "debugger",
                   None) if libtriton is not None else None
