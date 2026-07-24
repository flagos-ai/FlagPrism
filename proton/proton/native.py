from __future__ import annotations

import importlib
from types import ModuleType


def _optional_module(name: str) -> ModuleType | None:
    try:
        return importlib.import_module(name)
    except (ImportError, OSError):
        return None


def runtime_binding() -> ModuleType:
    _optional_module("triton._C.libtriton")
    native = _optional_module("flagtree_profiler.libproton")
    if native is None:
        native = _optional_module("triton._C.libproton")
    binding = getattr(native, "proton", None) if native is not None else None
    if binding is None:
        raise RuntimeError(
            "FlagTree profiler native support is unavailable. Install a "
            "flagtree-profiler wheel compatible with the installed FlagTree version."
        )
    return binding


def compiler_binding() -> ModuleType:
    native = _optional_module("flagtree_profiler.libproton")
    binding = getattr(native, "compiler", None) if native is not None else None
    if binding is not None:
        return binding

    # Transitional compatibility with core wheels that still embed Proton.
    libtriton = _optional_module("triton._C.libtriton")
    binding = getattr(libtriton, "proton", None) if libtriton is not None else None
    if binding is None:
        raise RuntimeError(
            "FlagTree profiler compiler interface is unavailable in this FlagTree build."
        )
    return binding
