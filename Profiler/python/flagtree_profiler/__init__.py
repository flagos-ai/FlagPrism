# ruff: noqa
from __future__ import annotations

from triton import __version__
from flagtree._flagprism import register_component

from .scope import scope, cpu_timed_scope, enter_scope, exit_scope
from .state import state, enter_state, exit_state
from .profile import (
    start,
    activate,
    deactivate,
    finalize,
    profile,
    DEFAULT_PROFILE_NAME,
)
from . import context, specs, mode

__all__ = (
    "DEFAULT_PROFILE_NAME",
    "activate",
    "context",
    "cpu_timed_scope",
    "deactivate",
    "enter_scope",
    "enter_state",
    "exit_scope",
    "exit_state",
    "finalize",
    "mode",
    "profile",
    "scope",
    "specs",
    "start",
    "state",
)


class _ProfilerComponent:
    name = "profiler"
    api_version = (2, 0)
    version = __version__
    required_capabilities = frozenset({"compiler.dialects.v1"})

    @staticmethod
    def load_dialects(context) -> None:
        from .native import compiler_binding

        compiler_binding().load_dialects(context)


component = register_component("profiler", _ProfilerComponent())
