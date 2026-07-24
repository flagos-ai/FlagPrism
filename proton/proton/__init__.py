# ruff: noqa
from __future__ import annotations

import sys
from importlib import metadata

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


try:
    __version__ = metadata.version("flagtree-profiler")
except metadata.PackageNotFoundError:
    __version__ = "0.1.0"

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
    api_version = 1
    version = __version__
    core_version_series = "3.5"

    @staticmethod
    def module():
        return sys.modules[__name__]

    @staticmethod
    def load_dialects(context) -> None:
        from .native import compiler_binding

        compiler_binding().load_dialects(context)

component = _ProfilerComponent()
