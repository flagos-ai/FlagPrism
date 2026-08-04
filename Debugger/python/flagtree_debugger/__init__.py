from __future__ import annotations

from triton import __version__
from triton._flagprism import register_component

from . import compiler as _compiler_module
from . import native as _native_module
from . import runtime as _runtime_module
from .api import *  # noqa: F403
from .api import __all__


class _DebuggerComponent:
    name = "debugger"
    api_version = (2, 0)
    version = __version__
    required_capabilities = frozenset(
        {
            "compiler.dialects.v1",
            "compiler.events.v1",
            "compiler.options.v1",
            "frontend.statement_events.v1",
            "language.debug_collect.v1",
            "runtime.launch_context.v1",
        }
    )

    @staticmethod
    def load_dialects(context) -> None:
        from .compiler import load_dialects

        load_dialects(context)

    @staticmethod
    def on_compiler_event(event) -> None:
        from .compiler import run_compiler_event

        run_compiler_event(event)

    @staticmethod
    def apply_compile_options(options: dict) -> None:
        from .compiler import apply_compile_options

        apply_compile_options(options)

    @staticmethod
    def on_statement_event(event) -> None:
        from .statement import annotate_statement

        annotate_statement(event)

    @staticmethod
    def debug_collect_start(semantic, level, addr_level):
        from .language import debug_collect_start

        return debug_collect_start(semantic, level, addr_level)

    @staticmethod
    def debug_collect_end(semantic):
        from .language import debug_collect_end

        return debug_collect_end(semantic)

    @staticmethod
    def launch_context(event):
        from .api import launch_context

        return launch_context(
            event.backend,
            event.metadata,
            event.grid,
            event.stream,
            event.launch_metadata,
            event.kernel_args,
        )


component = register_component("debugger", _DebuggerComponent())
