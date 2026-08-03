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
    api_version = 1
    version = __version__
    core_version_series = "3.5"

    @staticmethod
    def load_dialects(context) -> None:
        from .compiler import load_dialects

        load_dialects(context)

    @staticmethod
    def run_compiler_hook(stage: str, module, metadata: dict) -> None:
        from .compiler import run_compiler_hook

        run_compiler_hook(stage, module, metadata)

    @staticmethod
    def apply_compile_options(options: dict) -> None:
        from .compiler import apply_compile_options

        apply_compile_options(options)

    @staticmethod
    def annotate_statement(kind, generator, node, target, value) -> None:
        from .statement import annotate_statement

        annotate_statement(kind, generator, node, target, value)

    @staticmethod
    def debug_collect_start(semantic, level, addr_level):
        from .language import debug_collect_start

        return debug_collect_start(semantic, level, addr_level)

    @staticmethod
    def debug_collect_end(semantic):
        from .language import debug_collect_end

        return debug_collect_end(semantic)

    @staticmethod
    def ascend_launch_context(
        metadata, grid, stream, launch_metadata=None, kernel_args=None
    ):
        from .api import ascend_launch_context

        return ascend_launch_context(
            metadata, grid, stream, launch_metadata, kernel_args
        )


component = register_component("debugger", _DebuggerComponent())
