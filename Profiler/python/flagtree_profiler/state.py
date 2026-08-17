from functools import wraps

from .flags import get_profiling_on
from .native import runtime_binding

profiler_native = runtime_binding()


class state:
    """
    A context manager and decorator for entering and exiting a state.

    Usage:
        context manager:
        ```python
        with profiler.state("test0"):
            foo[1,](x, y)
        ```

        decorator:
        ```python
        @profiler.state("test0")
        def foo(x, y):
            ...
        ```

    Args:
        name (str): The name of the state.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self):
        if not get_profiling_on():
            return self
        profiler_native.enter_state(self.name)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if not get_profiling_on():
            return
        profiler_native.exit_state()

    def __call__(self, func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            if get_profiling_on():
                profiler_native.enter_state(self.name)
            ret = func(*args, **kwargs)
            if get_profiling_on():
                profiler_native.exit_state()
            return ret

        return wrapper


def enter_state(name: str) -> None:
    profiler_native.enter_state(name)


def exit_state() -> None:
    profiler_native.exit_state()
