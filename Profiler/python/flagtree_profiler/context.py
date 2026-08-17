from .flags import get_profiling_on
from .native import runtime_binding

profiler_native = runtime_binding()


def depth(session: int | None = 0) -> int | None:
    """
    Get the depth of the context.

    Args:
        session (int): The session ID of the profiling session. Defaults to 0.

    Returns:
        depth (int or None): The depth of the context. If profiling is off, returns None.
    """
    if not get_profiling_on():
        return None
    return profiler_native.get_context_depth(session)
