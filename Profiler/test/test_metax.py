# FlagPrism：MCPTI pause/resume membership and rejected-session retry on a real device.
"""MetaX session isolation, including kernels without Triton scopes."""

import json

import pytest


def test_metax_session_launch_membership(tmp_path):
    torch = pytest.importorskip("torch")
    triton = pytest.importorskip("triton")
    import triton.language as tl
    from flagtree import profiler

    if triton.runtime.driver.active.get_current_target().backend != "maca":
        pytest.skip("MetaX device required")

    @triton.jit
    def session_probe(X, Y, TAG: tl.constexpr):
        i = tl.arange(0, 32)
        x = tl.load(X + i)
        tl.store(Y + i, x + TAG)

    x = torch.ones(32, device="cuda")
    y = torch.empty_like(x)
    # Compile before collecting to avoid timing compilation or initialization.
    for tag in range(5):
        session_probe[(1, )](x, y, tag)
    torch.cuda.synchronize()
    paths = [tmp_path / "first", tmp_path / "second"]
    # hook=None exercises raw MCPTI activities without scope correlation.
    first = profiler.start(
        str(paths[0]),
        backend="metax",
        context="shadow",
        data="tree",
        mode="runtime_base:runtime_host_timing_fallback=false",
    )
    second = None
    try:
        session_probe[(1, )](x, y, 0)  # first only
        profiler.deactivate(first)
        session_probe[(1, )](x, y, 1)  # paused: neither
        # Vendor adapters currently reject overlapping sessions, even paused
        # ones. Preserve that API contract rather than enabling it implicitly.
        with pytest.raises(RuntimeError, match="does not support overlapping"):
            profiler.start(
                str(paths[1]),
                backend="metax",
                context="shadow",
                data="tree",
                mode="runtime_base:runtime_host_timing_fallback=false",
            )
        profiler.activate(first)
        session_probe[(1, )](x, y, 3)  # resumed first
        session_probe[(1, )](x, y, 4)  # first only
    finally:
        profiler.finalize(first)
        if second is not None:
            profiler.finalize(second)
    second = profiler.start(
        str(paths[1]),
        backend="metax",
        context="shadow",
        data="tree",
        mode="runtime_base:runtime_host_timing_fallback=false",
    )
    try:
        session_probe[(1, )](x, y, 2)
    finally:
        profiler.finalize(second)
    events = []
    for path in paths:
        document = json.loads(path.with_suffix(".vendor.json").read_text())
        # Assert exact launch membership, not merely a nonempty timing report.
        associations = document["associations"]
        events.append({
            a["runtime_event"]["correlation_id"]
            for a in associations
            if "session_probe" in a["runtime_event"]["op_name"]
        })
    assert len(events[0]) == 3
    assert len(events[1]) == 1
    assert not events[0] & events[1]


# FlagPrism end


# FlagPrism：exercise startup rollback, current-device metadata and IR routing.
def test_metax_failed_hook_is_retryable(tmp_path, monkeypatch):
    triton = pytest.importorskip("triton")
    if triton.runtime.driver.active.get_current_target().backend != "maca":
        pytest.skip("MetaX device required")
    import importlib
    api = importlib.import_module("flagtree.profiler.profile")
    original = api.HookManager.register

    def fail(*args, **kwargs):
        raise RuntimeError("injected hook registration failure")

    monkeypatch.setattr(api.HookManager, "register", fail)
    with pytest.raises(RuntimeError, match="injected hook"):
        api.start(str(tmp_path / "failed"), backend="metax", hook="triton")
    monkeypatch.setattr(api.HookManager, "register", original)
    session = api.start(str(tmp_path / "retry"), backend="metax")
    api.finalize(session)
    assert not api._active_sessions


@pytest.mark.parametrize("hook", ["triton", "instrumentation"])
def test_metax_ir_and_native_metrics(tmp_path, hook):
    torch = pytest.importorskip("torch")
    triton = pytest.importorskip("triton")
    import triton.language as tl
    from flagtree import profiler
    if triton.runtime.driver.active.get_current_target().backend != "maca":
        pytest.skip("MetaX device required")

    @triton.jit
    def ir_probe(X, Y):
        i = tl.arange(0, 32)
        tl.store(Y + i, tl.load(X + i) + 1)

    previous = torch.cuda.current_device()
    device = 1 if torch.cuda.device_count() > 1 else previous
    path = tmp_path / hook
    try:
        torch.cuda.set_device(device)
        x = torch.ones(32, device="cuda")
        y = torch.empty_like(x)
        session = profiler.start(
            str(path),
            backend="metax",
            hook=hook,
            mode=
            "runtime_base:vendor_metrics=launch_stats,resource_usage,kernel_duration,unsupported_counter:runtime_host_timing_fallback=false"
        )
        try:
            ir_probe[(1, )](x, y)
        finally:
            profiler.finalize(session)
        torch.testing.assert_close(y, x + 1)
        meta = json.loads(path.with_suffix(".meta.json").read_text())
        vendor = json.loads(path.with_suffix(".vendor.json").read_text())
        assert meta["device"]["id"] == device
        native = [
            a for a in vendor["associations"]
            if a["source"] == "mcpti_activity"
        ]
        assert native
        event = native[0]["runtime_event"]
        assert event["device_id"] == device
        assert event["end_time_ns"] > event["start_time_ns"] > 0
        metrics = native[0]["metrics"]
        assert metrics["metax.grid_x"] == 1
        assert metrics["metax.block_x"] > 0
        assert metrics["metax.registers_per_thread"] > 0
        assert metrics["metax.kernel_duration_ns"] > 0
        ir = [
            a for a in vendor["associations"]
            if a["source"] == "flagtree_ir_kernel"
        ]
        assert ir and ir[0]["metrics"]["op_event_count"] > 0
        assert all(a["runtime_event"]["device_id"] == device for a in ir)
        assert ir[0]["metrics"]["memory_read_bytes"] > 0
        assert ir[0]["metrics"]["memory_write_bytes"] > 0
        assert any("unsupported_counter" in reason
                   for reason in vendor["degrade_reasons"])
    finally:
        torch.cuda.set_device(previous)


# FlagPrism end
