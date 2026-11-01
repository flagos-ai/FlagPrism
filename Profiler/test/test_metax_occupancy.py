# FlagPrism：cross-check theoretical occupancy against the module SDK API.
import ctypes
import json
import os

import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
import triton.language as tl
from flagtree import profiler


@triton.jit
def occupancy_copy(X, Y, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(Y + i, tl.load(X + i) + 1)


@triton.jit
def occupancy_reduce(X, Y, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(Y, tl.sum(tl.load(X + i), 0))


@pytest.mark.parametrize("num_warps,block,reduce", [(1, 256, False),
                                                    (2, 256, False),
                                                    (4, 256, False),
                                                    (8, 256, False),
                                                    (4, 1024, True),
                                                    (8, 4096, True)])
def test_module_theoretical_occupancy(tmp_path, num_warps, block, reduce):
    if triton.runtime.driver.active.get_current_target().backend != "maca":
        pytest.skip("MetaX device required")
    runtime = ctypes.CDLL(
        os.path.join(os.getenv("MACA_PATH", "/opt/maca"),
                     "lib/libmcruntime.so"))
    runtime.mcModuleOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = [
        ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, ctypes.c_int,
        ctypes.c_size_t
    ]
    x = torch.ones(block, device="cuda")
    y = torch.empty(1 if reduce else block, device="cuda")
    fn = occupancy_reduce if reduce else occupancy_copy
    compiled = fn[(1, )](x, y, block, num_warps=num_warps)
    torch.cuda.synchronize()
    path = tmp_path / "occupancy"
    session = profiler.start(
        str(path),
        backend="metax",
        mode=
        "runtime_base:vendor_metrics=theoretical_occupancy,launch_stats,resource_usage"
    )
    try:
        fn[(1, )](x, y, block, num_warps=num_warps)
    finally:
        profiler.finalize(session)
    expected = x.sum().reshape(1) if reduce else x + 1
    torch.testing.assert_close(y, expected)
    document = json.loads(path.with_suffix(".vendor.json").read_text())
    kernels = [
        a for a in document["associations"] if a["source"] == "mcpti_activity"
    ]
    assert len(kernels) == 1
    metrics = kernels[0]["metrics"]
    assert "metax.theoretical_occupancy_pct" in metrics, kernels[0].get("note")
    threads = metrics["metax.block_x"] * metrics["metax.block_y"] * metrics[
        "metax.block_z"]
    assert threads == metrics["metax.threads_per_block"]
    active = ctypes.c_int()
    shared = compiled.metadata.shared
    assert runtime.mcModuleOccupancyMaxActiveBlocksPerMultiprocessor(
        ctypes.byref(active), compiled.function, threads, shared) == 0
    assert active.value == metrics["metax.active_blocks_per_sm"]
    assert shared == metrics["metax.occupancy_dynamic_shared_memory_bytes"]
    assert metrics["metax.occupancy_source"] == "maca_module_occupancy_api"
    assert metrics["metax.theoretical_occupancy_pct"] == pytest.approx(
        100 * active.value * threads / metrics["metax.max_threads_per_sm"])
    assert 0 < metrics["metax.theoretical_occupancy_pct"] <= 100


# FlagPrism end


# FlagPrism：reserve real dynamic shared memory, then compare the exact launch query.
@pytest.mark.parametrize("shared", [0, 8192, 32768])
def test_dynamic_shared_memory_occupancy(tmp_path, shared):
    if triton.runtime.driver.active.get_current_target().backend != "maca":
        pytest.skip("MetaX device required")
    runtime = ctypes.CDLL(
        os.path.join(os.getenv("MACA_PATH", "/opt/maca"),
                     "lib/libmcruntime.so"))
    runtime.mcModuleOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = [
        ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, ctypes.c_int,
        ctypes.c_size_t
    ]
    runtime.mcModuleLaunchKernel.argtypes = [
        ctypes.c_void_p
    ] + [ctypes.c_uint] * 7 + [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p
    ]
    x = torch.ones(256, device="cuda")
    y = torch.empty_like(x)
    compiled = occupancy_copy[(1, )](x, y, 256, num_warps=4)
    y.zero_()
    torch.cuda.synchronize()
    values = [
        ctypes.c_uint64(x.data_ptr()),
        ctypes.c_uint64(y.data_ptr()),
        ctypes.c_uint64(0),
        ctypes.c_uint64(0)
    ]
    params = (ctypes.c_void_p *
              len(values))(*[ctypes.addressof(v) for v in values])
    path = tmp_path / "dynamic_shared"
    session = profiler.start(
        str(path),
        backend="metax",
        mode="runtime_base:vendor_metrics=theoretical_occupancy,launch_stats")
    try:
        assert runtime.mcModuleLaunchKernel(compiled.function, 1, 1, 1, 256, 1,
                                            1, shared, None, params, None) == 0
    finally:
        profiler.finalize(session)
    torch.testing.assert_close(y, x + 1)
    report = json.loads(path.with_suffix(".vendor.json").read_text())
    captured = [
        a for a in report["associations"] if a["source"] == "mcpti_activity"
    ]
    assert len(captured) == 1
    metrics = captured[0]["metrics"]
    expected = ctypes.c_int()
    assert runtime.mcModuleOccupancyMaxActiveBlocksPerMultiprocessor(
        ctypes.byref(expected), compiled.function, 256, shared) == 0
    assert metrics["metax.active_blocks_per_sm"] == expected.value
    assert metrics["metax.occupancy_dynamic_shared_memory_bytes"] == shared


# FlagPrism end


# FlagPrism：query each kernel on its own device and restore the caller's device.
def test_multidevice_occupancy_restores_current_device(tmp_path):
    if triton.runtime.driver.active.get_current_target().backend != "maca":
        pytest.skip("MetaX device required")
    if torch.cuda.device_count() < 2:
        pytest.skip("Two MetaX devices required")
    previous = torch.cuda.current_device()
    tensors = []
    try:
        for device in (0, 1):
            with torch.cuda.device(device):
                x = torch.ones(256, device=f"cuda:{device}")
                y = torch.empty_like(x)
                occupancy_copy[(1, )](x, y, 256)
                torch.cuda.synchronize()
                tensors.append((x, y))
        path = tmp_path / "multi"
        torch.cuda.set_device(1)
        session = profiler.start(
            str(path),
            backend="metax",
            mode="runtime_base:vendor_metrics=theoretical_occupancy")
        try:
            for device in (1, 0):
                with torch.cuda.device(device):
                    occupancy_copy[(1, )](*tensors[device], 256)
        finally:
            torch.cuda.set_device(1)
            profiler.finalize(session)
        assert torch.cuda.current_device() == 1
        report = json.loads(path.with_suffix(".vendor.json").read_text())
        kernels = [
            a for a in report["associations"]
            if a["source"] == "mcpti_activity"
        ]
        assert len(kernels) == 2
        assert {a["runtime_event"]["device_id"] for a in kernels} == {0, 1}
        assert all(0 < a["metrics"]["metax.theoretical_occupancy_pct"] <= 100
                   for a in kernels)
    finally:
        torch.cuda.set_device(previous)


def test_graph_missing_occupancy_is_explicit(tmp_path):
    if triton.runtime.driver.active.get_current_target().backend != "maca":
        pytest.skip("MetaX device required")
    x = torch.ones(256, device="cuda")
    y = torch.empty_like(x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        occupancy_copy[(1, )](x, y, 256)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        occupancy_copy[(1, )](x, y, 256)
    path = tmp_path / "graph"
    session = profiler.start(
        str(path),
        backend="metax",
        mode="runtime_base:vendor_metrics=theoretical_occupancy")
    try:
        graph.replay()
    finally:
        profiler.finalize(session)
    report = json.loads(path.with_suffix(".vendor.json").read_text())
    kernels = [
        a for a in report["associations"] if a["source"] == "mcpti_activity"
    ]
    assert len(kernels) == 1
    metrics = kernels[0]["metrics"]
    if "metax.theoretical_occupancy_pct" not in metrics:
        assert "theoretical_occupancy" not in report["enabled_metrics"]
        assert "unavailable" in kernels[0]["note"].lower()
        assert any("occupancy unavailable" in reason.lower()
                   for reason in report["degrade_reasons"])
    else:
        assert 0 < metrics["metax.theoretical_occupancy_pct"] <= 100


# FlagPrism end


# FlagPrism：unloading a private module before flush must not query a stale function.
def test_unloaded_module_occupancy_is_unavailable(tmp_path):
    if triton.runtime.driver.active.get_current_target().backend != "maca":
        pytest.skip("MetaX device required")
    runtime = ctypes.CDLL(
        os.path.join(os.getenv("MACA_PATH", "/opt/maca"),
                     "lib/libmcruntime.so"))
    runtime.mcModuleLoadData.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p
    ]
    runtime.mcModuleGetFunction.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p
    ]
    runtime.mcModuleUnload.argtypes = [ctypes.c_void_p]
    runtime.mcModuleLaunchKernel.argtypes = [
        ctypes.c_void_p
    ] + [ctypes.c_uint] * 7 + [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p
    ]
    x = torch.ones(256, device="cuda")
    y = torch.empty_like(x)
    compiled = occupancy_copy[(1, )](x, y, 256, num_warps=4)
    torch.cuda.synchronize()
    image = ctypes.create_string_buffer(compiled.kernel)
    module, function = ctypes.c_void_p(), ctypes.c_void_p()
    assert runtime.mcModuleLoadData(ctypes.byref(module), image) == 0
    try:
        assert runtime.mcModuleGetFunction(ctypes.byref(function), module,
                                           compiled.name.encode()) == 0
        values = [
            ctypes.c_uint64(x.data_ptr()),
            ctypes.c_uint64(y.data_ptr()),
            ctypes.c_uint64(0),
            ctypes.c_uint64(0)
        ]
        params = (ctypes.c_void_p *
                  len(values))(*[ctypes.addressof(v) for v in values])
        path = tmp_path / "unloaded"
        session = profiler.start(
            str(path),
            backend="metax",
            mode="runtime_base:vendor_metrics=theoretical_occupancy")
        try:
            assert runtime.mcModuleLaunchKernel(function, 1, 1, 1, 256, 1, 1,
                                                0, None, params, None) == 0
            torch.cuda.synchronize()
            assert runtime.mcModuleUnload(module) == 0
            module = ctypes.c_void_p()
        finally:
            profiler.finalize(session)
        report = json.loads(path.with_suffix(".vendor.json").read_text())
        kernels = [
            a for a in report["associations"]
            if a["source"] == "mcpti_activity"
        ]
        assert len(kernels) == 1
        assert "metax.theoretical_occupancy_pct" not in kernels[0]["metrics"]
        assert "unload" in kernels[0]["note"]
        assert "theoretical_occupancy" not in report["enabled_metrics"]
    finally:
        if module:
            assert runtime.mcModuleUnload(module) == 0


# FlagPrism end
