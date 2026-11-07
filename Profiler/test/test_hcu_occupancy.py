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
    if ".hcu." not in type(triton.runtime.driver.active).__module__:
        pytest.skip("HCU device required")
    runtime = ctypes.CDLL("/opt/dtk/lib/libamdhip64.so")
    runtime.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = [
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
        backend="hcu",
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
        a for a in document["associations"] if a["source"] == "hcu_activity"
    ]
    assert len(kernels) == 1
    metrics = kernels[0]["metrics"]
    assert "hcu.theoretical_occupancy_pct" in metrics, kernels[0].get("note")
    threads = metrics["hcu.block_x"] * metrics["hcu.block_y"] * metrics[
        "hcu.block_z"]
    assert threads == metrics["hcu.threads_per_block"]
    active = ctypes.c_int()
    shared = compiled.metadata.shared
    assert runtime.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
        ctypes.byref(active), compiled.function, threads, shared) == 0
    assert active.value == metrics["hcu.active_blocks_per_cu"]
    assert shared == metrics["hcu.dynamic_shared_memory_bytes"]
    assert metrics["hcu.occupancy_source"] == "hip_module_occupancy_api"
    assert metrics["hcu.theoretical_occupancy_pct"] == pytest.approx(
        100 * active.value * threads / metrics["hcu.max_threads_per_cu"])
    assert 0 < metrics["hcu.theoretical_occupancy_pct"] <= 100
