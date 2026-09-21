"""NVIDIA CUPTI vendor metric integration tests."""

import json
import math
import pathlib

import pytest
import torch
import triton
import triton.language as tl

import flagtree.profiler as profiler


def _is_cuda_target() -> bool:
    # FlagPrism: an unavailable or temporarily inaccessible CUDA driver must
    # skip the device test at collection time instead of turning an optional
    # NVIDIA test into a suite-wide collection failure.
    try:
        return (triton.runtime.driver.active.get_current_target().backend
                == "cuda" and torch.cuda.is_available())
    except RuntimeError:
        return False


def _find_metric_node(nodes, predicate):
    for node in nodes:
        if predicate(node.get("metrics", {})):
            return node
        child = _find_metric_node(node.get("children", []), predicate)
        if child is not None:
            return child
    return None


@pytest.mark.skipif(not _is_cuda_target(),
                    reason="requires an NVIDIA CUDA target")
def test_nvidia_launch_stats_activity_fields(tmp_path: pathlib.Path):

    @triton.jit
    def copy_kernel(x_ptr, y_ptr, n: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * 128 + tl.arange(0, 128)
        mask = offsets < n
        values = tl.load(x_ptr + offsets, mask=mask)
        tl.store(y_ptr + offsets, values, mask=mask)

    x = torch.arange(256, device="cuda", dtype=torch.float32)
    y = torch.zeros_like(x)
    base = tmp_path / "nvidia_launch_stats"
    profiler.start(
        str(base),
        backend="nvidia",
        mode="runtime_base:vendor_metrics=launch_stats",
    )
    with profiler.scope("nvidia_launch_stats_scope"):
        copy_kernel[(2, )](x, y, 256, num_warps=4)
    torch.cuda.synchronize()
    profiler.finalize()

    assert torch.equal(x, y)
    vendor = json.loads(base.with_suffix(".vendor.json").read_text())
    assert vendor["backend"] == "nvidia"
    assert "launch_stats" in vendor["enabled_metrics"]
    association = next(
        item for item in vendor["associations"]
        if item["state"] == "collected" and "launch_stats" in item["metrics"])
    metrics = association["metrics"]
    assert metrics["launch_stats"] == 1
    assert metrics["grid_x"] == 2
    assert metrics["grid_blocks"] == 2
    assert metrics["block_threads"] > 0
    assert "registers_per_thread" in metrics
    assert "static_shared_memory_bytes" in metrics
    assert "dynamic_shared_memory_bytes" in metrics

    hatchet = json.loads(base.with_suffix(".hatchet").read_text())
    node = _find_metric_node(
        hatchet,
        lambda metrics: metrics.get("nvidia.launch_stats") == 1 and metrics.
        get("nvidia.grid_x") == 2,
    )
    assert node is not None
    assert node["metrics"]["nvidia.grid_x"] == 2


@pytest.mark.skipif(not _is_cuda_target(),
                    reason="requires an NVIDIA CUDA target")
def test_nvidia_memory_activity(tmp_path: pathlib.Path):
    x = torch.arange(1024, device="cuda", dtype=torch.float32)
    y = torch.zeros_like(x)
    base = tmp_path / "nvidia_memory_activity"
    profiler.start(
        str(base),
        backend="nvidia",
        mode="runtime_base:vendor_metrics=memory",
    )
    with profiler.scope("nvidia_memory_scope"):
        y.copy_(x)
    torch.cuda.synchronize()
    profiler.finalize()

    assert torch.equal(x, y)
    vendor = json.loads(base.with_suffix(".vendor.json").read_text())
    assert "memory" in vendor["enabled_metrics"]
    associations = [
        item for item in vendor["associations"] if item["state"] == "collected"
        and item["metrics"].get("memory_bytes", 0) > 0
    ]
    assert associations
    metrics = associations[0]["metrics"]
    assert metrics["memory_duration_us"] > 0
    assert metrics["memory_bandwidth_gb_s"] > 0


@pytest.mark.skipif(not _is_cuda_target(),
                    reason="requires an NVIDIA CUDA target")
def test_nvidia_instruction_mode_enables_pc_sampling(tmp_path: pathlib.Path):
    """FlagPrism: vendor instruction mode selects the shared CUPTI sampler."""
    base = tmp_path / "nvidia_instruction_mode"
    # FlagPrism: this wiring test intentionally launches no kernel, so it does
    # not depend on the host's performance-counter permission policy. A real
    # kernel sample is covered by the existing CUPTI PC-sampling test when the
    # device exposes that optional capability.
    profiler.start(
        str(base),
        backend="nvidia",
        mode="runtime_base:vendor_metrics=instruction",
    )
    profiler.finalize()

    vendor = json.loads(base.with_suffix(".vendor.json").read_text())
    meta = json.loads(base.with_suffix(".meta.json").read_text())
    assert "instruction" in vendor["enabled_metrics"]
    assert meta["config"]["cupti_pc_sampling"] == "true"


@pytest.mark.skipif(not _is_cuda_target(),
                    reason="requires an NVIDIA CUDA target")
def test_nvidia_occupancy_counter_or_explicit_degrade(tmp_path: pathlib.Path):
    """FlagPrism: exercise the NVPW counter path and its permission fallback."""

    @triton.jit
    def occupancy_kernel(x_ptr, y_ptr, n: tl.constexpr):
        offsets = tl.arange(0, n)
        values = tl.load(x_ptr + offsets)
        tl.store(y_ptr + offsets, values)

    x = torch.arange(256, device="cuda", dtype=torch.float32)
    y = torch.zeros_like(x)
    base = tmp_path / "nvidia_occupancy"
    profiler.start(
        str(base),
        backend="nvidia",
        mode="runtime_base:vendor_metrics=occupancy",
    )
    with profiler.scope("nvidia_occupancy_scope"):
        occupancy_kernel[(2, )](x, y, 256, num_warps=4)
    torch.cuda.synchronize()
    profiler.finalize()

    assert torch.equal(x, y)
    vendor = json.loads(base.with_suffix(".vendor.json").read_text())
    assert "occupancy" in vendor["enabled_metrics"]
    counter_associations = [
        item for item in vendor["associations"]
        if item["state"] == "collected" and "occupancy" in item["metrics"]
    ]
    if counter_associations:
        value = counter_associations[0]["metrics"]["occupancy"]
        assert math.isfinite(value)
        assert 0.0 <= value <= 100.0
        assert counter_associations[0]["metrics"][
            "hardware_counter_source"] == "nvpw"
    else:
        # CUPTI/NVPW may be disabled by the driver performance-counter policy.
        # The vendor session must still finalize and explain the missing data.
        assert any("NVPW" in reason or "counter" in reason
                   for reason in vendor["degrade_reasons"])


@pytest.mark.skipif(not _is_cuda_target(),
                    reason="requires an NVIDIA CUDA target")
def test_nvidia_multi_metric_aliases_and_streams(tmp_path: pathlib.Path):
    """FlagPrism: validate aliases and independent CUDA stream correlation."""

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * 128 + tl.arange(0, 128)
        mask = offsets < n
        values = tl.load(x_ptr + offsets, mask=mask)
        values += tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, values, mask=mask)

    n = 256
    x = torch.arange(n, device="cuda", dtype=torch.float32)
    y = torch.ones_like(x)
    out_a = torch.zeros_like(x)
    out_b = torch.zeros_like(x)
    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()
    base = tmp_path / "nvidia_multi_metric_streams"
    profiler.start(
        str(base),
        backend="nvidia",
        # FlagPrism: use compatibility aliases to exercise plan normalization.
        mode=
        "runtime_base:vendor_metrics=launchstats,kernel_duration_us,memory",
    )
    with torch.cuda.stream(stream_a):
        with profiler.scope("nvidia_stream_a"):
            add_kernel[(2, )](x, y, out_a, n, num_warps=4)
        out_a.copy_(x)
    with torch.cuda.stream(stream_b):
        with profiler.scope("nvidia_stream_b"):
            add_kernel[(2, )](x, y, out_b, n, num_warps=4)
        out_b.copy_(x)
    stream_a.synchronize()
    stream_b.synchronize()
    profiler.finalize()

    assert torch.equal(out_a, x)
    assert torch.equal(out_b, x)
    vendor = json.loads(base.with_suffix(".vendor.json").read_text())
    assert set(("launch_stats", "kernel_duration",
                "memory")) <= set(vendor["enabled_metrics"])
    kernel_associations = [
        item for item in vendor["associations"]
        if item["state"] == "collected" and "launch_stats" in item["metrics"]
        and "kernel_duration_us" in item["metrics"]
    ]
    assert len(kernel_associations) >= 2
    assert len(
        {item["runtime_event"]["stream_id"]
         for item in kernel_associations}) >= 2
    memory_associations = [
        item for item in vendor["associations"] if item["state"] == "collected"
        and item["metrics"].get("memory_bytes", 0) > 0
    ]
    assert memory_associations
