"""NVIDIA debugger launch/collect smoke test."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
pytest.importorskip("triton.backends.nvidia.driver")

import triton.language as tl
import flagtree.language as ftl
import flagtree.debugger as debugger


@pytest.mark.module_a
def test_nvidia_debugger_collects_a_real_kernel_launch():
    """FlagPrism: verify the CUDA hidden-argument and transfer path together."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available")

    @triton.jit
    def copy_kernel(x_ptr, y_ptr, n: tl.constexpr):
        offsets = tl.arange(0, n)
        ftl.debug_collect_start(level=1)
        values = tl.load(x_ptr + offsets)
        tl.store(y_ptr + offsets, values)
        ftl.debug_collect_end()

    debugger.clear_exported_runs()
    debugger.reset_config()
    debugger.activate(level=1, output_dir=None, timeline=True)
    try:
        x = torch.arange(16, device="cuda", dtype=torch.float32)
        y = torch.zeros_like(x)
        copy_kernel[(1, )](x, y, 16)
        torch.cuda.synchronize()
        runs = debugger.take_exported_runs()
    finally:
        debugger.deactivate()

    assert torch.equal(x, y)
    assert runs
    # FlagPrism: the CUDA transfer adapter must return a decoded run for an
    # explicitly marked load/store region.
    assert runs[0]["meta"]["backend_kind"] == 1
    assert runs[0]["decoded"]["records"]
    # FlagPrism: CUDA timeline records use PTX %globaltimer when explicitly
    # requested through the backend-neutral debugger API.
    assert any(
        record.get("record_kind") == "TIMELINE"
        for record in runs[0]["decoded"]["records"])


@pytest.mark.module_a
def test_nvidia_debugger_level2_exports_supported_artifacts(tmp_path):
    """FlagPrism: exercise CUDA value/address artifact export when available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available")

    @triton.jit
    def copy_kernel(x_ptr, y_ptr, n: tl.constexpr):
        offsets = tl.arange(0, n)
        ftl.debug_collect_start(level=2, addr_level=2)
        values = tl.load(x_ptr + offsets)
        tl.store(y_ptr + offsets, values)
        ftl.debug_collect_end()

    debugger.clear_exported_runs()
    debugger.reset_config()
    debugger.activate(level=2, addr_level=2, output_dir=tmp_path)
    try:
        x = torch.arange(16, device="cuda", dtype=torch.float32)
        y = torch.zeros_like(x)
        copy_kernel[(1, )](x, y, 16)
        torch.cuda.synchronize()
        runs = debugger.take_exported_runs()
    finally:
        debugger.deactivate()

    assert torch.equal(x, y)
    assert runs
    artifacts = runs[0]["runtime_metadata"].get("full_dump_artifacts") or []
    assert {artifact["kind"]
            for artifact in artifacts} >= {"value", "memory_address"}
    assert all(Path(artifact["path"]).is_file() for artifact in artifacts)


@pytest.mark.module_a
def test_nvidia_debugger_masked_multi_stream_level1():
    """FlagPrism: validate exact-stream export for masked tail accesses."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available")

    @triton.jit
    def masked_copy_kernel(x_ptr, y_ptr, n: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * 32 + tl.arange(0, 32)
        mask = offsets < n
        ftl.debug_collect_start(level=1, addr_level=1)
        values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        tl.store(y_ptr + offsets, values, mask=mask)
        ftl.debug_collect_end()

    n = 37
    x = torch.arange(n, device="cuda", dtype=torch.float32)
    y_a = torch.zeros_like(x)
    y_b = torch.zeros_like(x)
    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()
    debugger.clear_exported_runs()
    debugger.reset_config()
    debugger.activate(level=1, addr_level=1, output_dir=None, timeline=False)
    try:
        with torch.cuda.stream(stream_a):
            masked_copy_kernel[(2, )](x, y_a, n, num_warps=1)
        with torch.cuda.stream(stream_b):
            masked_copy_kernel[(2, )](x, y_b, n, num_warps=1)
        stream_a.synchronize()
        stream_b.synchronize()
        runs = debugger.take_exported_runs()
    finally:
        debugger.deactivate()

    assert torch.equal(x, y_a)
    assert torch.equal(x, y_b)
    assert len(runs) >= 2
    assert all(run["decoded"]["records"] for run in runs)
    assert {run["meta"]["backend_kind"] for run in runs} == {1}
