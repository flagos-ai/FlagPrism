from __future__ import annotations

import pytest

from .operator_test_utils import assert_close, ftl, synchronize, take_debug_run, tl, torch, triton

pytestmark = [
    pytest.mark.ascend_debugger_operator, pytest.mark.ascend_debugger_ci
]


@triton.jit
def _mm_kernel(a_ptr, b_ptr, out_ptr, size: tl.constexpr):
    row_offsets = tl.arange(0, size)[:, None]
    col_offsets = tl.arange(0, size)[None, :]
    reduction_offsets = tl.arange(0, size)
    ftl.debug_collect_start(level=1, addr_level=1)
    a = tl.load(a_ptr + row_offsets * size + reduction_offsets[None, :])
    b = tl.load(b_ptr + reduction_offsets[:, None] * size + col_offsets)
    result = tl.dot(a, b, allow_tf32=False)
    tl.store(out_ptr + row_offsets * size + col_offsets, result)
    ftl.debug_collect_end()


@triton.jit
def _bmm_kernel(a_ptr, b_ptr, out_ptr, size: tl.constexpr):
    batch = tl.program_id(0)
    batch_stride = size * size
    row_offsets = tl.arange(0, size)[:, None]
    col_offsets = tl.arange(0, size)[None, :]
    reduction_offsets = tl.arange(0, size)
    ftl.debug_collect_start(level=1, addr_level=1)
    a = tl.load(a_ptr + batch * batch_stride + row_offsets * size +
                reduction_offsets[None, :])
    b = tl.load(b_ptr + batch * batch_stride +
                reduction_offsets[:, None] * size + col_offsets)
    result = tl.dot(a, b, allow_tf32=False)
    tl.store(out_ptr + batch * batch_stride + row_offsets * size + col_offsets,
             result)
    ftl.debug_collect_end()


@triton.jit
def _addr_kernel(x_ptr, y_ptr, base_ptr, out_ptr, size: tl.constexpr):
    rows = tl.arange(0, size)[:, None]
    cols = tl.arange(0, size)[None, :]
    ftl.debug_collect_start(level=1, addr_level=1)
    x = tl.load(x_ptr + rows)
    y = tl.load(y_ptr + cols)
    base = tl.load(base_ptr + rows * size + cols)
    result = base + x * y
    tl.store(out_ptr + rows * size + cols, result)
    ftl.debug_collect_end()


@triton.jit
def _gather_kernel(x_ptr, index_ptr, out_ptr, n: tl.constexpr,
                   block: tl.constexpr):
    offsets = tl.arange(0, block)
    mask = offsets < n
    ftl.debug_collect_start(level=1, addr_level=1)
    indices = tl.load(index_ptr + offsets, mask=mask, other=0)
    result = tl.load(x_ptr + indices, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, result, mask=mask)
    ftl.debug_collect_end()


@triton.jit
def _masked_fill_kernel(x_ptr, mask_ptr, out_ptr, n: tl.constexpr,
                        block: tl.constexpr):
    offsets = tl.arange(0, block)
    active = offsets < n
    ftl.debug_collect_start(level=1, addr_level=1)
    x = tl.load(x_ptr + offsets, mask=active, other=0.0)
    fill_mask = tl.load(mask_ptr + offsets, mask=active, other=0)
    result = tl.where(fill_mask != 0, -3.0, x)
    tl.store(out_ptr + offsets, result, mask=active)
    ftl.debug_collect_end()


@triton.jit
def _index_add_kernel(index_ptr, source_ptr, out_ptr, n: tl.constexpr,
                      block: tl.constexpr):
    offsets = tl.arange(0, block)
    mask = offsets < n
    ftl.debug_collect_start(level=1, addr_level=1)
    indices = tl.load(index_ptr + offsets, mask=mask, other=0)
    source = tl.load(source_ptr + offsets, mask=mask, other=0.0)
    tl.atomic_add(out_ptr + indices, source, mask=mask)
    ftl.debug_collect_end()


def test_mm_operator(debug_session):
    a_cpu = torch.arange(256, dtype=torch.float32).reshape(16, 16) / 128.0
    b_cpu = torch.arange(256, dtype=torch.float32).reshape(
        16, 16).t().contiguous() / 256.0
    a, b = a_cpu.npu(), b_cpu.npu()
    out = torch.empty((16, 16), dtype=torch.float32, device="npu")
    with debug_session():
        _mm_kernel[(1, )](a, b, out, 16)
        synchronize()
        assert_close(out, a_cpu @ b_cpu, rtol=2e-3, atol=2e-3)
        take_debug_run()


def test_bmm_operator(debug_session):
    a_cpu = torch.arange(512, dtype=torch.float32).reshape(2, 16, 16) / 256.0
    b_cpu = torch.flip(a_cpu, dims=(2, )).contiguous()
    a, b = a_cpu.npu(), b_cpu.npu()
    out = torch.empty((2, 16, 16), dtype=torch.float32, device="npu")
    with debug_session():
        _bmm_kernel[(2, )](a, b, out, 16)
        synchronize()
        assert_close(out, torch.bmm(a_cpu, b_cpu), rtol=2e-3, atol=2e-3)
        take_debug_run()


def test_addr_operator(debug_session):
    x_cpu = torch.linspace(-1.0, 1.0, 8, dtype=torch.float32)
    y_cpu = torch.linspace(0.5, 1.5, 8, dtype=torch.float32)
    base_cpu = torch.arange(64, dtype=torch.float32).reshape(8, 8) / 32.0
    x, y, base = x_cpu.npu(), y_cpu.npu(), base_cpu.npu()
    out = torch.empty_like(base)
    with debug_session():
        _addr_kernel[(1, )](x, y, base, out, 8)
        synchronize()
        assert_close(out, torch.addr(base_cpu, x_cpu, y_cpu))
        take_debug_run()


def test_gather_operator(debug_session):
    x_cpu = torch.linspace(-4.0, 4.0, 16, dtype=torch.float32)
    index_cpu = torch.tensor([15, 0, 7, 3, 12, 1, 9, 5], dtype=torch.int32)
    x, index = x_cpu.npu(), index_cpu.npu()
    out = torch.empty(8, dtype=torch.float32, device="npu")
    with debug_session():
        _gather_kernel[(1, )](x, index, out, 8, block=8)
        synchronize()
        assert_close(out, x_cpu[index_cpu.to(torch.int64)])
        take_debug_run()


def test_masked_fill_operator(debug_session):
    x_cpu = torch.linspace(-2.0, 2.0, 16, dtype=torch.float32)
    mask_cpu = (torch.arange(16) % 3 == 0).to(torch.int32)
    x, mask = x_cpu.npu(), mask_cpu.npu()
    out = torch.empty_like(x)
    with debug_session():
        _masked_fill_kernel[(1, )](x, mask, out, 16, block=16)
        synchronize()
        assert_close(out, x_cpu.masked_fill(mask_cpu.bool(), -3.0))
        take_debug_run()


def test_index_add_operator(debug_session):
    index_cpu = torch.tensor([0, 1, 1, 3, 4, 4, 6, 7], dtype=torch.int32)
    source_cpu = torch.linspace(1.0, 8.0, 8, dtype=torch.float32)
    index, source = index_cpu.npu(), source_cpu.npu()
    out = torch.zeros(8, dtype=torch.float32, device="npu")
    expected = torch.zeros(8, dtype=torch.float32).index_add_(
        0, index_cpu.to(torch.int64), source_cpu)
    with debug_session():
        _index_add_kernel[(1, )](index, source, out, 8, block=8)
        synchronize()
        assert_close(out, expected)
        take_debug_run()
