from __future__ import annotations

import pytest

from .operator_test_utils import assert_close, ftl, synchronize, take_debug_run, tl, torch, triton

pytestmark = [
    pytest.mark.ascend_debugger_operator, pytest.mark.ascend_debugger_ci
]


@triton.jit
def _creation_kernel(out_ptr, n: tl.constexpr, MODE: tl.constexpr,
                     BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    ftl.debug_collect_start(level=1, addr_level=1)
    if MODE == 0:
        result = offsets.to(tl.float32)
    else:
        result = tl.zeros((BLOCK, ), tl.float32)
    tl.store(out_ptr + offsets, result, mask=mask)
    ftl.debug_collect_end()


@triton.jit
def _transpose_kernel(x_ptr, out_ptr, rows: tl.constexpr, cols: tl.constexpr,
                      BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < rows * cols
    out_row = offsets // rows
    out_col = offsets % rows
    input_offsets = out_col * cols + out_row
    ftl.debug_collect_start(level=1, addr_level=1)
    result = tl.load(x_ptr + input_offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, result, mask=mask)
    ftl.debug_collect_end()


@triton.jit
def _stack_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < 2 * n
    source_offsets = offsets % n
    ftl.debug_collect_start(level=1, addr_level=1)
    x = tl.load(x_ptr + source_offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + source_offsets, mask=mask, other=0.0)
    result = tl.where(offsets < n, x, y)
    tl.store(out_ptr + offsets, result, mask=mask)
    ftl.debug_collect_end()


@triton.jit
def _row_reduction_kernel(x_ptr, out_ptr, cols: tl.constexpr,
                          MODE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = row * cols + tl.arange(0, BLOCK)
    mask = tl.arange(0, BLOCK) < cols
    ftl.debug_collect_start(level=1, addr_level=1)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    if MODE == 0:
        result = tl.sum(x, axis=0)
        tl.store(out_ptr + row, result)
    elif MODE == 1:
        result = tl.max(tl.where(mask, x, -float("inf")), axis=0)
        tl.store(out_ptr + row, result)
    elif MODE == 2:
        result = tl.cumsum(x, axis=0)
        tl.store(out_ptr + offsets, result, mask=mask)
    else:
        shifted = x - tl.max(tl.where(mask, x, -float("inf")), axis=0)
        numerator = tl.exp(shifted)
        result = numerator / tl.sum(numerator, axis=0)
        tl.store(out_ptr + offsets, result, mask=mask)
    ftl.debug_collect_end()


@pytest.mark.parametrize(
    "mode,op_name,expected",
    [
        pytest.param(
            0, "arange", torch.arange(16, dtype=torch.float32), id="arange"),
        pytest.param(
            1, "zeros", torch.zeros(16, dtype=torch.float32), id="zeros"),
    ],
)
def test_creation_operator(debug_session, mode, op_name, expected):
    del op_name
    out = torch.empty(16, dtype=torch.float32, device="npu")
    with debug_session():
        _creation_kernel[(1, )](out, 16, MODE=mode, BLOCK=16)
        synchronize()
        assert_close(out, expected)
        take_debug_run()


def test_t_copy_operator(debug_session):
    x_cpu = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    x = x_cpu.npu()
    out = torch.empty_like(x)
    with debug_session():
        _transpose_kernel[(1, )](x, out, 4, 4, BLOCK=16)
        synchronize()
        assert_close(out, x_cpu.t().contiguous())
        take_debug_run()


def test_stack_operator(debug_session):
    x_cpu = torch.arange(8, dtype=torch.float32)
    y_cpu = x_cpu + 10
    x, y = x_cpu.npu(), y_cpu.npu()
    out = torch.empty((2, 8), dtype=torch.float32, device="npu")
    with debug_session():
        _stack_kernel[(1, )](x, y, out, 8, BLOCK=16)
        synchronize()
        assert_close(out, torch.stack((x_cpu, y_cpu)))
        take_debug_run()


REDUCTION_CASES = [
    pytest.param(0, "sum", lambda x: torch.sum(x, dim=1), id="sum"),
    pytest.param(1,
                 "max_dim",
                 lambda x: torch.max(x, dim=1).values,
                 id="max_dim"),
    pytest.param(2,
                 "cumsum_out",
                 lambda x: torch.cumsum(x, dim=1),
                 id="cumsum_out"),
    pytest.param(3, "softmax", lambda x: torch.softmax(x, dim=1),
                 id="softmax"),
]


@pytest.mark.parametrize("mode,op_name,reference", REDUCTION_CASES)
def test_row_reduction_operator(debug_session, mode, op_name, reference):
    del op_name
    x_cpu = torch.linspace(-2.0, 2.0, 32, dtype=torch.float32).reshape(4, 8)
    x = x_cpu.npu()
    out_shape = (4, ) if mode < 2 else (4, 8)
    out = torch.empty(out_shape, dtype=torch.float32, device="npu")
    with debug_session():
        _row_reduction_kernel[(4, )](x, out, 8, MODE=mode, BLOCK=8)
        synchronize()
        assert_close(out, reference(x_cpu), rtol=2e-4, atol=2e-4)
        take_debug_run()
