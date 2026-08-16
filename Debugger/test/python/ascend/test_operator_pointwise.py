from __future__ import annotations

import pytest

from .operator_test_utils import assert_close, ftl, synchronize, take_debug_run, tl, torch, triton

pytestmark = [
    pytest.mark.ascend_debugger_operator, pytest.mark.ascend_debugger_ci
]


@triton.jit
def _float_pointwise_kernel(x_ptr, y_ptr, z_ptr, out_ptr, n: tl.constexpr,
                            MODE: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    ftl.debug_collect_start(level=1, addr_level=1)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    z = tl.load(z_ptr + offsets, mask=mask, other=0.0)
    if MODE == 0:
        result = tl.abs(x)
    elif MODE == 1:
        result = tl.exp(x)
    elif MODE == 2:
        result = tl.sin(x)
    elif MODE == 3:
        result = x + y
    elif MODE == 4:
        result = x + 0.5 * y * z
    elif MODE == 5:
        result = tl.where(x > 0.0, y, z)
    elif MODE == 6:
        result = tl.maximum(x, 0.0)
    else:
        inner = 0.7978845608 * (x + 0.044715 * x * x * x)
        result = x * tl.sigmoid(2.0 * inner)
    tl.store(out_ptr + offsets, result, mask=mask)
    ftl.debug_collect_end()


@triton.jit
def _integer_pointwise_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr,
                              MODE: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    ftl.debug_collect_start(level=1, addr_level=1)
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0)
    if MODE == 0:
        result = tl.where(x == y, 1, 0)
    elif MODE == 1:
        result = tl.where((x != 0) & (y != 0), 1, 0)
    else:
        result = x & y
    tl.store(out_ptr + offsets, result, mask=mask)
    ftl.debug_collect_end()


def _gelu_tanh_reference(x):
    inner = 0.7978845608 * (x + 0.044715 * x * x * x)
    return x * torch.sigmoid(2.0 * inner)


FLOAT_CASES = [
    pytest.param(0, "abs", lambda x, y, z: torch.abs(x), id="abs"),
    pytest.param(1, "exp", lambda x, y, z: torch.exp(x), id="exp"),
    pytest.param(2, "sin", lambda x, y, z: torch.sin(x), id="sin"),
    pytest.param(3, "add", lambda x, y, z: x + y, id="add"),
    pytest.param(4, "addcmul", lambda x, y, z: x + 0.5 * y * z, id="addcmul"),
    pytest.param(5,
                 "where_self",
                 lambda x, y, z: torch.where(x > 0, y, z),
                 id="where_self"),
    pytest.param(6, "relu", lambda x, y, z: torch.relu(x), id="relu"),
    pytest.param(7, "gelu", lambda x, y, z: _gelu_tanh_reference(x),
                 id="gelu"),
]


@pytest.mark.parametrize("mode,op_name,reference", FLOAT_CASES)
def test_float_pointwise_operator(debug_session, mode, op_name, reference):
    del op_name
    x_cpu = torch.linspace(-2.0, 2.0, 16, dtype=torch.float32)
    y_cpu = torch.linspace(1.0, 3.0, 16, dtype=torch.float32)
    z_cpu = torch.linspace(-1.0, 1.0, 16, dtype=torch.float32)
    x, y, z = x_cpu.npu(), y_cpu.npu(), z_cpu.npu()
    out = torch.empty_like(x)
    with debug_session():
        _float_pointwise_kernel[(1, )](x, y, z, out, 16, MODE=mode, BLOCK=16)
        synchronize()
        assert_close(out, reference(x_cpu, y_cpu, z_cpu), rtol=2e-4, atol=2e-4)
        take_debug_run()


INTEGER_CASES = [
    pytest.param(0, "eq", lambda x, y: (x == y).to(torch.int32), id="eq"),
    pytest.param(1,
                 "logical_and",
                 lambda x, y: ((x != 0) & (y != 0)).to(torch.int32),
                 id="logical_and"),
    pytest.param(2,
                 "bitwise_and_tensor",
                 lambda x, y: x & y,
                 id="bitwise_and_tensor"),
]


@pytest.mark.parametrize("mode,op_name,reference", INTEGER_CASES)
def test_integer_pointwise_operator(debug_session, mode, op_name, reference):
    del op_name
    x_cpu = torch.arange(16, dtype=torch.int32) - 4
    y_cpu = torch.arange(16, dtype=torch.int32) % 5
    x, y = x_cpu.npu(), y_cpu.npu()
    out = torch.empty_like(x)
    with debug_session():
        _integer_pointwise_kernel[(1, )](x, y, out, 16, MODE=mode, BLOCK=16)
        synchronize()
        assert_close(out, reference(x_cpu, y_cpu), rtol=0, atol=0)
        take_debug_run()
