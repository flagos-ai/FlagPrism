from __future__ import annotations

import pytest

from .operator_test_utils import assert_close, ftl, synchronize, take_debug_run, tl, torch, triton

pytestmark = [
    pytest.mark.ascend_debugger_operator, pytest.mark.ascend_debugger_ci
]


@triton.jit
def _layer_norm_kernel(x_ptr, weight_ptr, bias_ptr, out_ptr,
                       cols: tl.constexpr):
    row = tl.program_id(0)
    offsets = row * cols + tl.arange(0, cols)
    feature_offsets = tl.arange(0, cols)
    ftl.debug_collect_start(level=1, addr_level=1)
    x = tl.load(x_ptr + offsets)
    weight = tl.load(weight_ptr + feature_offsets)
    bias = tl.load(bias_ptr + feature_offsets)
    mean = tl.sum(x, axis=0) / cols
    centered = x - mean
    variance = tl.sum(centered * centered, axis=0) / cols
    result = centered * tl.rsqrt(variance + 1.0e-5) * weight + bias
    tl.store(out_ptr + offsets, result)
    ftl.debug_collect_end()


@triton.jit
def _adaptive_avg_pool2d_kernel(x_ptr, out_ptr):
    output_offsets = tl.arange(0, 4)[:, None]
    window_offsets = tl.arange(0, 4)[None, :]
    output_row = output_offsets // 2
    output_col = output_offsets % 2
    window_row = window_offsets // 2
    window_col = window_offsets % 2
    source = output_row * 8 + output_col * 2 + window_row * 4 + window_col
    ftl.debug_collect_start(level=1, addr_level=1)
    window = tl.load(x_ptr + source)
    result = tl.sum(window, axis=1) * 0.25
    tl.store(out_ptr + tl.arange(0, 4), result)
    ftl.debug_collect_end()


@triton.jit
def _reflection_pad2d_kernel(x_ptr, out_ptr, block: tl.constexpr):
    offsets = tl.arange(0, block)
    mask = offsets < 36
    output_row = offsets // 6
    output_col = offsets % 6
    unpadded_row = output_row - 1
    unpadded_col = output_col - 1
    source_row = tl.where(unpadded_row < 0, -unpadded_row, unpadded_row)
    source_row = tl.where(source_row >= 4, 6 - source_row, source_row)
    source_col = tl.where(unpadded_col < 0, -unpadded_col, unpadded_col)
    source_col = tl.where(source_col >= 4, 6 - source_col, source_col)
    ftl.debug_collect_start(level=1, addr_level=1)
    result = tl.load(x_ptr + source_row * 4 + source_col, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, result, mask=mask)
    ftl.debug_collect_end()


@triton.jit
def _mse_loss_kernel(x_ptr, y_ptr, out_ptr, size: tl.constexpr):
    offsets = tl.arange(0, size)
    ftl.debug_collect_start(level=1, addr_level=1)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    difference = x - y
    result = tl.sum(difference * difference, axis=0) / size
    tl.store(out_ptr, result)
    ftl.debug_collect_end()


@triton.jit
def _silu_and_mul_kernel(x_ptr, y_ptr, out_ptr, size: tl.constexpr):
    offsets = tl.arange(0, size)
    ftl.debug_collect_start(level=1, addr_level=1)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    result = x * tl.sigmoid(x) * y
    tl.store(out_ptr + offsets, result)
    ftl.debug_collect_end()


def test_layer_norm_operator(debug_session):
    x_cpu = torch.linspace(-3.0, 3.0, 32, dtype=torch.float32).reshape(2, 16)
    weight_cpu = torch.linspace(0.5, 1.5, 16, dtype=torch.float32)
    bias_cpu = torch.linspace(-0.2, 0.2, 16, dtype=torch.float32)
    x, weight, bias = x_cpu.npu(), weight_cpu.npu(), bias_cpu.npu()
    out = torch.empty_like(x)
    with debug_session():
        _layer_norm_kernel[(2, )](x, weight, bias, out, 16)
        synchronize()
        expected = torch.nn.functional.layer_norm(x_cpu, (16, ), weight_cpu,
                                                  bias_cpu)
        assert_close(out, expected, rtol=3e-4, atol=3e-4)
        take_debug_run()


def test_adaptive_avg_pool2d_operator(debug_session):
    x_cpu = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    x = x_cpu.npu()
    out = torch.empty((1, 1, 2, 2), dtype=torch.float32, device="npu")
    with debug_session():
        _adaptive_avg_pool2d_kernel[(1, )](x, out)
        synchronize()
        assert_close(out,
                     torch.nn.functional.adaptive_avg_pool2d(x_cpu, (2, 2)))
        take_debug_run()


def test_reflection_pad2d_operator(debug_session):
    x_cpu = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    x = x_cpu.npu()
    out = torch.empty((1, 1, 6, 6), dtype=torch.float32, device="npu")
    with debug_session():
        _reflection_pad2d_kernel[(1, )](x, out, block=64)
        synchronize()
        assert_close(
            out, torch.nn.functional.pad(x_cpu, (1, 1, 1, 1), mode="reflect"))
        take_debug_run()


def test_mse_loss_operator(debug_session):
    x_cpu = torch.linspace(-2.0, 2.0, 16, dtype=torch.float32)
    y_cpu = torch.linspace(1.0, -1.0, 16, dtype=torch.float32)
    x, y = x_cpu.npu(), y_cpu.npu()
    out = torch.empty((), dtype=torch.float32, device="npu")
    with debug_session():
        _mse_loss_kernel[(1, )](x, y, out, 16)
        synchronize()
        assert_close(out, torch.nn.functional.mse_loss(x_cpu, y_cpu))
        take_debug_run()


def test_silu_and_mul_operator(debug_session):
    x_cpu = torch.linspace(-4.0, 4.0, 16, dtype=torch.float32)
    y_cpu = torch.linspace(0.5, 1.5, 16, dtype=torch.float32)
    x, y = x_cpu.npu(), y_cpu.npu()
    out = torch.empty_like(x)
    with debug_session():
        _silu_and_mul_kernel[(1, )](x, y, out, 16)
        synchronize()
        assert_close(out,
                     torch.nn.functional.silu(x_cpu) * y_cpu,
                     rtol=2e-4,
                     atol=2e-4)
        take_debug_run()
