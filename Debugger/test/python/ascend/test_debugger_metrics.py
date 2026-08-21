from __future__ import annotations

from pathlib import Path

import pytest

from .operator_test_utils import (
    address_summary,
    assert_close,
    assert_summary,
    expected_summary,
    ftl,
    summary_for_result,
    synchronize,
    take_debug_run,
    tl,
    torch,
    triton,
)

numpy = pytest.importorskip("numpy")

pytestmark = [
    pytest.mark.ascend_debugger_metrics, pytest.mark.ascend_debugger_ci
]


@triton.jit
def _value_metric_kernel(x_ptr, out_ptr, mode: tl.constexpr,
                         block: tl.constexpr):
    offsets = tl.arange(0, block)
    ftl.debug_collect_start(level=1, addr_level=0)
    if mode == 0:
        observed = tl.load(x_ptr + offsets)
    else:
        x = tl.load(x_ptr + offsets)
        if mode == 1:
            observed = tl.sqrt(x)
        else:
            observed = tl.exp(x)
    tl.store(out_ptr + offsets, observed)
    ftl.debug_collect_end()


@triton.jit
def _multi_instance_metric_kernel(x_ptr, out_ptr, block: tl.constexpr):
    program = tl.program_id(0)
    offsets = program * block + tl.arange(0, block)
    ftl.debug_collect_start(level=1, addr_level=0)
    observed = tl.load(x_ptr + offsets)
    tl.store(out_ptr + offsets, observed)
    ftl.debug_collect_end()


@triton.jit
def _address_metric_kernel(x_ptr, out_ptr, n: tl.constexpr,
                           block: tl.constexpr):
    offsets = tl.arange(0, block)
    mask = offsets < n
    ftl.debug_collect_start(level=1, addr_level=1)
    observed = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, observed, mask=mask)
    ftl.debug_collect_end()


@triton.jit
def _level2_metric_kernel(x_ptr, out_ptr, block: tl.constexpr):
    offsets = tl.arange(0, block)
    ftl.debug_collect_start(level=2, addr_level=2)
    observed = tl.load(x_ptr + offsets)
    tl.store(out_ptr + offsets, observed)
    ftl.debug_collect_end()


@triton.jit
def _overflow_metric_kernel(x_ptr, out_ptr, block: tl.constexpr):
    program = tl.program_id(0)
    offsets = program * block + tl.arange(0, block)
    ftl.debug_collect_start(level=1, addr_level=0)
    observed = tl.load(x_ptr + offsets)
    result = observed + 1.0
    tl.store(out_ptr + offsets, result)
    ftl.debug_collect_end()


VALUE_CASES = [
    pytest.param(
        0,
        torch.tensor(
            [
                1.0,
                -2.0,
                float("nan"),
                float("inf"),
                -float("inf"),
                0.0,
                -0.0,
                3.5,
            ],
            dtype=torch.float32,
        ),
        id="mixed-special-values",
    ),
    pytest.param(
        1,
        torch.tensor([-4.0, -1.0, 0.0, 1.0, 4.0, 9.0, 16.0, 25.0],
                     dtype=torch.float32),
        id="sqrt-produces-nan",
    ),
    pytest.param(
        2,
        torch.tensor([0.0, 1.0, 5.0, 10.0, 15.0, 18.0, 90.0, 100.0],
                     dtype=torch.float32),
        id="exp-produces-inf",
    ),
    pytest.param(
        0,
        torch.tensor(
            [
                float("nan"),
                float("inf"),
                -float("inf"),
                float("nan"),
                float("inf"),
                -float("inf"),
                float("nan"),
                float("inf"),
            ],
            dtype=torch.float32,
        ),
        id="all-nonfinite",
    ),
    pytest.param(
        0,
        torch.tensor([0.0, -0.0, 0.0, -0.0, 1.0, -1.0, 2.0, -2.0],
                     dtype=torch.float32),
        id="signed-zero",
    ),
]


@pytest.mark.parametrize("mode,input_values", VALUE_CASES)
def test_value_summary_metrics(debug_session, mode, input_values):
    x = input_values.npu()
    out = torch.empty_like(x)
    with debug_session(level=1, addr_level=0):
        _value_metric_kernel[(1, )](x, out, mode=mode, block=8)
        synchronize()
        if mode == 1:
            expected = torch.sqrt(input_values)
        elif mode == 2:
            expected = torch.exp(input_values)
        else:
            expected = input_values
        assert_close(out, expected)
        zero_mask = input_values == 0
        if mode == 0 and torch.any(zero_mask):
            assert torch.equal(
                torch.signbit(out.cpu()[zero_mask]),
                torch.signbit(input_values[zero_mask]),
            )
        run = take_debug_run()
        assert_summary(summary_for_result(run, "observed"),
                       expected_summary(expected))


def test_summary_metrics_are_separated_by_program_instance(debug_session):
    input_values = torch.tensor(
        [
            0.0,
            1.0,
            2.0,
            3.0,
            float("nan"),
            float("inf"),
            -1.0,
            0.0,
        ],
        dtype=torch.float32,
    )
    x = input_values.npu()
    out = torch.empty_like(x)
    with debug_session(level=1, addr_level=0):
        _multi_instance_metric_kernel[(2, )](x, out, block=4)
        synchronize()
        assert_close(out, input_values)
        run = take_debug_run()
        for instance in range(2):
            start = instance * 4
            expected = expected_summary(input_values[start:start + 4])
            assert_summary(
                summary_for_result(run, "observed", instance=instance),
                expected)


@pytest.mark.parametrize("active_lanes", [16, 5],
                         ids=["contiguous", "prefix-mask"])
def test_address_range_metrics(debug_session, active_lanes):
    input_values = torch.arange(16, dtype=torch.float32)
    x = input_values.npu()
    out = torch.full((16, ), -1.0, dtype=torch.float32, device="npu")
    with debug_session(level=1, addr_level=1):
        _address_metric_kernel[(1, )](x, out, active_lanes, block=16)
        synchronize()
        expected = torch.full((16, ), -1.0, dtype=torch.float32)
        expected[:active_lanes] = input_values[:active_lanes]
        assert_close(out, expected)
        run = take_debug_run()

        load_metrics = address_summary(run, "load")
        first = x.data_ptr()
        last = first + (active_lanes - 1) * x.element_size()
        assert load_metrics["first_addr"] == first
        assert load_metrics["last_addr"] == last
        assert load_metrics["min_addr"] == first
        assert load_metrics["max_addr"] == last
        assert load_metrics["active_lane_count"] == active_lanes
        assert load_metrics[
            "address_span_bytes"] == active_lanes * x.element_size()

        if active_lanes < 16:
            summary = summary_for_result(run, "observed")
            assert summary["element_count"] == 16
            assert summary["element_count"] != load_metrics[
                "active_lane_count"]


def test_level2_exports_value_and_address_artifacts(debug_session):
    input_values = torch.arange(8, dtype=torch.float32)
    x = input_values.npu()
    out = torch.empty_like(x)
    with debug_session(level=2, addr_level=2):
        _level2_metric_kernel[(1, )](x, out, block=8)
        synchronize()
        assert_close(out, input_values)
        run = take_debug_run()

        artifacts = run["runtime_metadata"].get("full_dump_artifacts") or []
        assert {artifact["kind"]
                for artifact in artifacts} >= {"value", "memory_address"}
        for artifact in artifacts:
            assert Path(artifact["path"]).is_file()

        value_arrays = [
            numpy.load(artifact["path"]) for artifact in artifacts
            if artifact["kind"] == "value"
        ]
        address_arrays = [
            numpy.load(artifact["path"]) for artifact in artifacts
            if artifact["kind"] == "memory_address"
        ]
        expected_values = input_values.numpy()
        expected_addresses = numpy.arange(
            8, dtype=numpy.uint64) * x.element_size() + x.data_ptr()
        assert any(
            numpy.array_equal(array.reshape(-1), expected_values)
            for array in value_arrays)
        assert any(
            numpy.array_equal(array.reshape(-1), expected_addresses)
            for array in address_arrays)


def test_ring_buffer_overflow_is_reported(debug_session):
    input_values = torch.arange(128, dtype=torch.float32)
    x = input_values.npu()
    out = torch.empty_like(x)
    with debug_session(level=1, addr_level=0, record_capacity=64):
        _overflow_metric_kernel[(32, )](x, out, block=4)
        synchronize()
        assert_close(out, input_values + 1.0)
        take_debug_run(overflow=True)
