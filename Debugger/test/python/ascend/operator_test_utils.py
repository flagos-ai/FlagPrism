from __future__ import annotations

import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")
triton = pytest.importorskip("triton")

from flagtree import debugger  # noqa: E402
from flagtree import language as ftl  # noqa: E402
from triton import language as tl  # noqa: E402

__all__ = ["debugger", "ftl", "tl"]

COUNT_RECORD = "SUMMARY_COUNT_BUNDLE_U64"
VALUE_RECORD = "SUMMARY_VALUE_BUNDLE_F32"
MEMORY_RECORD = "MEMORY_EVENT"
MEMORY_EVENT_NAMES = {
    1: "last_aligned_addr",
    2: "base_aligned_addr",
    3: "first_addr",
    4: "last_addr",
    5: "min_addr",
    6: "max_addr",
    7: "active_lane_count",
    8: "address_span_bytes",
}


def require_ascend_debugger() -> None:
    try:
        target = triton.runtime.driver.active.get_current_target()
    except RuntimeError as error:
        pytest.skip(f"Ascend runtime is unavailable: {error}")
    backend = str(getattr(target, "backend", "")).lower()
    if backend not in {"ascend", "cann", "npu"}:
        pytest.skip(
            f"Ascend-only FlagPrism test (active backend: {backend or 'unknown'})"
        )
    if not debugger.is_available():
        pytest.fail(
            "FlagPrism debugger support is unavailable in the Ascend build")


def synchronize() -> None:
    torch_npu.npu.synchronize()


def take_debug_run(*, expected_runs: int = 1, overflow: bool = False) -> dict:
    runs = debugger.take_exported_runs()
    assert len(runs) == expected_runs
    run = runs[-1]
    decoded = run.get("decoded") or {}
    header = decoded.get("header") or {}
    records = decoded.get("records") or []
    assert records
    if overflow:
        assert int(header.get("overflow_count", 0)) > 0, header
        assert int(header.get("flags", 0)) & 1, header
    else:
        assert int(header.get("overflow_count", 0)) == 0
    for field in ("report_path", "json_report_path", "op_log_report_path"):
        path = run.get(field)
        assert path, f"missing {field}"
        assert Path(path).is_file(), path
    return run


def assert_close(actual,
                 expected,
                 *,
                 rtol: float = 1e-4,
                 atol: float = 1e-4) -> None:
    torch.testing.assert_close(
        actual.detach().cpu(),
        expected.detach().cpu(),
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    )


def summaries_by_op_instance(
        run: dict) -> dict[tuple[int, int], dict[str, float | int]]:
    summaries: dict[tuple[int, int], dict[str, float | int]] = {}
    for record in run["decoded"]["records"]:
        kind = record.get("record_kind")
        key = (int(record.get("op_id",
                              0)), int(record.get("logical_instance_id", 0)))
        values = summaries.setdefault(key, {})
        if kind == COUNT_RECORD:
            for name in ("nan_count", "inf_count", "zero_count",
                         "element_count"):
                values[name] = int(record[name])
        elif kind == VALUE_RECORD:
            for name in ("mean", "min", "max", "l2_norm"):
                values[name] = float(record[name])
    return summaries


def _tracked_op(
    run: dict,
    *,
    result_name: str | None = None,
    access_type: str | None = None,
) -> dict:
    rows = list(run.get("debug_tracked_table") or [])
    if result_name is not None:
        rows = [
            row for row in rows
            if row.get("statementResultName") == result_name
        ]
    if access_type is not None:
        rows = [row for row in rows if row.get("accessType") == access_type]
    assert len(rows) == 1, rows
    return rows[0]


def summary_for_result(run: dict,
                       result_name: str,
                       *,
                       instance: int = 0) -> dict[str, float | int]:
    row = _tracked_op(run, result_name=result_name)
    summary = summaries_by_op_instance(run).get((int(row["opId"]), instance))
    assert summary is not None
    assert set(summary) == {
        "nan_count",
        "inf_count",
        "zero_count",
        "element_count",
        "mean",
        "min",
        "max",
        "l2_norm",
    }
    return summary


def expected_summary(values) -> dict[str, float | int]:
    flat = values.detach().float().cpu().reshape(-1)
    finite = torch.isfinite(flat)
    finite_values = flat[finite].double()
    result: dict[str, float | int] = {
        "nan_count": int(torch.isnan(flat).sum()),
        "inf_count": int(torch.isinf(flat).sum()),
        "zero_count": int(((flat == 0) & finite).sum()),
        "element_count": flat.numel(),
        "mean": 0.0,
        "min": 0.0,
        "max": 0.0,
        "l2_norm": 0.0,
    }
    if finite_values.numel():
        result.update({
            "mean":
            float(finite_values.mean()),
            "min":
            float(finite_values.min()),
            "max":
            float(finite_values.max()),
            "l2_norm":
            math.sqrt(float(torch.sum(finite_values * finite_values))),
        })
    return result


def assert_summary(actual: dict[str, float | int],
                   expected: dict[str, float | int]) -> None:
    for name in ("nan_count", "inf_count", "zero_count", "element_count"):
        assert int(actual[name]) == int(expected[name]), name
    for name in ("mean", "min", "max", "l2_norm"):
        assert float(actual[name]) == pytest.approx(float(expected[name]),
                                                    rel=2e-4,
                                                    abs=2e-4), name


def address_summary(run: dict,
                    access_type: str,
                    *,
                    instance: int = 0) -> dict[str, int]:
    row = _tracked_op(run, access_type=access_type)
    op_id = int(row["opId"])
    result: dict[str, int] = {}
    for record in run["decoded"]["records"]:
        if record.get("record_kind") != MEMORY_RECORD:
            continue
        if int(record.get("op_id", 0)) != op_id:
            continue
        if int(record.get("logical_instance_id", 0)) != instance:
            continue
        event_kind = record.get("event_kind")
        if isinstance(event_kind, str):
            name = event_kind.lower()
        else:
            name = MEMORY_EVENT_NAMES[int(event_kind)]
        result[name] = int(record.get("addr", 0))
    required = {
        "first_addr",
        "last_addr",
        "min_addr",
        "max_addr",
        "active_lane_count",
        "address_span_bytes",
    }
    assert required <= result.keys(), result
    return result
