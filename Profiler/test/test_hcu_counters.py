"""DTK hardware counter import must reject incomplete or non-finite data."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from flagtree.profiler.hcu import import_rocprof_csv, export_rocprof, DEFAULT_COUNTERS


def write_csv(path, **overrides):
    row = dict(KernelName="copy",
               Index="0",
               BeginNs="1000",
               EndNs="2000",
               SQ_WAVES="16")
    row.update({"gpu-id": "2", "queue-id": "0"})
    row.update(overrides)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)


@pytest.mark.parametrize("change", [{
    "SQ_WAVES": "nan"
}, {
    "SQ_WAVES": "-1"
}, {
    "EndNs": "1000"
}, {
    "BeginNs": "0"
}])
def test_reject_invalid_counter_csv(tmp_path, change):
    path = tmp_path / "invalid.csv"
    write_csv(path, **change)
    with pytest.raises(ValueError):
        import_rocprof_csv(path, ["SQ_WAVES"])


def test_counter_export_preserves_namespace_and_units(tmp_path):
    path = tmp_path / "input.csv"
    write_csv(path, SQ_WAVES="9007199254740993")
    base = tmp_path / "profile"
    result = export_rocprof(base, path, ["SQ_WAVES"])
    a = result["associations"][0]
    assert a["runtime_event"]["device_id"] == 2
    assert a["metrics"]["hcu.device_id_namespace"] == "rocprof_hsa_agent_index"
    assert a["metrics"]["hcu.hardware.SQ_WAVES"] == 9007199254740993
    timeline = json.loads(Path(str(base) + ".timeline.json").read_text())
    assert timeline["traceEvents"][0]["dur"] == 1


def test_real_hardware_counter_workload_scaling(tmp_path):
    torch = pytest.importorskip("torch")
    triton = pytest.importorskip("triton")
    if ".hcu." not in type(triton.runtime.driver.active).__module__:
        pytest.skip("HCU hardware required")
    script = tmp_path / "workload.py"
    script.write_text('''import torch
import triton
import triton.language as tl
@triton.jit
def counter_scaling(X,Y,B:tl.constexpr):
 i=tl.program_id(0)*B+tl.arange(0,B)
 tl.store(Y+i,tl.load(X+i)*2+1)
x=torch.ones(65536,device="cuda");y=torch.empty_like(x)
for grid in (64,256):
 for _ in range(3): counter_scaling[(grid,)](x,y,256)
torch.cuda.synchronize()
torch.testing.assert_close(y,x*2+1)
''')
    base = tmp_path / "hardware"
    result = subprocess.run([
        sys.executable, "-m", "flagtree.profiler.cli", "-b", "hcu",
        "--rocprof", "-n",
        str(base),
        str(script)
    ],
                            capture_output=True,
                            text=True,
                            timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(Path(str(base) + ".vendor.json").read_text())
    records = [
        a for a in report["associations"]
        if "counter_scaling" in a["runtime_event"]["op_name"]
    ]
    assert len(records) == 6
    for metric in DEFAULT_COUNTERS:
        values = [a["metrics"][f"hcu.hardware.{metric}"] for a in records]
        assert all(v > 0 for v in values)
        assert sum(values[3:]) == pytest.approx(4 * sum(values[:3]), rel=0.05)


def test_counter_capture_rejects_failed_application(tmp_path):
    triton = pytest.importorskip("triton")
    if ".hcu." not in type(triton.runtime.driver.active).__module__:
        pytest.skip("HCU hardware required")
    script = tmp_path / "failed.py"
    script.write_text(
        'import torch\nx = torch.ones(1024, device="cuda")\ntorch.cuda.synchronize()\nraise SystemExit(7)\n'
    )
    base = tmp_path / "failed"
    result = subprocess.run([
        sys.executable, "-m", "flagtree.profiler.cli", "-b", "hcu",
        "--rocprof", "-n",
        str(base),
        str(script)
    ],
                            capture_output=True,
                            text=True,
                            timeout=120)
    assert result.returncode != 0
    assert not Path(str(base) + ".vendor.json").exists()
    assert "failed" in result.stderr.lower(
    ) or "exited" in result.stderr.lower()
