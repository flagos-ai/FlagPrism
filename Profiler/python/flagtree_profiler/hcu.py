"""Explicit DTK ROCProfiler process capture and lossless counter import.

This path runs the application once with one PMC group. It is separate from
in-process ROCtracer sessions: hardware profiling may serialize dispatches.
Counter names and device IDs retain DTK's units and HSA-agent namespace.
"""
from __future__ import annotations

import csv
import json
from decimal import Decimal
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

DEFAULT_COUNTERS = ("SQ_WAVES", "SQ_INSTS_VALU", "SQ_INSTS_SALU")


def import_rocprof_csv(path, counters):
    associations = []
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "KernelName", "gpu-id", "queue-id", "Index", "BeginNs", "EndNs",
            *counters
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"ROCProfiler CSV missing columns: {sorted(required - set(reader.fieldnames or []))}"
            )
        for row in reader:
            begin, end = int(row["BeginNs"]), int(row["EndNs"])
            if begin <= 0 or end <= begin:
                raise ValueError(
                    "ROCProfiler kernel has invalid device timestamps")
            metrics = {}
            for name in counters:
                value = Decimal(row[name])
                if not value.is_finite() or value < 0:
                    raise ValueError(
                        f"Invalid ROCProfiler counter {name}: {row[name]}")
                metrics[f"hcu.hardware.{name}"] = int(
                    value) if value == value.to_integral_value() else float(
                        value)
            metrics["hcu.device_id_namespace"] = "rocprof_hsa_agent_index"
            associations.append({
                "state": "collected",
                "source": "hcu_rocprof_hardware_counter",
                "note":
                "Per-dispatch hardware counter; raw DTK units. Profiling can serialize dispatches.",
                "runtime_event": {
                    "scope_id": 0,
                    "op_name": row["KernelName"],
                    "task_id": int(row["Index"]),
                    "correlation_id": int(row["Index"]),
                    "device_id": int(row["gpu-id"]),
                    "stream_id": int(row["queue-id"]),
                    "start_time_ns": begin,
                    "end_time_ns": end
                },
                "metrics": metrics
            })
    if not associations:
        raise ValueError("ROCProfiler produced no kernel counter records")
    return associations


def export_rocprof(name, csv_path, counters):
    associations = import_rocprof_csv(csv_path, counters)
    base = Path(name)
    vendor = {
        "schema_version": 1,
        "backend": "hcu",
        "importer": "dtk_rocprof_csv",
        "requested_metrics": [{
            "name": c,
            "required": True
        } for c in counters],
        "enabled_metrics": list(counters),
        "raw_inputs": [str(csv_path)],
        "degrade_reasons": [],
        "associations": associations
    }
    meta = {
        "schema_version":
        1,
        "backend":
        "hcu",
        "mode":
        "external_hardware_counters",
        "device_id_namespace":
        "rocprof_hsa_agent_index",
        "application_runs":
        1,
        "counter_units":
        "raw DTK metric units",
        "timing_source":
        "rocprof_device_timestamps",
        "limitations": [
            "Dispatches may be serialized by ROCProfiler",
            "Not an in-process start/stop session",
            "HSA agent indices are not HIP device ordinals"
        ]
    }
    timeline = {"displayTimeUnit": "ns", "traceEvents": []}
    children = []
    for a in associations:
        e = a["runtime_event"]
        duration = e["end_time_ns"] - e["start_time_ns"]
        timeline["traceEvents"].append({
            "ph": "X",
            "cat": "hcu_rocprof_hardware_counter",
            "name": e["op_name"],
            "pid": e["device_id"],
            "tid": e["stream_id"],
            "ts": e["start_time_ns"] / 1000,
            "dur": duration / 1000,
            "args": a["metrics"]
        })
        children.append({
            "frame": {
                "name": e["op_name"],
                "type": "function"
            },
            "metrics": {
                "time (ns)": duration,
                "count": 1,
                "device_id": e["device_id"],
                **a["metrics"]
            },
            "children": []
        })
    outputs = {
        ".vendor.json":
        vendor,
        ".meta.json":
        meta,
        ".timeline.json":
        timeline,
        ".hatchet": [{
            "frame": {
                "name": "ROOT",
                "type": "function"
            },
            "metrics": {},
            "children": children
        }, {}]
    }
    for suffix, document in outputs.items():
        Path(str(base) +
             suffix).write_text(json.dumps(document, indent=2) + "\n")
    return vendor


def run_rocprof(name, target_args, counters=DEFAULT_COUNTERS, executable=None):
    if not target_args:
        raise ValueError("--rocprof requires a script or pytest target")
    counters = tuple(counters)
    if not counters or any(
            not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\[\d+\])?", c)
            for c in counters):
        raise ValueError(
            "Counter names must be DTK metric identifiers; only one PMC group is supported"
        )
    cli = executable or shutil.which(
        "rocprof") or "/opt/dtk/rocprofiler/bin/rocprof"
    if not Path(cli).is_file():
        raise FileNotFoundError(f"ROCProfiler executable not found: {cli}")
    base = Path(name).resolve()
    base.parent.mkdir(parents=True, exist_ok=True)
    # A fresh CSV prevents a failed capture from importing stale results.
    with tempfile.TemporaryDirectory(prefix="flagtree-hcu-",
                                     dir=base.parent) as temp:
        config = Path(temp) / "counters.txt"
        config.write_text("pmc: " + " ".join(counters) + "\n")
        output = Path(temp) / "counters.csv"
        command = [
            str(cli), "-i",
            str(config), "--timestamp", "on", "-o",
            str(output), sys.executable, "-m", f"{__package__}.hcu", "--child",
            *target_args
        ]
        status_path = Path(temp) / "child-status.json"
        child_env = {
            **os.environ, "FLAGPRISM_HCU_CHILD_STATUS": str(status_path)
        }
        with Path(str(base) + ".rocprof.log").open("w") as log:
            completed = subprocess.run(command,
                                       stdout=log,
                                       stderr=subprocess.STDOUT,
                                       check=False,
                                       env=child_env)
        if completed.returncode:
            raise RuntimeError(
                f"ROCProfiler exited {completed.returncode}; see {base}.rocprof.log"
            )
        if not status_path.is_file() or json.loads(
                status_path.read_text()).get("exit_code") != 0:
            raise RuntimeError(
                f"Profiled application failed; see {base}.rocprof.log")
        # Validate before publishing any success report.
        import_rocprof_csv(output, counters)
        saved = Path(str(base) + ".rocprof.csv")
        shutil.copyfile(output, saved)
        return export_rocprof(base, saved, counters)


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "--child":
        raise SystemExit(
            "Use flagtree-profiler --backend hcu --rocprof script.py")
    from .flags import set_command_line
    set_command_line()
    exit_code = 0
    try:
        if Path(sys.argv[2]).name == "pytest":
            import pytest
            exit_code = pytest.main(sys.argv[3:])
            raise SystemExit(exit_code)
        import runpy
        target = Path(sys.argv[2]).resolve()
        sys.argv = [str(target), *sys.argv[3:]]
        sys.path.insert(0, str(target.parent))
        runpy.run_path(str(target), run_name="__main__")
    except SystemExit as error:
        exit_code = error.code or 0
        raise
    except BaseException:
        exit_code = 1
        raise
    finally:
        status = os.getenv("FLAGPRISM_HCU_CHILD_STATUS")
        if status:
            Path(status).write_text(json.dumps({"exit_code": exit_code}))
