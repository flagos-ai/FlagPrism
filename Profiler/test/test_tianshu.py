import importlib
import json
from types import SimpleNamespace

from flagtree.profiler import tianshu

profiler_cli = importlib.import_module("flagtree.profiler.cli")
profiler_profile = importlib.import_module("flagtree.profiler.profile")


def test_ixkn_command_includes_kernel_filter(monkeypatch):
    monkeypatch.setattr(tianshu, "find_ixkn_cli", lambda explicit=None: "/opt/ixkn-cli")

    command = tianshu.build_ixkn_command(
        ["python3", "workload.py"],
        devices="0",
        sections="LaunchStats,Memory",
        kernel_name="_vector_add_kernel",
        launch_count=1,
        export_profile="/tmp/vector_add.ixkn",
    )

    assert command == [
        "/opt/ixkn-cli",
        "--devices",
        "0",
        "--section",
        "LaunchStats,Memory",
        "--kernel-name",
        "_vector_add_kernel",
        "--launch-count",
        "1",
        "--export-profile",
        "/tmp/vector_add.ixkn",
        "--force-overwrite",
        "--csv",
        "python3",
        "workload.py",
    ]


def test_tianshu_ir_summary_uses_unique_memory_instances():
    run = {
        "runtime_metadata": {
            "host_start_time_ns": 1_000_000_000_000_001,
            "host_end_time_ns": 1_000_000_000_001_001,
        },
        "debug_tracked_table": [
            {
                "opId": 3,
                "mlirOpName": "tt.load",
                "accessBytes": 4,
                "accessType": "load",
                "result": {"vecWidth": 256},
            },
            {
                "opId": 6,
                "mlirOpName": "tt.store",
                "accessBytes": 4,
                "accessType": "store",
                "result": {"vecWidth": 256},
            },
        ],
        "decoded": {
            "records": [
                {
                    "record_kind": "MEMORY_EVENT",
                    "op_id": op_id,
                    "logical_instance_id": instance_id,
                    "event_kind": event_kind,
                }
                for op_id in (3, 6)
                for instance_id in (0, 1)
                for event_kind in (3, 4, 5, 6, 7, 8)
            ]
        },
    }

    summary = profiler_profile._run_ir_summary(run)

    assert summary["host_start_time_ns"] == 1_000_000_000_000_001
    assert summary["host_duration_ns"] == 1_000
    assert summary["memory_access_bytes"] == 4096
    assert summary["memory_read_bytes"] == 2048
    assert summary["memory_write_bytes"] == 2048


def test_cli_forwards_tianshu_mode(monkeypatch):
    calls = []
    monkeypatch.setattr(
        profiler_cli,
        "start",
        lambda name, **options: calls.append((name, options)),
    )
    monkeypatch.setattr(profiler_cli, "do_setup_and_execute", lambda target: None)
    monkeypatch.setattr(profiler_cli, "finalize", lambda: None)

    args = SimpleNamespace(
        backend="tianshu",
        context="shadow",
        data="tree",
        hook="instrumentation",
        ixkn=False,
        mode="runtime_base:vendor_metrics=memory",
        name="tianshu_profile",
    )
    profiler_cli.run_profiling(args, ["workload.py"])

    assert calls == [
        (
            "tianshu_profile",
            {
                "backend": "tianshu",
                "context": "shadow",
                "data": "tree",
                "hook": "instrumentation",
                "mode": "runtime_base:vendor_metrics=memory",
            },
        )
    ]


def test_host_launch_timing_synthesizes_timeline(tmp_path):
    timeline_path = tmp_path / "tianshu.timeline.json"
    timeline_path.write_text(
        json.dumps({"displayTimeUnit": "us", "traceEvents": []}),
        encoding="utf-8",
    )
    run = {
        "debug_kernel_name": "vector_add",
        "runtime_metadata": {
            "host_start_time_ns": 1_000_000,
            "host_end_time_ns": 1_010_000,
        },
        "debug_tracked_table": [],
        "decoded": {"records": []},
    }

    profiler_profile._augment_timeline(timeline_path, [run])

    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    kernel_events = [
        event
        for event in timeline["traceEvents"]
        if event.get("cat") == "flagtree.ir_kernel"
    ]
    assert len(kernel_events) == 1
    assert kernel_events[0]["name"] == "vector_add"
    assert kernel_events[0]["ts"] == 1_000
    assert kernel_events[0]["dur"] == 10
    assert kernel_events[0]["args"]["timestamp_unit"] == "host ns"
