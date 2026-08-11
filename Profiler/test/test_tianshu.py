import importlib

from flagtree.profiler import tianshu


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
