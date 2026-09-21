import importlib.util
import pytest
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace


def _has_ascend_spec() -> bool:
    try:
        import triton
    except ImportError:
        return False
    return (Path(triton.__file__).parent / "spec" / "ascend").exists()


@pytest.mark.skipif(
    not _has_ascend_spec(),
    # FlagPrism: keep Ascend-specific compatibility checks out of NVIDIA-only
    # source distributions that do not contain the Ascend spec package.
    reason="Ascend Triton spec is not available in this backend build",
)
def test_debug_collect_runtime_uses_flagtree_backend_for_backend_name(
        monkeypatch):
    from flagtree.debugger.runtime import DebugCollectRuntime

    monkeypatch.setenv("FLAGTREE_BACKEND", "ascend")
    metadata = {
        "target": SimpleNamespace(backend="npu", arch="Ascend910B4"),
        "debug_target_name": "Ascend910B4",
    }
    normalized = DebugCollectRuntime._normalize_launch_metadata(metadata)
    assert normalized["debug_backend_name"] == "ascend"
    assert normalized["debug_target_name"] == "Ascend910B4"


def test_debug_collect_runtime_does_not_infer_backend_from_npu_target(
        monkeypatch):
    from flagtree.debugger.runtime import DebugCollectRuntime

    monkeypatch.delenv("FLAGTREE_BACKEND", raising=False)
    metadata = {
        "target": SimpleNamespace(backend="npu", arch="Ascend910B4"),
        "debug_target_name": "Ascend910B4",
    }
    normalized = DebugCollectRuntime._normalize_launch_metadata(metadata)
    assert normalized["debug_backend_name"] == "npu"


def test_debug_collect_runtime_prepare_export_decodes_header():
    from flagtree.debugger.runtime import default_debug_collect_runtime

    md = SimpleNamespace()
    md.debug_kernel_id = 7
    md.debug_enabled = True
    md.debug_protocol_version = 1
    md.debug_record_level = 1
    md.debug_export_mode = "POST_KERNEL_EXPORT"
    md.debug_record_capacity = 8
    md.debug_backend_name = "cuda"
    md.debug_metadata_json = json.dumps({
        "debugKernelId": 7,
        "kernelName": "unit_empty_kernel",
        "backendName": "cuda",
        "targetName": "host",
        "scopeCount": 0,
        "trackedOpCount": 0,
        "trackedOps": [],
    })
    default_debug_collect_runtime.clear_exported_runs()
    ctx = default_debug_collect_runtime.prepare(md, stream=None)
    assert ctx.debug_kernel_id == 7
    assert default_debug_collect_runtime.hidden_arg(ctx) != 0
    run = default_debug_collect_runtime.export(ctx, stream=None)
    assert run.debug_kernel_id == 7
    assert len(run.raw_buffer) >= 32
    assert run.decoded["header"]["capacity"] == 64
    assert run.decoded["header"]["record_size"] == 32
    assert run.decoded["records"] == []
    assert "FlagTree Debug Report" in run.report
    assert "kernel_name: unit_empty_kernel" in run.report
    assert "record_count: 0" in run.report
    assert default_debug_collect_runtime.peek_exported_runs() == [run]
    assert default_debug_collect_runtime.take_exported_runs() == [run]
    assert default_debug_collect_runtime.peek_exported_runs() == []


def test_debugger_binding_decodes_and_reports_summary_record():
    from flagtree.debugger.native import runtime_binding
    dbg = runtime_binding()
    assert dbg is not None

    header = struct.pack("<IIIIIIII", 1, 1, 0, 0, 32, 64, 0, 0)
    summary = struct.pack("<HHIQHHId", 1, 0, 1, 42, 6, 3, 0, 3.5)
    exported = {
        "meta": {
            "run_id": 9,
            "device_id": 0,
            "kernel_id": 7,
            "protocol_version": 1,
            "record_level": 1,
            "export_mode": 1,
            "backend_kind": 1,
        },
        "runtime_metadata": {
            "buffers": [],
            "tensors": []
        },
        "raw_buffer": header + summary,
    }
    metadata_json = json.dumps({
        "debugKernelId": 7,
        "kernelName": "unit_kernel",
        "backendName": "cuda",
        "targetName": "host",
        "scopeCount": 1,
        "trackedOpCount": 0,
        "trackedOps": [],
    })

    decoded = dbg.decode_exported_run(exported)
    assert decoded["header"]["write_idx"] == 1
    assert decoded["records"][0]["record_kind"] == "SUMMARY"
    assert decoded["records"][0]["op_id"] == 1
    assert decoded["records"][0]["logical_instance_id"] == 42

    report = dbg.render_text_report(exported, metadata_json)
    assert "FlagTree Debug Report" in report
    assert "record_count: 1" in report
    assert "IR Op Log Records" in report
    assert "summary:" in report
    assert "element_count: [3.5 (F64)]" in report

    json_report = dbg.render_json_report(exported, metadata_json)
    assert '"records_by_op"' in json_report
    assert '"instances":[42]' in json_report
    assert '"summary"' in json_report


@pytest.mark.parametrize("result_type", [2, 3])
@pytest.mark.parametrize("raw_value,expected", [
    (3.5, 3.5),
    (float("inf"), "Infinity"),
    (float("-inf"), "-Infinity"),
    (float("nan"), "NaN"),
])
def test_debugger_summary_json_preserves_nonfinite_values(
        result_type, raw_value, expected):
    from flagtree.debugger.native import runtime_binding
    dbg = runtime_binding()
    assert dbg is not None
    header = struct.pack("<8I", 1, 1, 0, 0, 32, 64, 0, 0)
    record = struct.pack("<HHIQHHI", 1, 0, 1, 42, 8, result_type, 0)
    record += struct.pack("<f4x" if result_type == 2 else "<d", raw_value)
    exported = {"meta": {"protocol_version": 1}, "raw_buffer": header + record}
    metadata = json.dumps({
        "debugKernelId": 7,
        "kernelName": "nonfinite_summary",
        "backendName": "host",
        "targetName": "host",
        "scopeCount": 0,
        "trackedOpCount": 0,
        "trackedOps": [],
    })

    def cells(node):
        if isinstance(node, dict):
            if node.get("result_type") in ("F32", "F64") and "value" in node:
                yield node["value"]
            for value in node.values():
                yield from cells(value)
        elif isinstance(node, list):
            for value in node:
                yield from cells(value)

    for render in (dbg.render_json_report, dbg.render_json_op_log_report):
        document = json.loads(
            render(exported, metadata),
            parse_constant=lambda value: pytest.fail(
                f"Non-standard JSON token: {value}"),
        )
        values = list(cells(document))
        assert values and all(value == expected for value in values)


def test_debugger_binding_decodes_deterministic_compact_bundle_records():
    from flagtree.debugger.native import runtime_binding
    dbg = runtime_binding()
    assert dbg is not None

    record_size = 64
    capacity = 4
    payload_offset = 32 + capacity * record_size
    header = struct.pack("<IIIIIIII", 4, capacity, 0, 0, record_size,
                         payload_offset, 0, 0)
    records = [bytearray(record_size) for _ in range(capacity)]
    struct.pack_into("<QQQQ", records[0], 16, 0, 0, 1, 16)
    struct.pack_into("<ffff", records[1], 16, 4.0, 0.0, 8.0, 18.5)
    struct.pack_into("<QQQQ", records[2], 16, 1, 2, 3, 32)
    struct.pack_into("<ffff", records[3], 16, 5.0, 1.0, 9.0, 22.0)
    exported = {
        "meta": {
            "run_id": 10,
            "device_id": 0,
            "kernel_id": 8,
            "protocol_version": 2,
            "record_level": 1,
            "export_mode": 1,
            "backend_kind": 1,
        },
        "runtime_metadata": {
            "buffers": [],
            "tensors": [],
            "records_per_instance":
            2,
            "record_layout":
            "deterministic_compact_v1",
            "record_plan": [
                {
                    "record_index": 0,
                    "op_id": 7,
                    "scope_id": 1,
                    "record_kind": 4,
                    "collector_kind": 0,
                    "result_type": 0,
                    "event_kind": 0,
                },
                {
                    "record_index": 1,
                    "op_id": 7,
                    "scope_id": 1,
                    "record_kind": 5,
                    "collector_kind": 0,
                    "result_type": 0,
                    "event_kind": 0,
                },
            ],
        },
        "raw_buffer": header + b"".join(bytes(record) for record in records),
    }

    decoded = dbg.decode_exported_run(exported)
    assert [record["record_kind"] for record in decoded["records"]] == [
        "SUMMARY_COUNT_BUNDLE_U64",
        "SUMMARY_VALUE_BUNDLE_F32",
        "SUMMARY_COUNT_BUNDLE_U64",
        "SUMMARY_VALUE_BUNDLE_F32",
    ]
    assert [record["op_id"] for record in decoded["records"]] == [7, 7, 7, 7]
    assert [record["logical_instance_id"]
            for record in decoded["records"]] == [0, 0, 1, 1]
    assert decoded["records"][0]["nan_count"] == 0
    assert decoded["records"][0]["zero_count"] == 1
    assert decoded["records"][2]["nan_count"] == 1
    assert decoded["records"][3]["mean"] == pytest.approx(5.0)


def test_debugger_binding_decodes_deterministic_compact_timeline_record():
    from flagtree.debugger.native import runtime_binding
    dbg = runtime_binding()
    assert dbg is not None

    record_size = 64
    capacity = 1
    payload_offset = 32 + capacity * record_size
    header = struct.pack("<IIIIIIII", 1, capacity, 0, 0, record_size,
                         payload_offset, 0, 0)
    record = bytearray(record_size)
    struct.pack_into("<QQQ", record, 16, 100, 145, 45)
    exported = {
        "meta": {
            "run_id": 11,
            "device_id": 0,
            "kernel_id": 9,
            "protocol_version": 2,
            "record_level": 1,
            "export_mode": 1,
            "backend_kind": 4,
        },
        "runtime_metadata": {
            "buffers": [],
            "tensors": [],
            "records_per_instance":
            1,
            "record_layout":
            "deterministic_compact_v1",
            "record_plan": [
                {
                    "record_index": 0,
                    "op_id": 7,
                    "scope_id": 1,
                    "record_kind": 6,
                    "collector_kind": 0,
                    "result_type": 0,
                    "event_kind": 0,
                },
            ],
        },
        "raw_buffer": header + bytes(record),
    }

    decoded = dbg.decode_exported_run(exported)
    assert decoded["records"] == [{
        "record_kind": "TIMELINE",
        "op_id": 7,
        "logical_instance_id": 0,
        "start_cycle": 100,
        "end_cycle": 145,
        "duration_cycle": 45,
    }]


@pytest.mark.skipif(
    not _has_ascend_spec(),
    reason="Ascend Triton spec is not available in this backend build",
)
def test_ascend_spec_compiled_kernel_keeps_core_launch_metadata_contract(
        monkeypatch):
    import triton

    compiler_path = (Path(triton.__file__).parent / "spec" / "ascend" /
                     "compiler" / "compiler.py")
    spec = importlib.util.spec_from_file_location(
        "triton.compiler._ascend_spec_compiler_under_test",
        compiler_path,
    )
    ascend_compiler = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, spec.name, ascend_compiler)
    spec.loader.exec_module(ascend_compiler)

    kernel = SimpleNamespace(
        metadata=SimpleNamespace(debug_enabled=True),
        name="debug_grid_kernel",
        function=0x1234,
        src=object(),
        _init_handles=lambda: None,
    )

    launch_metadata = ascend_compiler.CompiledKernel.launch_metadata(
        kernel, (7, ), 99)

    if launch_metadata is not None:
        assert "grid" not in launch_metadata.get()


@pytest.mark.skipif(
    not _has_ascend_spec(),
    reason="Ascend Triton spec is not available in this backend build",
)
def test_ascend_spec_jit_applies_instrumentation_mode_to_compile_options(
    monkeypatch, ):
    import triton
    import triton.backends.ascend as ascend_backend

    jit_path = Path(
        triton.__file__).parent / "spec" / "ascend" / "runtime" / "jit.py"
    if not jit_path.exists():
        jit_path = (Path(ascend_backend.__file__).parent / "spec" / "triton" /
                    "runtime" / "jit.py")
    spec = importlib.util.spec_from_file_location(
        "triton.runtime._ascend_spec_jit_instrumentation_under_test",
        jit_path,
    )
    ascend_jit = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, spec.name, ascend_jit)
    spec.loader.exec_module(ascend_jit)

    from flagtree.debugger.compiler import set_instrumentation_mode

    set_instrumentation_mode("debugger")
    try:
        kwargs = {}
        ascend_jit._flagprism.apply_compile_options(kwargs)
        assert kwargs["instrumentation_mode"].startswith("debugger|")

        explicit = {"instrumentation_mode": "custom"}
        ascend_jit._flagprism.apply_compile_options(explicit)
        assert explicit == {"instrumentation_mode": "custom"}
    finally:
        set_instrumentation_mode("")


@pytest.mark.skipif(
    not _has_ascend_spec(),
    reason="Ascend Triton spec is not available in this backend build",
)
def test_ascend_options_hash_includes_instrumentation_mode():
    from triton.backends.ascend.compiler import NPUOptions

    plain = NPUOptions(arch="Ascend910B4")
    instrumented = NPUOptions(
        arch="Ascend910B4",
        instrumentation_mode="debugger",
    )

    assert plain.hash() != instrumented.hash()


@pytest.mark.parametrize("mode,requested,expected", [
    ("", False, False),
    ("", True, True),
    ('debugger|config={"debug_record_level":1}', False, False),
    ('debugger|config={"debug_record_level":1}', True, True),
    ('debugger|config={"debug_record_level":2}', False, True),
    ('debugger|config={"debug_record_level":2}', True, True),
    ('profiler|config={"debug_record_level":2}', False, False),
])
def test_enflame_l2_requires_i64_lowering(monkeypatch, mode, requested,
                                          expected):
    compiler = pytest.importorskip("triton.backends.enflame.compiler")
    # Exercise option precedence without opening a device. FlagGems explicitly
    # passes ENABLE_I64=False for f32 operators, but L2 adds i64 payloads.
    fields = compiler.GCUOptions.__dataclass_fields__

    class Options:
        __dataclass_fields__ = fields

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setattr(compiler, "GCUOptions", Options)
    options = compiler._GCUBackend.parse_options(None, {
        "ENABLE_I64": requested,
        "instrumentation_mode": mode,
    })
    assert options.enable_i64 is expected


def test_l2_host_summary_uses_device_payload(tmp_path):
    import numpy as np
    from flagtree.debugger.api import _fill_summary_bundles_from_full_dump

    payload = tmp_path / "values.npy"
    np.save(payload,
            np.array([0, 3, 4, np.nan, np.inf, -np.inf], dtype=np.float32))
    run = {"raw_buffer": bytes(32 + 2 * 64), "runtime_metadata": {}}
    decoded = {
        "records": [
            {
                "record_kind": "SUMMARY_COUNT_BUNDLE_U64",
                "op_id": 7,
                "logical_instance_id": 0
            },
            {
                "record_kind": "SUMMARY_VALUE_BUNDLE_F32",
                "op_id": 7,
                "logical_instance_id": 0
            },
        ]
    }
    artifacts = [{
        "op_id": 7,
        "logical_instance_id": 0,
        "kind": "value",
        "path": str(payload)
    }]
    _fill_summary_bundles_from_full_dump(
        run, decoded, {
            "debug_record_size": 64,
            "debug_full_dump_plan": [{
                "kind": "value",
                "op_id": 7
            }]
        }, artifacts)
    assert struct.unpack_from("<4Q", run["raw_buffer"], 48) == (1, 2, 1, 6)
    assert struct.unpack_from("<4f", run["raw_buffer"], 112) == pytest.approx(
        (7 / 3, 0, 4, 5))
    assert run["runtime_metadata"][
        "summary_source"] == "host_from_device_full_dump"
    with pytest.raises(RuntimeError, match="Missing L2 value payload"):
        _fill_summary_bundles_from_full_dump(
            run, decoded, {
                "debug_record_size": 64,
                "debug_full_dump_plan": [{
                    "kind": "value",
                    "op_id": 7
                }]
            }, [])


def test_enflame_l2_integer_payload_roundtrip(tmp_path):
    import os
    import numpy as np
    import torch
    pytest.importorskip("torch_gcu")
    import triton
    import triton.language as tl
    from flagtree import debugger

    if triton.runtime.driver.active.get_current_target().backend != "gcu":
        pytest.skip("Enflame device required")
    device = int(os.environ.get("FLAGPRISM_TEST_DEVICE", "0"))
    torch.gcu.set_device(device)

    @triton.jit
    def integer_payload(X, Y, BLOCK: tl.constexpr):
        offsets = tl.arange(0, BLOCK)
        values = tl.load(X + offsets)
        positive = values > 0
        tl.store(Y + offsets, positive)

    values = np.array([-2147483648, -2, -1, 0, 1, 2, 2147483647, 0],
                      dtype=np.int32)
    x = torch.from_numpy(values.copy()).to(f"gcu:{device}")
    y = torch.empty(8, dtype=torch.bool, device=f"gcu:{device}")
    debugger.activate(auto_collect=True,
                      level=2,
                      addr_level=0,
                      output_dir=tmp_path)
    try:
        integer_payload[(1, )](x, y, 8)
        torch.gcu.synchronize()
        np.testing.assert_array_equal(y.cpu().numpy(), values > 0)
        artifacts = [
            a for run in debugger.take_exported_runs()
            for a in run["runtime_metadata"]["full_dump_artifacts"]
            if a["kind"] == "value" and a["artifact_dtype"] == "int64"
        ]
        arrays = [
            np.load(a["path"], allow_pickle=False).reshape(-1)
            for a in artifacts
        ]
        assert any(np.array_equal(v, values.astype(np.int64)) for v in arrays)
        assert any(
            np.array_equal(v, (values > 0).astype(np.int64)) for v in arrays)
    finally:
        debugger.deactivate()


@pytest.mark.parametrize("conditional,offset,length,expected", [
    (True, 0, 0, None),
    (False, 0, 0, "empty full-dump"),
    (True, 1, 0, "empty full-dump"),
    (True, 1024, 2, "size differs"),
])
def test_l2_inactive_slots_are_not_missing_capture(tmp_path, conditional,
                                                   offset, length, expected):
    from flagtree.debugger.api import _write_full_dump_artifacts

    run = {
        "raw_buffer": bytes(2048),
        "runtime_metadata": {
            "records_per_instance": 4
        }
    }

    def rec(kind, op, off=0, size=0):
        return {
            "record_kind": kind,
            "op_id": op,
            "logical_instance_id": 0,
            "payload_offset": off,
            "payload_length": size
        }

    decoded = {
        "header": {},
        "records": [
            rec("FULL_VALUE", 1, 1024, 4),
            rec("SUMMARY_COUNT_BUNDLE_U64", 2),
            rec("SUMMARY_VALUE_BUNDLE_F32", 2),
            rec("FULL_VALUE", 2, offset, length),
        ]
    }
    metadata = {
        "debug_full_dump_plan": [{
            "record_index": slot,
            "op_id": op,
            "artifact_dtype": "float32",
            "shape": [1],
            "kind": "value",
            "payload_length": 4,
            "conditional": cond
        } for slot, op, cond in [(0, 1, False), (3, 2, conditional)]]
    }
    if expected:
        with pytest.raises(RuntimeError, match=expected):
            _write_full_dump_artifacts(tmp_path / "run.txt", run, decoded,
                                       metadata)
    else:
        artifacts = _write_full_dump_artifacts(tmp_path / "run.txt", run,
                                               decoded, metadata)
        assert len(artifacts) == 1
        assert run["runtime_metadata"]["inactive_record_slots"] == [1, 2, 3]
        assert run["runtime_metadata"]["inactive_full_dump_records"][0][
            "op_id"] == 2


@pytest.mark.parametrize("instances", [1, 4])
@pytest.mark.parametrize("missing",
                         [None, "all", "plan", "instance", "active"])
def test_l2_all_inactive_requires_complete_plan(tmp_path, instances, missing):
    from flagtree.debugger.api import _write_full_dump_artifacts

    plans = [{
        "record_index": slot,
        "op_id": slot + 1,
        "payload_length": 4,
        "conditional": missing != "active" or slot == 0,
    } for slot in range(2)]
    records = [{
        "record_kind": "FULL_VALUE",
        "op_id": slot + 1,
        "logical_instance_id": instance,
        "payload_offset": 0,
        "payload_length": 0,
    } for instance in range(instances) for slot in range(2)]
    if missing == "all":
        records = []
    elif missing == "plan":
        records[-1]["record_kind"] = "SUMMARY"
    elif missing == "instance":
        records = records[:-2]
    run = {
        "raw_buffer": b"",
        "runtime_metadata": {
            "records_per_instance": 2,
            "grid": (2, 2, 1) if instances == 4 else (1, 1, 1)
        }
    }
    decoded = {"header": {}, "records": records}
    metadata = {"debug_full_dump_plan": plans}
    if missing:
        with pytest.raises(RuntimeError,
                           match="empty full-dump|did not produce"):
            _write_full_dump_artifacts(tmp_path / "run.txt", run, decoded,
                                       metadata)
        return
    assert _write_full_dump_artifacts(tmp_path / "run.txt", run, decoded,
                                      metadata) == []
    assert run["runtime_metadata"]["inactive_record_slots"] == list(
        range(2 * instances))
    index = json.loads(Path(run["full_dump_index_path"]).read_text())
    assert index["artifacts"] == []
    assert len(index["inactive_records"]) == 2 * instances


@pytest.mark.parametrize("execute_first", [True, False])
def test_enflame_l2_branch_and_zero_trip_loop(tmp_path, execute_first):
    import os
    import numpy as np
    import torch
    pytest.importorskip("torch_gcu")
    import triton
    import triton.language as tl
    import flagtree.language as fl
    from flagtree import debugger

    if triton.runtime.driver.active.get_current_target().backend != "gcu":
        pytest.skip("Enflame device required")
    device = int(os.environ.get("FLAGPRISM_TEST_DEVICE", "0"))
    torch.gcu.set_device(device)

    @triton.jit
    def branched(X, Y, N, EXECUTE_FIRST, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        if (pid == 0) & EXECUTE_FIRST:
            fl.debug_collect_start(level=2, addr_level=0)
            value = tl.load(X + offsets) + 3.0
            tl.store(Y + offsets, value)
            fl.debug_collect_end()
        else:
            for start in range(BLOCK, N, BLOCK):
                fl.debug_collect_start(level=2, addr_level=0)
                value = tl.load(X + offsets) * 2.0
                tl.store(Y + offsets, value)
                fl.debug_collect_end()

    x = torch.arange(8, dtype=torch.float32, device="cpu").to(f"gcu:{device}")
    y = torch.full_like(x, -1)
    debugger.activate(auto_collect=False,
                      level=2,
                      addr_level=0,
                      output_dir=tmp_path)
    try:
        branched[(2, )](x, y, 8, execute_first, 8)
        torch.gcu.synchronize()
        np.testing.assert_array_equal(
            y.cpu().numpy(),
            np.arange(8) + 3 if execute_first else np.full(8, -1))
        runs = debugger.take_exported_runs()
        assert runs
        inactive_count = 0
        for run in runs:
            if not execute_first:
                assert run["runtime_metadata"]["full_dump_artifacts"] == []
                assert run["decoded"]["records"] == []
                index = json.loads(
                    Path(run["full_dump_index_path"]).read_text())
                assert index["artifacts"] == []
                assert index["inactive_records"]
            inactive = run["runtime_metadata"]["inactive_full_dump_records"]
            inactive_count += len(inactive)
            inactive_keys = {(r["op_id"], r["logical_instance_id"])
                             for r in inactive}
            assert all(
                (r["op_id"], r["logical_instance_id"]) not in inactive_keys
                for r in run["decoded"]["records"])
            assert all(r["payload_length"] > 0
                       for r in run["decoded"]["records"]
                       if r["record_kind"] == "FULL_VALUE")
        assert inactive_count > 0
    finally:
        debugger.deactivate()


@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("capacity,grid,records,payload,message", [
    (64, (2, 1, 1), 40, 16, "debug_record_capacity"),
    (64, (1, 1, 1), 1, 2**32, "32-bit payload offsets"),
])
def test_l2_host_launch_rejects_unsafe_buffer(capacity, grid, records, payload,
                                              message, level):
    from flagtree.debugger.native import runtime_binding
    binding = runtime_binding()
    assert binding is not None
    metadata = {
        "name": "capacity_guard",
        "debug_enabled": True,
        "debug_backend_name": "enflame",
        "debug_record_level": level,
        "debug_record_capacity": capacity,
        "debug_record_size": 64,
        "debug_full_dump_payload_bytes_per_instance": payload
    }
    # These validations run before opening a device or allocating a buffer.
    with pytest.raises(ValueError, match=message):
        binding.prepare_launch(metadata, 0, {
            "grid": grid,
            "records_per_instance": records
        })


def _instrument_scalar_pair(tmp_path, first_type, first_level):
    from triton._C.libtriton import ir
    from flagtree.debugger.native import compiler_binding

    binding = compiler_binding()
    context = ir.context()
    ir.load_dialects(context)
    binding.load_dialects(context)
    source = f'''module attributes {{
      flagtree.debug.addr_level = 0 : i32,
      flagtree.debug.enable_hidden_arg_abi = true,
      flagtree.debug.record_level = 2 : i32
    }} {{
      tt.func @payload(%a: !tt.ptr<{first_type}>, %b: !tt.ptr<f32>) {{
        %x = tt.load %a {{flagtree.debug.op_id = 1 : i32,
                          flagtree.debug.record_level = {first_level} : i32}} : !tt.ptr<{first_type}>
        %y = tt.load %b {{flagtree.debug.op_id = 2 : i32}} : !tt.ptr<f32>
        tt.return
      }}
    }}'''
    path = tmp_path / "payload.mlir"
    path.write_text(source)
    module = ir.parse_mlir_module(str(path), context)
    passes = ir.pass_manager(context)
    binding.add_insert_instrumentation(passes)
    passes.run(module, "payload_regression")
    return binding, module, context


def test_l2_program_stride_preserves_entry_alignment(tmp_path):
    binding, module, context = _instrument_scalar_pair(tmp_path, "i64", 2)
    stride = binding.get_debug_full_dump_payload_bytes_per_instance(module)
    plan = json.loads(binding.get_debug_full_dump_plan_json(module))
    assert stride == 16
    assert [(p["payload_offset"], p["payload_length"])
            for p in plan] == [(0, 8), (8, 4)]
    for program in range(3):
        for entry in plan:
            assert (program * stride +
                    entry["payload_offset"]) % entry["element_bytes"] == 0


def test_l2_host_summary_preserves_l1_records(tmp_path):
    import numpy as np
    from flagtree.debugger.api import _fill_summary_bundles_from_full_dump

    payload = tmp_path / "values.npy"
    np.save(payload, np.array([2, 4], dtype=np.float32))
    raw = bytes([17]) * (32 + 4 * 64)
    run = {"raw_buffer": raw, "runtime_metadata": {}}
    records = [{
        "record_kind": kind,
        "op_id": op,
        "logical_instance_id": 0
    } for op in (1, 2) for kind in ("SUMMARY_COUNT_BUNDLE_U64",
                                    "SUMMARY_VALUE_BUNDLE_F32")]
    metadata = {
        "debug_record_size": 64,
        "debug_full_dump_plan": [{
            "kind": "value",
            "op_id": 2
        }]
    }
    artifacts = [{
        "op_id": 2,
        "logical_instance_id": 0,
        "kind": "value",
        "path": str(payload)
    }]
    _fill_summary_bundles_from_full_dump(run, {"records": records}, metadata,
                                         artifacts)
    assert run["raw_buffer"][:160] == raw[:160]
    assert run["runtime_metadata"][
        "summary_source"] == "mixed_device_and_host_from_device_full_dump"
    assert struct.unpack_from("<4Q", run["raw_buffer"], 176) == (0, 0, 0, 2)


@pytest.mark.parametrize("global_level", [1, 2])
@pytest.mark.parametrize("mixed", [False, True])
def test_enflame_l2_scalar_payload_content(tmp_path, mixed, global_level):
    import numpy as np
    import torch
    pytest.importorskip("torch_gcu")
    import triton
    import triton.language as tl
    import flagtree.language as fl
    from flagtree import debugger

    if triton.runtime.driver.active.get_current_target().backend != "gcu":
        pytest.skip("Enflame device required")

    @triton.jit
    def scalar_pair(A, B, C, D, MIXED: tl.constexpr):
        p = tl.program_id(0)
        if MIXED:
            fl.debug_collect_start(level=1, addr_level=0)
        else:
            fl.debug_collect_start(level=2, addr_level=0)
        a = tl.load(A + p)
        fl.debug_collect_end()
        fl.debug_collect_start(level=2, addr_level=0)
        b = tl.load(B + p)
        fl.debug_collect_end()
        tl.store(C + p, a)
        tl.store(D + p, b)

    # torch_gcu represents torch.int64 with 32-bit device storage. Transport
    # genuine i64 bits in an i32 tensor and give Triton the explicit pointer
    # type so this regression exercises an actual eight-byte payload.
    expected_a = np.array([2., 5.] if mixed else [12345678901, -9876543210],
                          dtype=np.float32 if mixed else np.int64)
    a = torch.from_numpy(
        expected_a if mixed else expected_a.view(np.int32)).to("gcu")
    b = torch.tensor([1.25, -3.5], device="gcu")
    c, d = torch.empty_like(a), torch.empty_like(b)
    debugger.activate(auto_collect=False,
                      level=global_level,
                      addr_level=0,
                      output_dir=tmp_path)
    try:
        scalar_pair[(2, )](a if mixed else triton.reinterpret(a, tl.int64), b,
                           c if mixed else triton.reinterpret(c, tl.int64), d,
                           mixed)
        torch.gcu.synchronize()
        torch.testing.assert_close(c.cpu(), a.cpu())
        torch.testing.assert_close(d.cpu(), b.cpu())
        runs = debugger.take_exported_runs()
        assert runs
        artifacts = [
            entry for run in runs
            for entry in run["runtime_metadata"]["full_dump_artifacts"]
            if entry["kind"] == "value"
        ]
        for instance in (0, 1):
            values = [
                np.load(e["path"]).item() for e in artifacts
                if e["logical_instance_id"] == instance
            ]
            assert b.cpu()[instance].item() in values
            if not mixed:
                assert expected_a[instance].item() in values
        if mixed:
            records = [r for run in runs for r in run["decoded"]["records"]]
            assert sum(r["record_kind"] == "SUMMARY_VALUE_BUNDLE_F32"
                       for r in records) == 4
            means = sorted(r["mean"] for r in records
                           if r["record_kind"] == "SUMMARY_VALUE_BUNDLE_F32")
            assert means == pytest.approx([-3.5, 1.25, 2.0, 5.0])
    finally:
        debugger.deactivate()


def test_native_decoder_accepts_unsorted_inactive_slots():
    from flagtree.debugger.native import runtime_binding
    binding = runtime_binding()
    header = struct.pack("<8I", 3, 3, 0, 0, 32, 128, 0, 0)
    records = b"".join(
        struct.pack("<HHIQHHId", 1, 0, op, 0, 6, 3, 0, float(op))
        for op in (1, 2, 3))
    run = {
        "meta": {
            "protocol_version": 1
        },
        "runtime_metadata": {
            "inactive_record_slots": [2, 0]
        },
        "raw_buffer": header + records
    }
    assert [r["op_id"]
            for r in binding.decode_exported_run(run)["records"]] == [2]


@pytest.mark.parametrize("programs", [1, 64])
def test_enflame_l1_summary_values_and_capacity(tmp_path, programs):
    import torch
    pytest.importorskip("torch_gcu")
    import triton
    import triton.language as tl
    from flagtree import debugger

    if triton.runtime.driver.active.get_current_target().backend != "gcu":
        pytest.skip("Enflame device required")

    @triton.jit
    def copy_summary(X, Y):
        i = tl.arange(0, 8)
        x = tl.load(X + i)
        tl.store(Y + tl.program_id(0) * 8 + i, x)

    x = torch.tensor(
        [float('nan'),
         float('inf'), -float('inf'), 0., 3., 4., -2., 0.],
        device="gcu")
    y = torch.empty(programs * 8, device="gcu")
    debugger.activate(auto_collect=True,
                      level=1,
                      addr_level=0,
                      record_capacity=64,
                      output_dir=tmp_path)
    try:
        copy_summary[(programs, )](x, y)
        torch.gcu.synchronize()
        runs = debugger.take_exported_runs()
        assert len(runs) == 1
        decoded = runs[0]["decoded"]
        counts = [
            r for r in decoded["records"]
            if r["record_kind"] == "SUMMARY_COUNT_BUNDLE_U64"
        ]
        values = [
            r for r in decoded["records"]
            if r["record_kind"] == "SUMMARY_VALUE_BUNDLE_F32"
        ]
        assert counts and values
        for r in counts:
            assert (r["nan_count"], r["inf_count"], r["zero_count"],
                    r["element_count"]) == (1, 2, 2, 8)
        for r in values:
            assert (r["mean"], r["min"], r["max"],
                    r["l2_norm"]) == pytest.approx((1., -2., 4., 29.**0.5))
        assert (decoded["header"]["overflow_count"] > 0) == (programs == 64)
    finally:
        debugger.deactivate()


@pytest.mark.parametrize("true_count", [0, 4, 8])
def test_enflame_boolean_float_summary_is_exact(tmp_path, true_count):
    import torch
    pytest.importorskip("torch_gcu")
    import triton
    import triton.language as tl
    from flagtree import debugger
    if triton.runtime.driver.active.get_current_target().backend != "gcu":
        pytest.skip("Enflame device required")

    @triton.jit
    def binary_summary(X, Y):
        i = tl.arange(0, 8)
        x = tl.load(X + i)
        y = (x > 0).to(tl.float32)
        tl.store(Y + i, y)

    x = torch.cat((torch.ones(true_count, dtype=torch.int32),
                   torch.zeros(8 - true_count, dtype=torch.int32))).to('gcu')
    y = torch.empty(8, device='gcu')
    debugger.activate(auto_collect=True,
                      level=1,
                      addr_level=0,
                      output_dir=tmp_path)
    try:
        binary_summary[(1, )](x, y)
        torch.gcu.synchronize()
        records = [
            r for run in debugger.take_exported_runs()
            for r in run['decoded']['records']
        ]
        counts = [
            r for r in records
            if r['record_kind'] == 'SUMMARY_COUNT_BUNDLE_U64'
        ]
        values = [
            r for r in records
            if r['record_kind'] == 'SUMMARY_VALUE_BUNDLE_F32'
        ]
        assert len(counts) == len(values) == 1
        r = counts[0]
        assert (r['nan_count'], r['inf_count'], r['zero_count'],
                r['element_count']) == (0, 0, 8 - true_count, 8)
        r = values[0]
        assert (r['mean'], r['min'], r['max'], r['l2_norm']) == pytest.approx(
            (true_count / 8, float(true_count == 8), float(true_count > 0),
             true_count**0.5))
    finally:
        debugger.deactivate()
