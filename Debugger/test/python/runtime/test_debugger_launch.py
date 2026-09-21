import importlib.util
import importlib
import json
from pathlib import Path
import re
import struct
from types import SimpleNamespace

import pytest
import torch
import triton

from flagtree.debugger import api as debugger


def _reset_debugger_state():
    debugger.deactivate()
    debugger.clear_exported_runs()
    debugger.reset_config()


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _require_backend(*expected: str) -> None:
    try:
        backend = str(
            triton.runtime.driver.active.get_current_target().backend).lower()
    except Exception as exc:
        pytest.skip(f"cannot determine active backend: {exc}")
    if backend not in expected:
        pytest.skip(
            f"launcher source test requires one of {expected}, active backend is {backend}"
        )


def test_infer_runtime_metadata_captures_tensor_views_and_storage():
    base = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    view = base[:, 1:]
    metadata = SimpleNamespace(argument_names=("base", "view", "count"))

    runtime_metadata = debugger._infer_runtime_metadata(
        metadata, (base, view, 16))

    assert len(runtime_metadata["buffers"]) == 1
    buffer = runtime_metadata["buffers"][0]
    assert buffer["buffer_id"] == 1
    assert buffer["buffer_name"] == "base"
    assert buffer["base_address"] == base.untyped_storage().data_ptr()
    assert buffer["size_bytes"] == base.untyped_storage().nbytes()
    assert buffer["alignment"] > 0

    assert runtime_metadata["tensors"] == [
        {
            "argument_index": 0,
            "logical_name": "base",
            "dtype": "float32",
            "shape": [4, 4],
            "stride": [4, 1],
            "layout": "contiguous",
            "buffer_id": 1,
            "base_address": base.data_ptr(),
            "size_bytes": 64,
        },
        {
            "argument_index": 1,
            "logical_name": "view",
            "dtype": "float32",
            "shape": [4, 3],
            "stride": [4, 1],
            "layout": "strided",
            "buffer_id": 1,
            "base_address": view.data_ptr(),
            "size_bytes": 60,
        },
    ]


def test_prepare_kernel_launch_requires_hook():
    debugger.clear_launch_prepare_hook()
    metadata = SimpleNamespace(debug_enabled=True, name="debug_kernel")
    with pytest.raises(RuntimeError, match="register_launch_prepare_hook"):
        debugger.prepare_kernel_launch(metadata, 17, None)


def test_prepare_kernel_launch_normalizes_args_and_finalizer():
    debugger.clear_launch_prepare_hook()
    calls = []

    def finalize(error):
        calls.append(error)

    def hook(metadata, stream, launch_metadata, kernel_args):
        calls.append((metadata.name, stream, launch_metadata, kernel_args))
        return debugger.PreparedKernelLaunch(kernel_args=9, finalize=finalize)

    debugger.register_launch_prepare_hook(hook)
    try:
        metadata = SimpleNamespace(debug_enabled=True, name="debug_kernel")
        prepared = debugger.prepare_kernel_launch(metadata, 23,
                                                  {"grid":
                                                   (1, 1, 1)}, ("arg0", ))
        assert prepared is not None
        assert prepared.kernel_args == (9, )

        launch_error = RuntimeError("launch failed")
        debugger.finalize_prepared_launch(prepared, launch_error)
        assert calls == [
            ("debug_kernel", 23, {
                "grid": (1, 1, 1)
            }, ("arg0", )),
            launch_error,
        ]
    finally:
        debugger.clear_launch_prepare_hook()


def test_activate_installs_default_prepare_hook(monkeypatch):
    _reset_debugger_state()

    seen = {}

    class FakeHandle:
        hidden_arg_value = 77

        def finish(self):
            seen["finished"] = True
            return {"meta": {"kernel_id": 1}, "raw_buffer": b""}

        def release(self):
            seen["released"] = True

    monkeypatch.setattr(
        debugger,
        "_load_binding",
        lambda: SimpleNamespace(
            prepare_launch=lambda metadata, stream_handle, runtime_metadata:
            (seen.update({
                "metadata": metadata,
                "stream_handle": stream_handle,
                "runtime_metadata": runtime_metadata,
            }) or FakeHandle()),
            decode_exported_run=lambda exported: {
                "header": {},
                "records": []
            },
        ),
    )

    debugger.activate(
        record_level=2,
        record_capacity=2048,
        output_dir=None,
        runtime_metadata_builder=lambda metadata, launch_metadata, kernel_args:
        {
            "buffers": [{
                "buffer_id": 1,
                "buffer_name": "x"
            }],
            "kernel_args": list(kernel_args),
            "launch_metadata": launch_metadata,
        },
    )
    try:
        metadata = SimpleNamespace(
            name="debug_kernel",
            hash="0123456789abcdef",
            backend_name="ascend",
            debug_records_per_instance=5,
            debug_tracked_table=[{
                "op_id": 7,
                "mlir_op": "tt.load"
            }],
            target=SimpleNamespace(arch="Ascend910B", backend="ascend"),
        )
        prepared = debugger.prepare_kernel_launch(metadata, 99,
                                                  {"grid": (2, 3, 4)},
                                                  ("ptr0", "ptr1"))

        assert prepared is not None
        assert prepared.kernel_args == (77, )
        assert seen["stream_handle"] == 99
        assert seen["metadata"]["debug_backend_name"] == "ascend"
        assert seen["metadata"]["debug_addr_level"] == 0
        assert seen["metadata"]["debug_record_capacity"] == 2048
        assert seen["runtime_metadata"]["buffers"][0]["buffer_name"] == "x"
        assert seen["runtime_metadata"]["kernel_args"] == ["ptr0", "ptr1"]
        assert seen["runtime_metadata"]["grid"] == (2, 3, 4)
        assert seen["runtime_metadata"]["records_per_instance"] == 5

        debugger.finalize_prepared_launch(prepared, None)
        assert seen["finished"] is True
        runs = debugger.take_exported_runs()
        assert len(runs) == 1
        assert runs[0]["meta"] == {"kernel_id": 1}
        assert runs[0]["raw_buffer"] == b""
        assert runs[0]["debug_kernel_name"] == "debug_kernel"
        assert runs[0]["debug_tracked_table"] == [{
            "op_id": 7,
            "mlir_op": "tt.load"
        }]
        assert runs[0]["decoded"] == {"header": {}, "records": []}
        runtime_metadata = runs[0]["runtime_metadata"]
        assert runtime_metadata["host_start_time_ns"] > 0
        assert runtime_metadata["host_end_time_ns"] >= runtime_metadata[
            "host_start_time_ns"]
        assert runtime_metadata["host_duration_ns"] == (
            runtime_metadata["host_end_time_ns"] -
            runtime_metadata["host_start_time_ns"])
    finally:
        _reset_debugger_state()


def test_default_prepare_hook_preserves_builder_launch_metadata(monkeypatch):
    _reset_debugger_state()

    seen = {}

    class FakeHandle:
        hidden_arg_value = 77

        def finish(self):
            return {"meta": {"kernel_id": 1}, "raw_buffer": b""}

        def release(self):
            pass

    monkeypatch.setattr(
        debugger,
        "_load_binding",
        lambda: SimpleNamespace(
            prepare_launch=lambda metadata, stream_handle, runtime_metadata:
            (seen.update({"runtime_metadata": runtime_metadata}) or FakeHandle(
            )),
            decode_exported_run=lambda exported: {
                "header": {},
                "records": []
            },
        ),
    )

    debugger.activate(
        output_dir=None,
        runtime_metadata_builder=lambda metadata, launch_metadata, kernel_args:
        {
            "grid": (9, 9, 9),
            "records_per_instance": 7,
        },
    )
    try:
        metadata = SimpleNamespace(
            name="debug_kernel",
            debug_records_per_instance=5,
        )
        prepared = debugger.prepare_kernel_launch(metadata, 99,
                                                  {"grid": (2, 3, 4)}, ())

        assert prepared is not None
        assert seen["runtime_metadata"]["grid"] == (9, 9, 9)
        assert seen["runtime_metadata"]["records_per_instance"] == 7
    finally:
        _reset_debugger_state()


def test_configure_supplies_defaults_for_enable_debug(tmp_path, monkeypatch):
    _reset_debugger_state()

    seen = {}

    class FakeHandle:
        hidden_arg_value = 77

        def finish(self):
            return {
                "meta": {
                    "run_id": 4,
                    "kernel_id": 17
                },
                "raw_buffer": b"abcd",
            }

        def release(self):
            pass

    monkeypatch.setattr(debugger.sys, "argv",
                        ["/work/tests/test_debugger_output.py"])
    monkeypatch.setattr(
        debugger,
        "_load_binding",
        lambda: SimpleNamespace(
            prepare_launch=lambda metadata, stream_handle, runtime_metadata:
            (seen.update({"metadata": metadata}) or FakeHandle()),
            decode_exported_run=lambda exported: {
                "meta": exported["meta"],
                "header": {
                    "write_idx": 1,
                    "capacity": 4096
                },
                "records": [{
                    "record_kind": "SUMMARY",
                    "op_id": 1
                }],
            },
            render_text_report=lambda exported, metadata_json:
            "rendered report text",
        ),
    )

    debugger.configure(
        output_dir=tmp_path,
        record_capacity=4096,
        export_mode="streaming",
        export_on_error=True,
        export_raw_records=True,
    )
    debugger.activate(level=2, addr_level=1)
    try:
        assert debugger.current_compile_config() == {
            "debug_enabled": True,
            "debug_protocol_version": 2,
            "debug_record_level": 2,
            "debug_addr_level": 1,
            "debug_export_mode": "STREAMING_EXPORT",
            "debug_record_capacity": 4096,
            "debug_timeline_enabled": False,
        }

        metadata = SimpleNamespace(
            name="debug_kernel",
            debug_kernel_id=17,
            debug_metadata_json='{"debugKernelId": 17}',
        )
        prepared = debugger.prepare_kernel_launch(metadata, 99, None, ())
        debugger.finalize_prepared_launch(prepared, None)

        assert seen["metadata"]["debug_record_level"] == 2
        assert seen["metadata"]["debug_addr_level"] == 1
        assert seen["metadata"]["debug_record_capacity"] == 4096
        assert seen["metadata"]["debug_export_mode"] == "STREAMING_EXPORT"

        run = debugger.take_exported_runs()[0]
        assert Path(run["report_path"]).parent == tmp_path
        assert Path(run["raw_records_path"]).name.endswith("_raw_records.txt")
    finally:
        debugger.deactivate()

    assert debugger.get_config()["record_capacity"] == 4096
    assert debugger.get_config()["export_raw_records"] is True
    _reset_debugger_state()


def test_configure_rejects_unknown_keys_and_invalid_capacity():
    _reset_debugger_state()
    try:
        with pytest.raises(TypeError, match="unknown debugger config key"):
            debugger.configure(debugger_output_dir="/tmp/wrong-key")
        with pytest.raises(ValueError, match="record capacity"):
            debugger.configure(record_capacity=0)
    finally:
        _reset_debugger_state()


def test_timeline_config_is_exposed_to_compiler():
    # FlagPrism: timeline is opt-in and must reach the compiler configuration
    # without changing the default debugger record level or address level.
    _reset_debugger_state()
    try:
        debugger.configure(timeline=True)
        debugger.activate(level=1)
        assert debugger.get_config()["timeline"] is True
        assert debugger.current_compile_config(
        )["debug_timeline_enabled"] is True
    finally:
        _reset_debugger_state()


def test_enable_debug_rejects_invalid_addr_level():
    _reset_debugger_state()
    try:
        with pytest.raises(ValueError, match="addr_level"):
            debugger.activate(addr_level=3)
        with pytest.raises(ValueError, match="addr_level"):
            debugger.activate(addr_level=-1)
    finally:
        _reset_debugger_state()


@pytest.mark.parametrize("keyword", ["level", "record_level"])
@pytest.mark.parametrize("value", [-1, 0, 3])
def test_enable_debug_rejects_invalid_record_level(keyword, value):
    _reset_debugger_state()
    try:
        with pytest.raises(ValueError, match="level must be 1 or 2"):
            debugger.activate(**{keyword: value})
    finally:
        _reset_debugger_state()


def test_default_prepare_hook_uses_flagtree_backend(monkeypatch):
    _reset_debugger_state()

    seen = {}

    class FakeHandle:
        hidden_arg_value = 77

        def finish(self):
            return {"meta": {"kernel_id": 1}, "raw_buffer": b""}

        def release(self):
            pass

    monkeypatch.setenv("FLAGTREE_BACKEND", "ascend")
    monkeypatch.setattr(
        debugger, "_load_binding", lambda: SimpleNamespace(
            prepare_launch=lambda metadata, stream_handle, runtime_metadata:
            (seen.update({"metadata": metadata}) or FakeHandle())))

    debugger.activate(output_dir=None)
    try:
        metadata = SimpleNamespace(
            name="debug_kernel",
            hash="0123456789abcdef",
            target=SimpleNamespace(arch="Ascend910B4", backend="npu"),
        )
        prepared = debugger.prepare_kernel_launch(metadata, 99, None, ())

        assert prepared is not None
        assert seen["metadata"]["debug_backend_name"] == "ascend"
        assert seen["metadata"]["debug_target_name"] == "Ascend910B4"
    finally:
        _reset_debugger_state()


def test_metadata_only_launch_writes_empty_report(tmp_path, monkeypatch):
    _reset_debugger_state()

    seen = {}

    def decode_exported_run(exported):
        seen["raw_buffer"] = bytes(exported["raw_buffer"])
        return {
            "meta": exported["meta"],
            "header": {
                "write_idx": 0,
                "capacity": 0,
                "overflow_count": 0,
                "flags": 0,
                "record_size": 64,
            },
            "records": [],
        }

    monkeypatch.setattr(debugger.sys, "argv",
                        ["/work/tests/test_metadata_only.py"])
    monkeypatch.setattr(
        debugger,
        "_load_binding",
        lambda: SimpleNamespace(
            decode_exported_run=decode_exported_run,
            render_text_statement_report=lambda exported, metadata_json:
            "metadata only",
            render_json_statement_report=lambda exported, metadata_json:
            '{"records_by_op":[]}',
        ),
    )

    debugger.configure(output_dir=tmp_path)
    debugger.activate(level=1)
    try:
        metadata = SimpleNamespace(
            name="metadata_only_kernel",
            debug_enabled=True,
            debug_kernel_id=23,
            debug_metadata_json='{"debugKernelId": 23}',
            debug_records_per_instance=0,
            debug_record_size=64,
            debug_record_plan=[],
            target=SimpleNamespace(arch="Ascend910B", backend="ascend"),
        )
        prepared = debugger.prepare_metadata_only_kernel_launch(
            metadata, 99, {"grid": (2, 3, 4)}, ("x", ))
        assert prepared is not None
        assert prepared.kernel_args == ()

        debugger.finalize_prepared_launch(prepared, None)

        assert len(seen["raw_buffer"]) == 32
        runs = debugger.take_exported_runs()
        assert len(runs) == 1
        run = runs[0]
        assert run["runtime_metadata"]["grid"] == (2, 3, 4)
        assert run["runtime_metadata"]["records_per_instance"] == 0
        report_path = Path(run["report_path"])
        assert report_path.exists()
        assert "metadata only" in report_path.read_text()
        assert Path(
            run["json_report_path"]).read_text() == '{"records_by_op":[]}'
    finally:
        _reset_debugger_state()


def test_default_prepare_hook_writes_timestamped_report(tmp_path, monkeypatch):
    _reset_debugger_state()

    seen = {}

    class FakeHandle:
        hidden_arg_value = 77

        def finish(self):
            return {
                "meta": {
                    "run_id": 9,
                    "kernel_id": 17
                },
                "runtime_metadata": {
                    "buffers": [],
                    "tensors": []
                },
                "raw_buffer": b"abcd",
            }

        def release(self):
            pass

    def prepare_launch(metadata, stream_handle, runtime_metadata):
        seen["metadata"] = metadata
        seen["stream_handle"] = stream_handle
        seen["runtime_metadata"] = runtime_metadata
        return FakeHandle()

    monkeypatch.setattr(debugger.sys, "argv",
                        ["/work/tests/test_debugger_output.py"])
    monkeypatch.setattr(
        debugger,
        "_load_binding",
        lambda: SimpleNamespace(
            prepare_launch=prepare_launch,
            decode_exported_run=lambda exported: {
                "meta": exported["meta"],
                "header": {
                    "write_idx": 1,
                    "capacity": 64
                },
                "records": [{
                    "record_kind": "SUMMARY",
                    "op_id": 1
                }],
            },
            render_text_report=lambda exported, metadata_json:
            "rendered report text",
            render_json_report=lambda exported, metadata_json:
            '{"records_by_op":[]}',
        ),
    )

    debugger.configure(output_dir=tmp_path)
    debugger.activate()
    try:
        metadata = SimpleNamespace(
            name="debug_kernel",
            debug_kernel_id=17,
            debug_metadata_json='{"debugKernelId": 17}',
        )
        prepared = debugger.prepare_kernel_launch(metadata, 99, None, ())
        debugger.finalize_prepared_launch(prepared, None)

        runs = debugger.take_exported_runs()
        assert len(runs) == 1
        run = runs[0]
        report_path = Path(run["report_path"])
        assert report_path.parent == tmp_path
        assert report_path.exists()
        json_report_path = Path(run["json_report_path"])
        assert json_report_path.parent == tmp_path
        assert json_report_path.suffix == ".json"
        assert json_report_path.stem == report_path.stem
        assert json_report_path.read_text() == '{"records_by_op":[]}'
        assert re.match(
            r"test_debugger_output_debug_kernel_\d{8}_\d{6}_\d{3}_run9\.txt",
            report_path.name,
        )
        text = report_path.read_text()
        assert "FlagPrism Debug Export" in text
        assert "kernel_name: debug_kernel" in text
        assert "raw_buffer_bytes: 4" in text
        assert "Decoded Header" in text
        assert "Decoded Records" not in text
        assert "rendered report text" in text
        assert "raw_records_path" not in run
        assert run["decoded"]["header"]["write_idx"] == 1
        assert run["report"] == "rendered report text"
        assert run["json_report"] == '{"records_by_op":[]}'
    finally:
        _reset_debugger_state()


def test_configure_output_dir_none_disables_file_export(tmp_path, monkeypatch):
    _reset_debugger_state()

    class FakeHandle:
        hidden_arg_value = 77

        def finish(self):
            return {"meta": {"run_id": 1, "kernel_id": 1}, "raw_buffer": b""}

        def release(self):
            pass

    monkeypatch.setattr(
        debugger,
        "_load_binding",
        lambda: SimpleNamespace(
            prepare_launch=lambda metadata, stream_handle, runtime_metadata:
            FakeHandle(),
            decode_exported_run=lambda exported: {
                "header": {},
                "records": []
            },
        ),
    )

    debugger.configure(output_dir=tmp_path)
    assert debugger.get_output_dir() == str(tmp_path)
    debugger.configure(output_dir=None)
    assert debugger.get_output_dir() is None

    debugger.activate()
    try:
        metadata = SimpleNamespace(name="debug_kernel", debug_kernel_id=1)
        prepared = debugger.prepare_kernel_launch(metadata, 99, None, ())
        debugger.finalize_prepared_launch(prepared, None)
        runs = debugger.take_exported_runs()
        assert len(runs) == 1
        assert "report_path" not in runs[0]
        assert list(tmp_path.iterdir()) == []
    finally:
        _reset_debugger_state()


def test_level2_full_dump_writes_npy_artifacts_and_reports_paths(
        tmp_path, monkeypatch):
    _reset_debugger_state()

    payload = struct.pack("<ff", 1.5, 2.5)
    header = struct.pack("<IIIIIIII", 1, 1, 0, 0, 64, 96, 0, 0)
    record = bytearray(64)
    struct.pack_into("<II", record, 16, 96, len(payload))
    raw_buffer = header + bytes(record) + payload
    full_dump_plan = [{
        "record_index": 0,
        "op_id": 7,
        "scope_id": 1,
        "kind": "value",
        "source": "result",
        "artifact_dtype": "float32",
        "shape": [2],
        "element_count": 2,
        "element_bytes": 4,
        "payload_offset": 0,
        "payload_length": len(payload),
    }]
    seen = {}

    class FakeHandle:
        hidden_arg_value = 77

        def __init__(self, runtime_metadata):
            self.runtime_metadata = runtime_metadata

        def finish(self):
            return {
                "meta": {
                    "run_id": 5,
                    "kernel_id": 17
                },
                "runtime_metadata": dict(self.runtime_metadata),
                "raw_buffer": raw_buffer,
            }

        def release(self):
            seen["released"] = True

    def prepare_launch(metadata, stream_handle, runtime_metadata):
        seen["metadata"] = metadata
        seen["runtime_metadata"] = runtime_metadata
        return FakeHandle(runtime_metadata)

    def render_text_report(exported, metadata_json):
        artifacts = exported["runtime_metadata"]["full_dump_artifacts"]
        assert len(artifacts) == 1
        return f"kind={artifacts[0]['kind']} path={artifacts[0]['path']}"

    monkeypatch.setattr(debugger.sys, "argv",
                        ["/work/tests/test_debugger_output.py"])
    monkeypatch.setattr(
        debugger,
        "_load_binding",
        lambda: SimpleNamespace(
            prepare_launch=prepare_launch,
            decode_exported_run=lambda exported: {
                "meta":
                exported["meta"],
                "header": {
                    "write_idx": 1,
                    "capacity": 1,
                    "overflow_count": 0,
                    "flags": 0,
                },
                "records": [{
                    "record_kind": "FULL_VALUE",
                    "op_id": 7,
                    "logical_instance_id": 0,
                    "payload_offset": 96,
                    "payload_length": len(payload),
                }],
            },
            render_text_report=render_text_report,
            render_json_report=lambda exported, metadata_json:
            '{"records_by_op":[]}',
        ),
    )

    debugger.configure(output_dir=tmp_path)
    debugger.activate(level=2, record_capacity=1)
    try:
        metadata = SimpleNamespace(
            name="debug_kernel",
            debug_kernel_id=17,
            debug_metadata_json='{"debugKernelId": 17}',
            debug_full_dump_payload_bytes_per_instance=len(payload),
            debug_full_dump_plan=full_dump_plan,
            debug_records_per_instance=1,
        )
        prepared = debugger.prepare_kernel_launch(metadata, 99,
                                                  {"grid": (1, )}, ())
        assert prepared is not None
        assert seen["runtime_metadata"]["full_dump_plan"] == full_dump_plan

        debugger.finalize_prepared_launch(prepared, None)

        run = debugger.take_exported_runs()[0]
        artifacts = run["runtime_metadata"]["full_dump_artifacts"]
        assert len(artifacts) == 1
        artifact_path = Path(artifacts[0]["path"])
        assert artifact_path.exists()
        npy_bytes = artifact_path.read_bytes()
        assert npy_bytes.startswith(b"\x93NUMPY\x01\x00")
        assert npy_bytes.endswith(payload)
        assert Path(run["full_dump_index_path"]).exists()
        report_text = Path(run["report_path"]).read_text()
        assert f"path={artifact_path}" in report_text
        assert "payload_offset" not in report_text
    finally:
        _reset_debugger_state()


def test_precision_diagnostics_compare_direct_float_conversion(tmp_path):
    numpy = pytest.importorskip("numpy")
    before = numpy.array([0.1, 1.1, 3.14159, 1000.3], dtype=numpy.float32)
    after = before.astype(numpy.float16).astype(numpy.float32)
    before_path = tmp_path / "before.npy"
    after_path = tmp_path / "after.npy"
    numpy.save(before_path, before)
    numpy.save(after_path, after)

    metadata = {
        "debug_tracked_table": [{
            "opId":
            2,
            "statementId":
            17,
            "sourceLoc":
            "test.py:17",
            "tritonStatement":
            "narrowed = value.to(tl.float16)",
            "mlirOpName":
            "arith.truncf",
            "result": {
                "elementDtype": "f16"
            },
            "operands": [{
                "producerOpId": 1,
                "value": {
                    "elementDtype": "f32"
                },
            }],
        }]
    }
    artifacts = [{
        "op_id": 1,
        "logical_instance_id": 0,
        "kind": "value",
        "path": str(before_path),
    }, {
        "op_id": 2,
        "logical_instance_id": 0,
        "kind": "value",
        "path": str(after_path),
    }]

    diagnostics = debugger._build_precision_diagnostics(metadata, artifacts)

    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic["from"]["dtype"] == "f32"
    assert diagnostic["to"]["dtype"] == "f16"
    assert diagnostic["status"] == "ok"
    assert diagnostic["suspicious_lane_count"] == 0
    assert diagnostic["conversion_loss"] == "lossy"
    assert diagnostic["changed_lane_count"] == 4
    assert diagnostic["max_abs_error"] > 0
    assert diagnostic["worst_lane"]["index"] == 3

    report = debugger._inject_precision_diagnostics(
        json.dumps({"records_by_op": [{
            "statement_id": 17
        }]}), diagnostics)
    document = json.loads(report)
    assert document["records_by_op"][0]["precision_conversion"] == diagnostics
    assert "precision_conversion:" in debugger._render_precision_diagnostics(
        diagnostics)


def test_level2_full_dump_requires_output_dir_before_launch(monkeypatch):
    _reset_debugger_state()

    def prepare_launch(metadata, stream_handle, runtime_metadata):
        raise AssertionError(
            "prepare_launch should not run without output_dir")

    monkeypatch.setattr(
        debugger,
        "_load_binding",
        lambda: SimpleNamespace(prepare_launch=prepare_launch),
    )

    debugger.configure(output_dir=None)
    debugger.activate(level=2, record_capacity=1)
    try:
        metadata = SimpleNamespace(
            name="debug_kernel",
            debug_full_dump_payload_bytes_per_instance=8,
            debug_full_dump_plan=[{
                "record_index": 0,
                "op_id": 7,
                "kind": "value",
                "source": "result",
                "artifact_dtype": "float32",
                "shape": [2],
                "payload_offset": 0,
                "payload_length": 8,
            }],
            debug_records_per_instance=1,
        )
        with pytest.raises(RuntimeError,
                           match="full dump requires debugger output_dir"):
            debugger.prepare_kernel_launch(metadata, 99, {"grid": (1, )}, ())
    finally:
        _reset_debugger_state()


def test_default_prepare_hook_writes_raw_records_sidecar_when_enabled(
        tmp_path, monkeypatch):
    _reset_debugger_state()

    class FakeHandle:
        hidden_arg_value = 77

        def finish(self):
            return {
                "meta": {
                    "run_id": 3,
                    "kernel_id": 17
                },
                "runtime_metadata": {
                    "buffers": [],
                    "tensors": []
                },
                "raw_buffer": b"abcd",
            }

        def release(self):
            pass

    monkeypatch.setattr(debugger.sys, "argv",
                        ["/work/tests/test_debugger_output.py"])
    monkeypatch.setattr(
        debugger,
        "_load_binding",
        lambda: SimpleNamespace(
            prepare_launch=lambda metadata, stream_handle, runtime_metadata:
            FakeHandle(),
            decode_exported_run=lambda exported: {
                "meta": exported["meta"],
                "header": {
                    "write_idx": 1,
                    "capacity": 64
                },
                "records": [{
                    "record_kind": "SUMMARY",
                    "op_id": 1
                }],
            },
            render_text_report=lambda exported, metadata_json:
            "rendered report text",
        ),
    )

    debugger.configure(output_dir=tmp_path, export_raw_records=True)
    debugger.activate()
    try:
        metadata = SimpleNamespace(
            name="debug_kernel",
            debug_kernel_id=17,
            debug_metadata_json='{"debugKernelId": 17}',
        )
        prepared = debugger.prepare_kernel_launch(metadata, 99, None, ())
        debugger.finalize_prepared_launch(prepared, None)

        run = debugger.take_exported_runs()[0]
        report_text = Path(run["report_path"]).read_text()
        raw_records_path = Path(run["raw_records_path"])
        assert raw_records_path.parent == tmp_path
        assert raw_records_path.name.endswith("_raw_records.txt")
        assert "Decoded Records" not in report_text
        raw_text = raw_records_path.read_text()
        assert "FlagTree Debug Raw Records" in raw_text
        assert "Decoded Records" in raw_text
        assert "'op_id': 1" in raw_text
    finally:
        _reset_debugger_state()


def test_cuda_launcher_keeps_standard_scratch_abi():
    _require_backend("cuda", "nvidia")
    if not hasattr(triton, "knobs"):
        pytest.skip(
            "CUDA launcher source requires triton.knobs, unavailable in this backend package"
        )
    # FlagPrism: load the backend through Triton's registry/package so this
    # test is independent of the checkout location and user home directory.
    module = importlib.import_module("triton.backends.nvidia.driver")
    src = module.make_launcher({}, {0: "*fp32"}, None)

    params_line = "void *params[] = { &arg0, &global_scratch, &profile_scratch };"
    assert "debug_hidden_arg" not in src
    assert params_line in src


def test_hip_launcher_keeps_standard_scratch_abi():
    _require_backend("amd", "hip")
    if not hasattr(triton, "knobs"):
        pytest.skip(
            "HIP launcher source requires triton.knobs, unavailable in this backend package"
        )
    # FlagPrism: use the registered backend module instead of constructing a
    # path into a particular FlagTree checkout.
    module = importlib.import_module("triton.backends.amd.driver")
    src = module.make_launcher({}, {0: "*fp32"}, 64)

    params_line = "void *params[] = { &arg0, &global_scratch, &profile_scratch };"
    assert "debug_hidden_arg" not in src
    assert params_line in src
