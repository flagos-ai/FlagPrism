from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from datetime import datetime
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import pprint
import re
import sys
import struct
import time
from typing import Any, Callable, Mapping, Optional, Sequence


@dataclass(frozen=True)
class PreparedKernelLaunch:
    kernel_args: tuple[int, ...] = ()
    finalize: Optional[Callable[[Optional[BaseException]], None]] = None


@dataclass(frozen=True)
class DebuggerConfig:
    enabled: bool = False
    record_level: int = 1
    addr_level: int = 0
    timeline_enabled: bool = False
    export_mode: str = "POST_KERNEL_EXPORT"
    record_capacity: int = 1024
    export_on_error: bool = False
    output_dir: str | None = "/tmp/flagtree_debugger_manual"
    export_raw_records: bool = False
    runtime_metadata_builder: Optional[Callable[[Any, Any, Sequence[Any]],
                                                Any]] = None
    export_handler: Optional[Callable[[dict[str, Any]], None]] = None


_DEFAULT_OUTPUT_DIR = Path("/tmp/flagtree_debugger_manual")
_DEFAULT_RECORD_CAPACITY = 1024
_DEFAULT_ADDR_LEVEL = 0
_DEFAULT_EXPORT_MODE = "POST_KERNEL_EXPORT"
_DEFAULT_EXPORT_ON_ERROR = False
_USE_CURRENT_CONFIG = object()
_USE_CURRENT_OUTPUT_DIR = _USE_CURRENT_CONFIG
_launch_prepare_hook = None
_output_dir: Path | None = _DEFAULT_OUTPUT_DIR
_record_capacity = _DEFAULT_RECORD_CAPACITY
_export_mode = _DEFAULT_EXPORT_MODE
_export_on_error = _DEFAULT_EXPORT_ON_ERROR
_raw_record_export_enabled = False
_timeline_enabled = False
_active_config = DebuggerConfig()
_exported_runs: list[dict[str, Any]] = []
_CONFIG_KEYS = frozenset({
    "output_dir",
    "record_capacity",
    "export_mode",
    "export_on_error",
    "export_raw_records",
    "timeline",
})
_DISABLED_BUILD_MESSAGE = (
    "FlagPrism debugger native support is unavailable. Reinstall FlagTree with "
    "`TRITON_BUILD_FLAGPRISM=ON`.")


def _normalize_kernel_args(kernel_args: Any) -> tuple[int, ...]:
    if kernel_args is None:
        return ()
    if isinstance(kernel_args, int):
        return (int(kernel_args), )
    if isinstance(kernel_args, Sequence):
        return tuple(int(arg) for arg in kernel_args)
    raise TypeError(
        "debugger launch hook must return an int, a sequence of ints, or "
        "PreparedKernelLaunch")


def _wrap_launch_prepare_hook(hook: Callable[..., Any]) -> Callable[..., Any]:
    signature = inspect.signature(hook)
    positional = [
        parameter for parameter in signature.parameters.values()
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    accepts_varargs = any(parameter.kind == inspect.Parameter.VAR_POSITIONAL
                          for parameter in signature.parameters.values())
    if accepts_varargs or len(positional) >= 4:
        return hook
    if len(positional) == 3:
        return lambda metadata, stream, launch_metadata, kernel_args: hook(
            metadata, stream, launch_metadata)
    raise TypeError("debugger launch hook must accept "
                    "(metadata, stream, launch_metadata[, kernel_args])")


def _load_binding():
    from .native import runtime_binding

    binding = runtime_binding()
    if binding is None:
        raise RuntimeError(_DISABLED_BUILD_MESSAGE)
    return binding


def is_available() -> bool:
    from .native import compiler_binding, runtime_binding

    return runtime_binding() is not None and compiler_binding() is not None


def _require_available() -> None:
    if not is_available():
        raise RuntimeError(_DISABLED_BUILD_MESSAGE)


def _normalize_output_dir(path: Any) -> Path | None:
    if path is None:
        return None
    return Path(os.fspath(path)).expanduser()


def configure(config: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
    """Update debugger defaults used by the next ``activate(level=...)``.

    Supported keys are ``output_dir``, ``record_capacity``, ``export_mode``,
    ``export_on_error``, ``export_raw_records``, and ``timeline``. Keys not
    provided keep their current values.
    """
    global _output_dir, _record_capacity, _export_mode
    global _export_on_error, _raw_record_export_enabled, _timeline_enabled

    updates = {}
    if config is not None:
        updates.update(dict(config))
    updates.update(kwargs)

    unknown = sorted(set(updates) - _CONFIG_KEYS)
    if unknown:
        raise TypeError(
            f"unknown debugger config key(s): {', '.join(unknown)}")

    if "output_dir" in updates:
        _output_dir = _normalize_output_dir(updates["output_dir"])
    if "record_capacity" in updates:
        capacity = int(updates["record_capacity"])
        if capacity <= 0:
            raise ValueError("debugger record capacity must be positive")
        _record_capacity = capacity
    if "export_mode" in updates:
        _export_mode = _normalize_export_mode(updates["export_mode"])
    if "export_on_error" in updates:
        _export_on_error = bool(updates["export_on_error"])
    if "export_raw_records" in updates:
        _raw_record_export_enabled = bool(updates["export_raw_records"])
    if "timeline" in updates:
        _timeline_enabled = bool(updates["timeline"])


def reset_config() -> None:
    """Restore debugger defaults used by ``activate(level=...)``."""
    configure(
        output_dir=_DEFAULT_OUTPUT_DIR,
        record_capacity=_DEFAULT_RECORD_CAPACITY,
        export_mode=_DEFAULT_EXPORT_MODE,
        export_on_error=_DEFAULT_EXPORT_ON_ERROR,
        export_raw_records=False,
        timeline=False,
    )


def get_config() -> dict[str, Any]:
    """Return the current debugger defaults."""
    return {
        "output_dir": get_output_dir(),
        "record_capacity": int(_record_capacity),
        "export_mode": str(_export_mode),
        "export_on_error": bool(_export_on_error),
        "export_raw_records": bool(_raw_record_export_enabled),
        "timeline": bool(_timeline_enabled),
    }


def set_output_dir(path: str | os.PathLike | None) -> None:
    """Compatibility wrapper for ``configure(output_dir=...)``."""
    configure(output_dir=path)


def get_output_dir() -> str | None:
    """Return the current automatic report export directory, or ``None``."""
    if _output_dir is None:
        return None
    return str(_output_dir)


def _normalize_export_mode(export_mode: str | int) -> str:
    if isinstance(export_mode, int):
        return "STREAMING_EXPORT" if export_mode == 2 else "POST_KERNEL_EXPORT"
    normalized = str(export_mode).strip().replace("-", "_").upper()
    if normalized in {"STREAMING", "STREAMING_EXPORT"}:
        return "STREAMING_EXPORT"
    return "POST_KERNEL_EXPORT"


def _normalize_addr_level(addr_level: int) -> int:
    value = int(addr_level)
    if value < 0 or value > 2:
        raise ValueError("debugger addr_level must be 0, 1, or 2")
    return value


def _normalize_record_level(record_level: int) -> int:
    value = int(record_level)
    if value not in (1, 2):
        raise ValueError("debugger level must be 1 or 2")
    return value


def _derive_kernel_id(metadata_dict: dict[str, Any]) -> int:
    kernel_hash = metadata_dict.get("hash")
    if isinstance(kernel_hash, str) and kernel_hash:
        kernel_id = int(kernel_hash[:8], 16)
        return kernel_id or 1

    digest = hashlib.sha256(
        repr(sorted(metadata_dict.items())).encode("utf-8")).hexdigest()
    kernel_id = int(digest[:8], 16)
    return kernel_id or 1


def _target_to_name(target: Any) -> str:
    if isinstance(target, dict):
        arch = target.get("arch")
        if arch is not None:
            return str(arch)
        backend = target.get("backend")
        if backend is not None:
            return str(backend)
        return ""
    arch = getattr(target, "arch", None)
    if arch is not None:
        return str(arch)
    backend = getattr(target, "backend", None)
    if backend is not None:
        return str(backend)
    return ""


def _target_backend(target: Any) -> str:
    if isinstance(target, dict):
        backend = target.get("backend")
        return "" if backend is None else str(backend)
    backend = getattr(target, "backend", None)
    return "" if backend is None else str(backend)


def _normalize_backend_name(backend_name: Any) -> str:
    name = "" if backend_name is None else str(backend_name)
    return name


def _safe_filename_component(value: Any, fallback: str) -> str:
    text = "" if value is None else str(value)
    text = text.strip() or fallback
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    safe = safe.strip("._-")
    return safe or fallback


def _current_script_stem() -> str:
    script = sys.argv[0] if sys.argv else ""
    if not script:
        return "interactive"
    stem = Path(script).stem
    return _safe_filename_component(stem, "interactive")


def _exported_run_meta(exported_run: dict[str, Any]) -> dict[str, Any]:
    meta = exported_run.get("meta")
    return meta if isinstance(meta, dict) else {}


def _render_export_summary(exported_run: dict[str, Any],
                           decoded: dict[str, Any] | None,
                           metadata_dict: dict[str, Any]) -> str:
    meta = _exported_run_meta(exported_run)
    raw_buffer = exported_run.get("raw_buffer", b"")
    raw_size = len(raw_buffer) if hasattr(raw_buffer, "__len__") else 0
    lines = [
        "FlagPrism Debug Export",
        f"kernel_name: {metadata_dict.get('debug_kernel_name') or metadata_dict.get('name') or '<unknown>'}",
        f"kernel_id: {meta.get('kernel_id', metadata_dict.get('debug_kernel_id', 0))}",
        f"run_id: {meta.get('run_id', '<unknown>')}",
        f"backend: {metadata_dict.get('debug_backend_name', '')}",
        f"target: {metadata_dict.get('debug_target_name', '')}",
        f"raw_buffer_bytes: {raw_size}",
    ]
    if decoded is not None:
        lines.extend([
            "",
            "Decoded Header",
            pprint.pformat(decoded.get("header", {}), sort_dicts=True),
        ])
    return "\n".join(lines)


def _build_report_path(output_dir: Path, exported_run: dict[str, Any],
                       metadata_dict: dict[str, Any]) -> Path:
    meta = _exported_run_meta(exported_run)
    script_stem = _current_script_stem()
    kernel_name = _safe_filename_component(
        metadata_dict.get("debug_kernel_name") or metadata_dict.get("name"),
        "kernel",
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    run_id = _safe_filename_component(meta.get("run_id", "0"), "0")
    return output_dir / f"{script_stem}_{kernel_name}_{timestamp}_run{run_id}.txt"


def _render_raw_records(exported_run: dict[str, Any], decoded: dict[str, Any],
                        metadata_dict: dict[str, Any]) -> str:
    meta = _exported_run_meta(exported_run)
    lines = [
        "FlagTree Debug Raw Records",
        f"kernel_name: {metadata_dict.get('debug_kernel_name') or metadata_dict.get('name') or '<unknown>'}",
        f"kernel_id: {meta.get('kernel_id', metadata_dict.get('debug_kernel_id', 0))}",
        f"run_id: {meta.get('run_id', '<unknown>')}",
        "",
        "Decoded Header",
        pprint.pformat(decoded.get("header", {}), sort_dicts=True),
        "",
        "Decoded Records",
        pprint.pformat(decoded.get("records", []), sort_dicts=True),
    ]
    return "\n".join(lines)


def _is_full_dump_run(metadata_dict: dict[str, Any]) -> bool:
    return (int(
        metadata_dict.get("debug_full_dump_payload_bytes_per_instance", 0)) > 0
            and bool(metadata_dict.get("debug_full_dump_plan")))


def _record_level_id(value: Any) -> int:
    if isinstance(value, int):
        return 2 if value == 2 else 1
    text = str(value)
    return 2 if text in {"2", "LEVEL_TENSOR_FULL"} else 1


def _export_mode_id(value: Any) -> int:
    if isinstance(value, int):
        return 2 if value == 2 else 1
    text = str(value)
    return 2 if text == "STREAMING_EXPORT" else 1


def _empty_debug_raw_buffer(metadata_dict: Mapping[str, Any]) -> bytes:
    record_size = int(metadata_dict.get("debug_record_size", 64) or 64)
    fields = (
        0,  # writeIdx
        0,  # capacity
        0,  # overflowCount
        0,  # flags
        record_size,
        32,  # payloadOffset
        0,
        0,
    )
    return b"".join(
        int(field).to_bytes(4, "little", signed=False) for field in fields)


def _metadata_only_exported_run(
        metadata_dict: dict[str, Any],
        runtime_metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "meta": {
            "run_id":
            len(_exported_runs) + 1,
            "device_id":
            int(metadata_dict.get("debug_device_id", 0) or 0),
            "kernel_id":
            int(
                metadata_dict.get("debug_kernel_id",
                                  _derive_kernel_id(metadata_dict))),
            "protocol_version":
            int(metadata_dict.get("debug_protocol_version", 2) or 2),
            "record_level":
            _record_level_id(metadata_dict.get("debug_record_level", 1)),
            "export_mode":
            _export_mode_id(
                metadata_dict.get("debug_export_mode",
                                  _active_config.export_mode)),
            "backend_kind":
            0,
        },
        "runtime_metadata": dict(runtime_metadata),
        "raw_buffer": _empty_debug_raw_buffer(metadata_dict),
    }


def _npy_dtype_descriptor(dtype: str) -> str:
    if dtype == "float32":
        return "<f4"
    if dtype == "float64":
        return "<f8"
    if dtype == "int64":
        return "<i8"
    if dtype == "uint64":
        return "<u8"
    raise ValueError(f"unsupported debugger artifact dtype: {dtype}")


def _npy_dtype_element_bytes(dtype: str) -> int:
    if dtype in {"float32"}:
        return 4
    if dtype in {"float64", "int64", "uint64"}:
        return 8
    raise ValueError(f"unsupported debugger artifact dtype: {dtype}")


def _write_npy(path: Path, payload: bytes, dtype: str,
               shape: Sequence[int]) -> None:
    descr = _npy_dtype_descriptor(dtype)
    dims = tuple(int(dim) for dim in shape)
    if any(dim < 0 for dim in dims):
        raise ValueError(
            "debugger artifact shape cannot contain negative dimensions")
    element_count = 1
    for dim in dims:
        element_count *= dim
    expected_bytes = element_count * _npy_dtype_element_bytes(dtype)
    if len(payload) != expected_bytes:
        raise RuntimeError(
            "debugger artifact payload size does not match planned dtype/shape"
        )
    if len(dims) == 1:
        shape_repr = f"({dims[0]},)"
    else:
        shape_repr = "(" + ", ".join(str(dim) for dim in dims) + ")"
    header = f"{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_repr}, }}"
    header_bytes = header.encode("latin1")
    padding = (16 - ((10 + len(header_bytes) + 1) % 16)) % 16
    header_bytes = header_bytes + b" " * padding + b"\n"
    if len(header_bytes) > 0xFFFF:
        raise ValueError("debugger artifact .npy header is too large")
    path.write_bytes(b"\x93NUMPY\x01\x00" +
                     len(header_bytes).to_bytes(2, "little") + header_bytes +
                     payload)


def _full_dump_plan_by_record(
        metadata_dict: dict[str, Any]) -> dict[int, dict[str, Any]]:
    plan = metadata_dict.get("debug_full_dump_plan") or []
    result = {}
    for entry in plan:
        if not isinstance(entry, Mapping):
            continue
        result[int(entry.get("record_index", 0))] = dict(entry)
    return result


def _record_index_for_slot(slot_index: int,
                           runtime_metadata: Mapping[str, Any]) -> int:
    records_per_instance = int(
        runtime_metadata.get("records_per_instance") or 0)
    if records_per_instance <= 0:
        return slot_index
    return slot_index % records_per_instance


def _write_full_dump_artifacts(
        report_path: Path, exported_run: dict[str, Any], decoded: dict[str,
                                                                       Any],
        metadata_dict: dict[str, Any]) -> list[dict[str, Any]]:
    runtime_metadata = dict(exported_run.get("runtime_metadata") or {})
    plan_by_record = _full_dump_plan_by_record(metadata_dict)
    raw_buffer = bytes(exported_run.get("raw_buffer", b""))
    header = decoded.get("header", {})
    if int(header.get("overflow_count",
                      0)) != 0 or int(header.get("flags", 0)) & 1:
        raise RuntimeError(
            "level-2 debugger full dump cannot export from an overflowed debug buffer"
        )

    artifact_dir = report_path.with_suffix("")
    artifact_dir = artifact_dir.with_name(f"{artifact_dir.name}_artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[dict[str, Any]] = []
    inactive: list[dict[str, Any]] = []
    records = decoded.get("records", [])
    for slot_index, record in enumerate(records):
        if not isinstance(
                record, Mapping) or record.get("record_kind") != "FULL_VALUE":
            continue
        record_index = _record_index_for_slot(slot_index, runtime_metadata)
        plan = plan_by_record.get(record_index)
        if plan is None:
            raise RuntimeError(
                f"missing full-dump plan for record_index={record_index}")
        payload_offset = int(record.get("payload_offset", 0))
        payload_length = int(record.get("payload_length", 0))
        if payload_length <= 0:
            if payload_length == 0 and payload_offset == 0 and plan.get(
                    "conditional"):
                inactive.append({
                    "op_id":
                    int(record["op_id"]),
                    "logical_instance_id":
                    int(record["logical_instance_id"]),
                    "record_index":
                    record_index,
                    "reason":
                    "unexecuted_control_flow"
                })
                continue
            raise RuntimeError(
                f"empty full-dump payload for record_index={record_index}")
        if payload_length != int(plan["payload_length"]):
            raise RuntimeError(
                f"full-dump payload size differs from plan for record_index={record_index}"
            )
        if payload_offset < 0 or payload_offset + payload_length > len(
                raw_buffer):
            raise RuntimeError(
                f"full-dump payload range is outside raw buffer for record_index={record_index}"
            )
        payload = raw_buffer[payload_offset:payload_offset + payload_length]
        dtype = str(plan.get("artifact_dtype", ""))
        shape = plan.get("shape") or [int(plan.get("element_count", 0))]
        kind = str(plan.get("kind", "value"))
        op_id = int(record.get("op_id", plan.get("op_id", 0)))
        instance_id = int(record.get("logical_instance_id", 0))
        stem = (f"op{op_id}_inst{instance_id}_rec{record_index}_"
                f"{_safe_filename_component(kind, 'dump')}.npy")
        artifact_path = artifact_dir / stem
        _write_npy(artifact_path, payload, dtype, shape)
        artifacts.append({
            "op_id": op_id,
            "logical_instance_id": instance_id,
            "record_index": record_index,
            "kind": kind,
            "source": plan.get("source", ""),
            "artifact_dtype": dtype,
            "shape": list(shape),
            "payload_offset": payload_offset,
            "payload_length": payload_length,
            "path": str(artifact_path),
        })

    if plan_by_record and not artifacts:
        # Empty output is valid only if every planned capture in every program
        # instance was explicitly classified as unexecuted control flow.
        instance_count = math.prod(runtime_metadata.get("grid") or (1, ))
        expected = {(record_index, instance)
                    for instance in range(instance_count)
                    for record_index in plan_by_record}
        inactive_records = {(r["record_index"], r["logical_instance_id"])
                            for r in inactive}
        if not expected or inactive_records != expected:
            raise RuntimeError(
                "level-2 debugger did not produce any full-dump artifacts")

    inactive_keys = {(r["op_id"], r["logical_instance_id"]) for r in inactive}
    active_keys = {(a["op_id"], a["logical_instance_id"]) for a in artifacts}
    if inactive_keys & active_keys:
        raise RuntimeError(
            "Partially missing L2 payloads for an executed operation")
    runtime_metadata["inactive_record_slots"] = [
        slot for slot, record in enumerate(records)
        if (record["op_id"], record["logical_instance_id"]) in inactive_keys
    ]
    runtime_metadata["inactive_full_dump_records"] = inactive
    index_path = artifact_dir / "tensor_index.json"
    index = {
        "kernel_name":
        metadata_dict.get("debug_kernel_name") or metadata_dict.get("name")
        or "",
        "kernel_id":
        metadata_dict.get("debug_kernel_id", 0),
        "run_id":
        _exported_run_meta(exported_run).get("run_id", 0),
        "artifacts":
        artifacts,
        "inactive_records":
        inactive,
    }
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True))
    runtime_metadata["full_dump_artifacts"] = artifacts
    exported_run["runtime_metadata"] = runtime_metadata
    exported_run["full_dump_artifact_dir"] = str(artifact_dir)
    exported_run["full_dump_index_path"] = str(index_path)
    return artifacts


def _precision_tolerance(dtype: str) -> tuple[float, float]:
    normalized = dtype.lower()
    if "bf16" in normalized:
        return 1e-2, 1e-2
    if "f16" in normalized:
        return 1e-3, 5e-4
    if "f32" in normalized:
        return 1e-6, 1e-6
    return 0.0, 0.0


def _diagnostic_number(value: Any) -> float | str:
    number = float(value)
    if number != number:
        return "nan"
    if number == float("inf"):
        return "inf"
    if number == -float("inf"):
        return "-inf"
    return number


def _build_precision_diagnostics(
        metadata_dict: Mapping[str, Any],
        artifacts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    try:
        import numpy as np
    except ImportError:
        return []

    value_artifacts: dict[tuple[int, int], Mapping[str, Any]] = {}
    for artifact in artifacts:
        if artifact.get("kind") != "value" or not artifact.get("path"):
            continue
        key = (int(artifact.get("op_id",
                                0)), int(artifact.get("logical_instance_id",
                                                      0)))
        value_artifacts.setdefault(key, artifact)

    diagnostics: list[dict[str, Any]] = []
    tracked_table = metadata_dict.get("debug_tracked_table") or []
    for tracked in tracked_table:
        if not isinstance(tracked, Mapping):
            continue
        mlir_op = str(tracked.get("mlirOpName", ""))
        if mlir_op not in {"arith.extf", "arith.truncf"}:
            continue
        operands = tracked.get("operands") or []
        operand = next((item for item in operands if isinstance(item, Mapping)
                        and int(item.get("producerOpId", 0)) != 0), None)
        if operand is None:
            continue
        op_id = int(tracked.get("opId", 0))
        producer_op_id = int(operand.get("producerOpId", 0))
        from_dtype = str((operand.get("value") or {}).get("elementDtype", ""))
        to_dtype = str((tracked.get("result") or {}).get("elementDtype", ""))
        abs_tolerance, rel_tolerance = _precision_tolerance(to_dtype)

        instance_ids = sorted(
            instance_id for artifact_op_id, instance_id in value_artifacts
            if artifact_op_id == op_id and (producer_op_id,
                                            instance_id) in value_artifacts)
        for instance_id in instance_ids:
            before_artifact = value_artifacts[(producer_op_id, instance_id)]
            after_artifact = value_artifacts[(op_id, instance_id)]
            try:
                before = np.load(str(before_artifact["path"])).reshape(-1)
                after = np.load(str(after_artifact["path"])).reshape(-1)
            except (OSError, ValueError):
                continue
            if before.shape != after.shape:
                continue

            before64 = before.astype(np.float64, copy=False)
            after64 = after.astype(np.float64, copy=False)
            same_nonfinite = ((np.isnan(before64) & np.isnan(after64)) |
                              ((before64 == after64) & np.isinf(before64)
                               & np.isinf(after64)))
            finite_pairs = np.isfinite(before64) & np.isfinite(after64)
            abs_error = np.zeros(before64.shape, dtype=np.float64)
            abs_error[finite_pairs] = np.abs(after64[finite_pairs] -
                                             before64[finite_pairs])
            abs_error[~finite_pairs & ~same_nonfinite] = np.inf
            rel_error = np.zeros(before64.shape, dtype=np.float64)
            nonzero = finite_pairs & (np.abs(before64) > 0)
            rel_error[nonzero] = (abs_error[nonzero] /
                                  np.abs(before64[nonzero]))
            rel_error[finite_pairs & ~nonzero & (abs_error > 0)] = np.inf
            rel_error[~finite_pairs & ~same_nonfinite] = np.inf

            suspicious = (~same_nonfinite & ~finite_pairs) | (
                finite_pairs &
                (abs_error > abs_tolerance + rel_tolerance * np.abs(before64)))
            suspicious_indices = np.flatnonzero(suspicious)
            changed = (~same_nonfinite & ~finite_pairs) | (finite_pairs &
                                                           (abs_error > 0))
            changed_indices = np.flatnonzero(changed)
            finite_error = abs_error[np.isfinite(abs_error)]
            if finite_error.size:
                max_abs_error = float(np.max(finite_error))
                mean_abs_error = float(np.mean(finite_error))
                rms_error = float(np.sqrt(np.mean(finite_error**2)))
                l2_error = float(np.sqrt(np.sum(finite_error**2)))
            else:
                max_abs_error = mean_abs_error = rms_error = l2_error = 0.0
            finite_relative = rel_error[np.isfinite(rel_error)]
            max_rel_error = (float(np.max(finite_relative))
                             if finite_relative.size else 0.0)

            worst_lane = None
            if abs_error.size:
                worst_index = int(np.argmax(abs_error))
                worst_lane = {
                    "index": worst_index,
                    "before": _diagnostic_number(before64[worst_index]),
                    "after": _diagnostic_number(after64[worst_index]),
                    "abs_error": _diagnostic_number(abs_error[worst_index]),
                    "rel_error": _diagnostic_number(rel_error[worst_index]),
                }

            diagnostics.append({
                "op_id":
                op_id,
                "producer_op_id":
                producer_op_id,
                "logical_instance_id":
                instance_id,
                "statement_id":
                int(tracked.get("statementId", 0)),
                "source_loc":
                str(tracked.get("sourceLoc", "")),
                "statement":
                str(tracked.get("tritonStatement", "")),
                "conversion_op":
                mlir_op,
                "from": {
                    "dtype": from_dtype,
                    "artifact": str(before_artifact["path"]),
                },
                "to": {
                    "dtype": to_dtype,
                    "artifact": str(after_artifact["path"]),
                },
                "comparison_basis":
                "numeric value before/after conversion",
                "element_count":
                int(before64.size),
                "max_abs_error":
                max_abs_error,
                "mean_abs_error":
                mean_abs_error,
                "max_rel_error":
                max_rel_error,
                "rms_error":
                rms_error,
                "l2_error":
                l2_error,
                "suspicious_lane_count":
                int(suspicious_indices.size),
                "suspicious_lanes":
                [int(index) for index in suspicious_indices[:64]],
                "suspicious_lanes_truncated":
                bool(suspicious_indices.size > 64),
                "changed_lane_count":
                int(changed_indices.size),
                "conversion_loss":
                "lossy" if changed_indices.size else "exact",
                "worst_lane":
                worst_lane,
                "tolerance": {
                    "abs": abs_tolerance,
                    "rel": rel_tolerance,
                },
                "status":
                "warning" if suspicious_indices.size else "ok",
            })
    return diagnostics


def _render_precision_diagnostics(
        diagnostics: Sequence[Mapping[str, Any]]) -> str:
    if not diagnostics:
        return ""
    lines = ["Precision Conversion Diagnostics"]
    for diagnostic in diagnostics:
        lines.extend([
            f"statement_id: {diagnostic['statement_id']}",
            f"statement: {diagnostic['statement']}",
            "precision_conversion:",
            f"  op_id: {diagnostic['op_id']}",
            f"  logical_instance_id: {diagnostic['logical_instance_id']}",
            f"  from: {diagnostic['from']['dtype']}",
            f"  to: {diagnostic['to']['dtype']}",
            f"  max_abs_error: {diagnostic['max_abs_error']}",
            f"  mean_abs_error: {diagnostic['mean_abs_error']}",
            f"  max_rel_error: {diagnostic['max_rel_error']}",
            f"  rms_error: {diagnostic['rms_error']}",
            f"  l2_error: {diagnostic['l2_error']}",
            f"  suspicious_lane_count: {diagnostic['suspicious_lane_count']}",
            f"  suspicious_lanes: {diagnostic['suspicious_lanes']}",
            f"  changed_lane_count: {diagnostic['changed_lane_count']}",
            f"  conversion_loss: {diagnostic['conversion_loss']}",
            f"  worst_lane: {diagnostic['worst_lane']}",
            f"  tolerance: {diagnostic['tolerance']}",
            f"  status: {diagnostic['status']}",
            "",
        ])
    return "\n".join(lines).rstrip()


def _inject_precision_diagnostics(report: str,
                                  diagnostics: Sequence[Mapping[str, Any]],
                                  *,
                                  op_log: bool = False) -> str:
    if not report or not diagnostics:
        return report
    try:
        document = json.loads(report)
    except (TypeError, ValueError):
        return report
    document["precision_diagnostics"] = list(diagnostics)
    if op_log:
        records = (document.get("op_log") or {}).get("records_by_op") or []
        for record in records:
            matches = [
                item for item in diagnostics
                if int(item["op_id"]) == int(record.get("op_id", 0))
            ]
            if matches:
                record["precision_conversion"] = matches
    else:
        for statement in document.get("records_by_op") or []:
            matches = [
                item for item in diagnostics
                if int(item["statement_id"]) == int(
                    statement.get("statement_id", 0))
            ]
            if matches:
                statement["precision_conversion"] = matches
    return json.dumps(document, indent=2, sort_keys=True)


def _fill_summary_bundles_from_full_dump(exported_run, decoded, metadata,
                                         artifacts):
    """Derive L2 summaries from actual device payloads, retaining the record ABI."""
    import numpy as np

    by_value = {
        (a["op_id"], a["logical_instance_id"]): a
        for a in artifacts if a["kind"] == "value"
    }
    raw = bytearray(exported_run["raw_buffer"])
    record_size = int(metadata["debug_record_size"])
    computed = {}
    device_summary_ops = set()
    host_summary_ops = {
        int(entry["op_id"])
        for entry in metadata.get("debug_full_dump_plan", [])
        if entry["kind"] == "value"
    }
    inactive_slots = set(exported_run["runtime_metadata"].get(
        "inactive_record_slots", []))
    for slot, record in enumerate(decoded["records"]):
        if slot in inactive_slots:
            continue
        kind = record["record_kind"]
        if kind not in {
                "SUMMARY_COUNT_BUNDLE_U64", "SUMMARY_VALUE_BUNDLE_F32"
        }:
            continue
        if record["op_id"] not in host_summary_ops:
            device_summary_ops.add(record["op_id"])
            continue
        key = (record["op_id"], record["logical_instance_id"])
        if key not in computed:
            if key not in by_value:
                raise RuntimeError(
                    f"Missing L2 value payload for summary {key}")
            values = np.load(by_value[key]["path"],
                             allow_pickle=False).astype(np.float32).reshape(-1)
            finite = values[np.isfinite(values)]
            counts = (int(np.isnan(values).sum()), int(np.isinf(values).sum()),
                      int((values == 0).sum()), int(values.size))
            with np.errstate(over="ignore", invalid="ignore"):
                metrics = (float(finite.mean()) if finite.size else 0.0,
                           float(finite.min()) if finite.size else 0.0,
                           float(finite.max()) if finite.size else 0.0,
                           float(
                               np.sqrt(
                                   np.sum(finite * finite, dtype=np.float32))))
            computed[key] = counts, metrics
        counts, metrics = computed[key]
        offset = 32 + slot * record_size + 16
        if kind == "SUMMARY_COUNT_BUNDLE_U64":
            struct.pack_into("<4Q", raw, offset, *counts)
        else:
            struct.pack_into("<4f", raw, offset, *metrics)
    exported_run["raw_buffer"] = bytes(raw)
    exported_run["runtime_metadata"]["host_summary_op_ids"] = sorted(
        host_summary_ops)
    exported_run["runtime_metadata"]["summary_source"] = (
        "mixed_device_and_host_from_device_full_dump"
        if device_summary_ops else "host_from_device_full_dump")


def _finalize_exported_run(exported_run: dict[str, Any],
                           metadata_dict: dict[str, Any]) -> dict[str, Any]:
    exported_run["debug_kernel_name"] = str(
        metadata_dict.get("debug_kernel_name") or metadata_dict.get("name")
        or "")
    tracked_table = metadata_dict.get("debug_tracked_table")
    if isinstance(tracked_table,
                  Sequence) and not isinstance(tracked_table,
                                               (str, bytes, bytearray)):
        exported_run["debug_tracked_table"] = list(tracked_table)

    binding = _load_binding()
    decoded = binding.decode_exported_run(exported_run)
    exported_run["decoded"] = decoded

    output_dir = _output_dir
    report_path = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = _build_report_path(output_dir, exported_run,
                                         metadata_dict)

    precision_diagnostics: list[dict[str, Any]] = []
    if _is_full_dump_run(metadata_dict):
        if report_path is None:
            raise RuntimeError(
                "level-2 debugger full dump requires debugger output_dir")
        artifacts = _write_full_dump_artifacts(report_path, exported_run,
                                               decoded, metadata_dict)
        if metadata_dict.get("debug_host_summary_bundles"):
            _fill_summary_bundles_from_full_dump(exported_run, decoded,
                                                 metadata_dict, artifacts)
        decoded = binding.decode_exported_run(exported_run)
        exported_run["decoded"] = decoded
        precision_diagnostics = _build_precision_diagnostics(
            metadata_dict, artifacts)
        if precision_diagnostics:
            runtime_metadata = dict(exported_run.get("runtime_metadata") or {})
            runtime_metadata["precision_diagnostics"] = precision_diagnostics
            exported_run["runtime_metadata"] = runtime_metadata
            exported_run["precision_diagnostics"] = precision_diagnostics
            index_path = exported_run.get("full_dump_index_path")
            if index_path:
                index = json.loads(Path(index_path).read_text())
                index["precision_diagnostics"] = precision_diagnostics
                Path(index_path).write_text(
                    json.dumps(index, indent=2, sort_keys=True))

    summary = _render_export_summary(exported_run, decoded, metadata_dict)
    report = ""
    json_report = ""
    op_log_report = ""
    op_log_json_report = ""
    metadata_json = metadata_dict.get("debug_metadata_json")
    if metadata_json:
        render_text_statement_report = getattr(binding,
                                               "render_text_statement_report",
                                               None)
        if callable(render_text_statement_report):
            report = render_text_statement_report(exported_run,
                                                  str(metadata_json))
        else:
            report = binding.render_text_report(exported_run,
                                                str(metadata_json))
        precision_report = _render_precision_diagnostics(precision_diagnostics)
        if precision_report:
            report = f"{report.rstrip()}\n\n{precision_report}\n"
        exported_run["report"] = report

        render_text_op_log_report = getattr(binding,
                                            "render_text_op_log_report", None)
        if callable(render_text_op_log_report):
            op_log_report = render_text_op_log_report(exported_run,
                                                      str(metadata_json))
            exported_run["op_log_report"] = op_log_report

        render_json_statement_report = getattr(binding,
                                               "render_json_statement_report",
                                               None)
        render_json_report = (render_json_statement_report
                              if callable(render_json_statement_report) else
                              getattr(binding, "render_json_report", None))
        if callable(render_json_report):
            json_report = render_json_report(exported_run, str(metadata_json))
            json_report = _inject_precision_diagnostics(
                json_report, precision_diagnostics)
            exported_run["json_report"] = json_report

        render_json_op_log_report = getattr(binding,
                                            "render_json_op_log_report", None)
        if callable(render_json_op_log_report):
            op_log_json_report = render_json_op_log_report(
                exported_run, str(metadata_json))
            op_log_json_report = _inject_precision_diagnostics(
                op_log_json_report, precision_diagnostics, op_log=True)
            exported_run["op_log_json_report"] = op_log_json_report

    summary_source = exported_run.get("runtime_metadata",
                                      {}).get("summary_source")
    if summary_source:
        summary += f"\nsummary_source: {summary_source}"
        for key in ("json_report", "op_log_json_report"):
            if exported_run.get(key):
                document = json.loads(exported_run[key])
                document["summary_source"] = summary_source
                exported_run[key] = json.dumps(document,
                                               indent=2,
                                               sort_keys=True)
        json_report = exported_run.get("json_report", json_report)
        op_log_json_report = exported_run.get("op_log_json_report",
                                              op_log_json_report)

    report_text = summary
    if report:
        report_text += "\n\n"
        report_text += report.lstrip("\n")

    op_log_report_text = summary
    if op_log_report:
        op_log_report_text += "\n\n"
        op_log_report_text += op_log_report.lstrip("\n")

    if report_path is not None:
        report_path.write_text(report_text)
        exported_run["report_path"] = str(report_path)
        if op_log_report:
            op_log_report_path = report_path.with_name(
                f"{report_path.stem}_op_log.txt")
            op_log_report_path.write_text(op_log_report_text)
            exported_run["op_log_report_path"] = str(op_log_report_path)
        if json_report:
            json_report_path = report_path.with_suffix(".json")
            json_report_path.write_text(json_report)
            exported_run["json_report_path"] = str(json_report_path)
        if op_log_json_report:
            op_log_json_report_path = report_path.with_name(
                f"{report_path.stem}_op_log.json")
            op_log_json_report_path.write_text(op_log_json_report)
            exported_run["op_log_json_report_path"] = str(
                op_log_json_report_path)
        if _active_config.export_raw_records:
            raw_records_path = report_path.with_name(
                f"{report_path.stem}_raw_records.txt")
            raw_records_path.write_text(
                _render_raw_records(exported_run, decoded, metadata_dict))
            exported_run["raw_records_path"] = str(raw_records_path)

    return exported_run


def _normalize_device_id(device: Any) -> int:
    if isinstance(device, int):
        return device
    index = getattr(device, "index", None)
    if isinstance(index, int):
        return index
    return 0


def _metadata_to_dict(metadata: Any) -> dict[str, Any]:
    if metadata is None:
        return {}
    if isinstance(metadata, dict):
        return dict(metadata)
    asdict = getattr(metadata, "_asdict", None)
    if callable(asdict):
        return dict(asdict())
    if hasattr(metadata, "__dict__"):
        return dict(vars(metadata))
    return {}


def _materialize_launch_metadata(launch_metadata: Any) -> Any:
    if launch_metadata is None or isinstance(launch_metadata, dict):
        return launch_metadata
    getter = getattr(launch_metadata, "get", None)
    if callable(getter):
        try:
            return getter()
        except TypeError:
            return launch_metadata
    return launch_metadata


def _normalize_launch_grid(
        launch_metadata: Any) -> tuple[int, int, int] | None:
    if not isinstance(launch_metadata, Mapping):
        return None
    grid = launch_metadata.get("grid")
    if grid is None:
        return None
    values = tuple(int(dim) for dim in grid)
    if not values:
        return None
    return (
        values[0],
        values[1] if len(values) > 1 else 1,
        values[2] if len(values) > 2 else 1,
    )


def _call_runtime_value(value: Any, name: str) -> Any:
    member = getattr(value, name, None)
    return member() if callable(member) else member


def _runtime_int_list(value: Any) -> list[int] | None:
    if value is None or isinstance(value, (str, bytes, bytearray)):
        return None
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError):
        return None


def _runtime_argument_names(metadata: Any, argument_count: int) -> list[str]:
    metadata_dict = _metadata_to_dict(metadata)
    for key in (
            "debug_argument_names",
            "argument_names",
            "arg_names",
            "signature",
    ):
        candidate = metadata_dict.get(key)
        if isinstance(candidate, Mapping):
            names = [str(name) for name in candidate]
        elif isinstance(candidate,
                        Sequence) and not isinstance(candidate,
                                                     (str, bytes, bytearray)):
            names = [str(name) for name in candidate]
        else:
            continue
        if len(names) >= argument_count:
            return names
    return [f"arg{index}" for index in range(argument_count)]


def _runtime_alignment(address: int) -> int:
    if address <= 0:
        return 0
    # The native contract stores alignment in uint32_t. Keep the greatest
    # representable power-of-two divisor rather than overflowing the field.
    return min(address & -address, 1 << 31)


def _runtime_tensor_layout(value: Any, shape: list[int],
                           stride: list[int]) -> str:
    is_contiguous = getattr(value, "is_contiguous", None)
    if callable(is_contiguous):
        try:
            if bool(is_contiguous()):
                return "contiguous"
        except (RuntimeError, TypeError):
            pass
    if shape or stride:
        return "strided"
    return "unknown"


def _runtime_tensor_size_bytes(value: Any, shape: list[int], stride: list[int],
                               element_size: int) -> int:
    numel = _call_runtime_value(value, "numel")
    try:
        logical_bytes = int(numel) * element_size
    except (TypeError, ValueError):
        logical_bytes = 0
    if not shape or not stride or len(shape) != len(stride):
        return max(0, logical_bytes)
    if any(dim == 0 for dim in shape):
        return 0
    span_elements = 1
    for dim, step in zip(shape, stride):
        if dim > 0:
            span_elements += (dim - 1) * abs(step)
    return max(logical_bytes, span_elements * element_size)


def _infer_runtime_metadata(metadata: Any,
                            kernel_args: Sequence[Any]) -> dict[str, Any]:
    """Build the runtime tensor/buffer inventory from launch arguments.

    The launcher deliberately forwards the original Python arguments, so this
    stays backend-neutral and does not alter the hidden-argument ABI. Objects
    without the tensor protocol used here are ignored.
    """
    names = _runtime_argument_names(metadata, len(kernel_args))
    buffers: list[dict[str, Any]] = []
    tensors: list[dict[str, Any]] = []
    buffer_ids: dict[tuple[str, int, int], int] = {}

    for argument_index, original_value in enumerate(kernel_args):
        value = original_value
        data_ptr = getattr(value, "data_ptr", None)
        if not callable(data_ptr):
            # Tensor descriptor objects expose their underlying tensor as
            # ``base`` on current Triton frontends.
            value = getattr(original_value, "base", None)
            data_ptr = getattr(value, "data_ptr", None)
        if value is None or not callable(data_ptr):
            continue

        try:
            address = int(data_ptr())
            shape = _runtime_int_list(_call_runtime_value(value, "shape"))
            stride = _runtime_int_list(_call_runtime_value(value, "stride"))
            element_size = int(_call_runtime_value(value, "element_size"))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            continue
        if address <= 0 or shape is None or stride is None or element_size <= 0:
            continue

        logical_name = names[argument_index]
        dtype = str(_call_runtime_value(value, "dtype") or "")
        if dtype.startswith("torch."):
            dtype = dtype[len("torch."):]
        device = str(_call_runtime_value(value, "device") or "")
        tensor_size = _runtime_tensor_size_bytes(value, shape, stride,
                                                 element_size)

        buffer_base = address
        buffer_size = tensor_size
        storage = _call_runtime_value(value, "untyped_storage")
        if storage is not None:
            storage_data_ptr = getattr(storage, "data_ptr", None)
            if callable(storage_data_ptr):
                try:
                    buffer_base = int(storage_data_ptr())
                except (RuntimeError, TypeError, ValueError):
                    buffer_base = address
            storage_nbytes = _call_runtime_value(storage, "nbytes")
            try:
                buffer_size = int(storage_nbytes)
            except (TypeError, ValueError):
                buffer_size = tensor_size

        key = (device, buffer_base, buffer_size)
        buffer_id = buffer_ids.get(key)
        if buffer_id is None:
            buffer_id = len(buffer_ids) + 1
            buffer_ids[key] = buffer_id
            buffers.append({
                "buffer_id": buffer_id,
                "buffer_name": logical_name,
                "base_address": buffer_base,
                "size_bytes": max(0, buffer_size),
                "alignment": _runtime_alignment(buffer_base),
            })

        tensors.append({
            "argument_index": argument_index,
            "logical_name": logical_name,
            "dtype": dtype,
            "shape": shape,
            "stride": stride,
            "layout": _runtime_tensor_layout(value, shape, stride),
            "buffer_id": buffer_id,
            "base_address": address,
            "size_bytes": tensor_size,
        })

    if not buffers and not tensors:
        return {}
    return {"buffers": buffers, "tensors": tensors}


def _build_launch_metadata_dict(metadata: Any) -> dict[str, Any]:
    metadata_dict = _metadata_to_dict(metadata)
    target = metadata_dict.get("target")
    target_name = _target_to_name(target)
    backend_name = (metadata_dict.get("debug_backend_name")
                    or os.environ.get("FLAGTREE_BACKEND")
                    or metadata_dict.get("backend_name")
                    or _target_backend(target) or "")
    backend_name = _normalize_backend_name(backend_name)

    launch_dict = dict(metadata_dict)
    launch_dict["debug_enabled"] = True
    launch_dict["debug_protocol_version"] = int(
        metadata_dict.get("debug_protocol_version", 2))
    launch_dict["debug_record_level"] = int(
        metadata_dict.get("debug_record_level", _active_config.record_level))
    launch_dict["debug_addr_level"] = _normalize_addr_level(
        metadata_dict.get("debug_addr_level", _active_config.addr_level))
    launch_dict["debug_export_mode"] = _normalize_export_mode(
        metadata_dict.get("debug_export_mode", _active_config.export_mode))
    launch_dict["debug_record_capacity"] = int(
        metadata_dict.get("debug_record_capacity",
                          _active_config.record_capacity))
    launch_dict["debug_record_size"] = int(
        metadata_dict.get("debug_record_size", 32))
    launch_dict["debug_kernel_id"] = int(
        metadata_dict.get("debug_kernel_id", _derive_kernel_id(metadata_dict)))
    launch_dict["debug_kernel_name"] = str(
        metadata_dict.get("debug_kernel_name", metadata_dict.get("name", "")))
    launch_dict["debug_backend_name"] = str(backend_name)
    launch_dict["debug_target_name"] = str(
        metadata_dict.get("debug_target_name", target_name))
    try:
        from triton.runtime.driver import driver

        launch_dict["debug_device_id"] = _normalize_device_id(
            driver.active.get_current_device())
    except Exception:
        launch_dict["debug_device_id"] = 0
    return launch_dict


def _default_launch_prepare_hook(
        metadata: Any, stream: int, launch_metadata: Any,
        kernel_args: Sequence[Any]) -> PreparedKernelLaunch:
    launch_metadata = _materialize_launch_metadata(launch_metadata)
    metadata_dict = _build_launch_metadata_dict(metadata)
    runtime_metadata = _infer_runtime_metadata(metadata, kernel_args)
    if _active_config.runtime_metadata_builder is not None:
        built_runtime_metadata = _active_config.runtime_metadata_builder(
            metadata, launch_metadata, kernel_args)
        if built_runtime_metadata is not None:
            runtime_metadata.update(dict(built_runtime_metadata))
    grid = _normalize_launch_grid(launch_metadata)
    if grid is not None:
        runtime_metadata.setdefault("grid", grid)
    records_per_instance = metadata_dict.get("debug_records_per_instance")
    if records_per_instance is not None:
        runtime_metadata.setdefault("records_per_instance",
                                    int(records_per_instance))
    if metadata_dict.get("debug_record_layout"):
        runtime_metadata.setdefault("record_layout",
                                    metadata_dict["debug_record_layout"])
    if metadata_dict.get("debug_record_plan") is not None:
        runtime_metadata.setdefault("record_plan",
                                    metadata_dict["debug_record_plan"])
    if _is_full_dump_run(metadata_dict):
        if _output_dir is None:
            raise RuntimeError(
                "level-2 debugger full dump requires debugger output_dir")
        runtime_metadata.setdefault(
            "full_dump_plan", metadata_dict.get("debug_full_dump_plan", []))

    handle = _load_binding().prepare_launch(metadata_dict, int(stream),
                                            runtime_metadata)
    host_start_time_ns = time.time_ns()

    def finalize(error: Optional[BaseException]) -> None:
        if error is not None and not _active_config.export_on_error:
            handle.release()
            return

        host_end_time_ns = time.time_ns()
        exported_run = handle.finish()
        exported_runtime_metadata = dict(
            exported_run.get("runtime_metadata", {}))
        exported_runtime_metadata.setdefault("host_start_time_ns",
                                             host_start_time_ns)
        exported_runtime_metadata.setdefault("host_end_time_ns",
                                             host_end_time_ns)
        exported_runtime_metadata.setdefault(
            "host_duration_ns",
            max(0, host_end_time_ns - host_start_time_ns),
        )
        exported_run["runtime_metadata"] = exported_runtime_metadata
        exported_run = _finalize_exported_run(exported_run, metadata_dict)
        _exported_runs.append(exported_run)
        if _active_config.export_handler is not None:
            _active_config.export_handler(exported_run)

    return PreparedKernelLaunch(
        kernel_args=(int(handle.hidden_arg_value), ),
        finalize=finalize,
    )


def prepare_metadata_only_kernel_launch(
    metadata: Any,
    stream: int,
    launch_metadata: Any = None,
    kernel_args: Optional[Sequence[Any]] = None,
) -> Optional[PreparedKernelLaunch]:
    del stream
    enabled = bool(getattr(metadata, "debug_enabled",
                           False)) or _active_config.enabled
    if not enabled:
        return None

    launch_metadata = _materialize_launch_metadata(launch_metadata)
    metadata_dict = _build_launch_metadata_dict(metadata)
    runtime_metadata = _infer_runtime_metadata(metadata,
                                               tuple(kernel_args or ()))
    if _active_config.runtime_metadata_builder is not None:
        built_runtime_metadata = _active_config.runtime_metadata_builder(
            metadata, launch_metadata, tuple(kernel_args or ()))
        if built_runtime_metadata is not None:
            runtime_metadata.update(dict(built_runtime_metadata))
    grid = _normalize_launch_grid(launch_metadata)
    if grid is not None:
        runtime_metadata.setdefault("grid", grid)
    runtime_metadata.setdefault(
        "records_per_instance",
        int(metadata_dict.get("debug_records_per_instance", 0) or 0),
    )
    if metadata_dict.get("debug_record_layout"):
        runtime_metadata.setdefault("record_layout",
                                    metadata_dict["debug_record_layout"])
    if metadata_dict.get("debug_record_plan") is not None:
        runtime_metadata.setdefault("record_plan",
                                    metadata_dict["debug_record_plan"])
    host_start_time_ns = time.time_ns()

    def finalize(error: Optional[BaseException]) -> None:
        if error is not None and not _active_config.export_on_error:
            return

        host_end_time_ns = time.time_ns()
        runtime_metadata["host_start_time_ns"] = host_start_time_ns
        runtime_metadata["host_end_time_ns"] = host_end_time_ns
        runtime_metadata["host_duration_ns"] = max(
            0, host_end_time_ns - host_start_time_ns)
        exported_run = _metadata_only_exported_run(metadata_dict,
                                                   runtime_metadata)
        exported_run = _finalize_exported_run(exported_run, metadata_dict)
        _exported_runs.append(exported_run)
        if _active_config.export_handler is not None:
            _active_config.export_handler(exported_run)

    return PreparedKernelLaunch(kernel_args=(), finalize=finalize)


def register_launch_prepare_hook(hook: Callable[..., Any]) -> None:
    global _launch_prepare_hook
    _launch_prepare_hook = _wrap_launch_prepare_hook(hook)


def clear_launch_prepare_hook() -> None:
    global _launch_prepare_hook
    _launch_prepare_hook = None


def is_active() -> bool:
    return _active_config.enabled


def current_compile_config() -> dict[str, Any]:
    if not _active_config.enabled:
        return {}
    return {
        "debug_enabled": True,
        "debug_protocol_version": 2,
        "debug_record_level": int(_active_config.record_level),
        "debug_addr_level": int(_active_config.addr_level),
        "debug_export_mode":
        _normalize_export_mode(_active_config.export_mode),
        "debug_record_capacity": int(_active_config.record_capacity),
        "debug_timeline_enabled": bool(_active_config.timeline_enabled),
    }


def activate(
    *,
    auto_collect: bool = False,
    level: int | None = None,
    addr_level: int = _DEFAULT_ADDR_LEVEL,
    timeline: Any = _USE_CURRENT_CONFIG,
    record_level: int | None = None,
    export_mode: Any = _USE_CURRENT_CONFIG,
    record_capacity: Any = _USE_CURRENT_CONFIG,
    export_on_error: Any = _USE_CURRENT_CONFIG,
    output_dir: Any = _USE_CURRENT_OUTPUT_DIR,
    export_raw_records: Any = _USE_CURRENT_CONFIG,
    runtime_metadata_builder: Optional[Callable[[Any, Any, Sequence[Any]],
                                                Any]] = None,
    export_handler: Optional[Callable[[dict[str, Any]], None]] = None,
) -> None:
    global _active_config
    _require_available()
    if output_dir is not _USE_CURRENT_OUTPUT_DIR:
        configure(output_dir=output_dir)

    if level is not None and record_level is not None:
        raise TypeError("use either level or record_level, not both")
    effective_level = record_level if record_level is not None else level
    if effective_level is None:
        effective_level = 1
    effective_level = _normalize_record_level(effective_level)
    effective_addr_level = _normalize_addr_level(addr_level)
    effective_export_mode = (_export_mode if export_mode is _USE_CURRENT_CONFIG
                             else _normalize_export_mode(export_mode))
    effective_record_capacity = (_record_capacity
                                 if record_capacity is _USE_CURRENT_CONFIG else
                                 int(record_capacity))
    if effective_record_capacity <= 0:
        raise ValueError("debugger record capacity must be positive")
    effective_export_on_error = (_export_on_error
                                 if export_on_error is _USE_CURRENT_CONFIG else
                                 bool(export_on_error))
    effective_export_raw_records = (_raw_record_export_enabled if
                                    export_raw_records is _USE_CURRENT_CONFIG
                                    else bool(export_raw_records))
    effective_timeline = (_timeline_enabled if timeline is _USE_CURRENT_CONFIG
                          else bool(timeline))

    _active_config = DebuggerConfig(
        enabled=True,
        record_level=effective_level,
        addr_level=effective_addr_level,
        timeline_enabled=effective_timeline,
        export_mode=effective_export_mode,
        record_capacity=effective_record_capacity,
        export_on_error=effective_export_on_error,
        output_dir=get_output_dir(),
        export_raw_records=effective_export_raw_records,
        runtime_metadata_builder=runtime_metadata_builder,
        export_handler=export_handler,
    )
    register_launch_prepare_hook(_default_launch_prepare_hook)

    from .compiler import set_instrumentation_mode

    set_instrumentation_mode(
        "debugger_auto_numeric" if auto_collect else "debugger")


def deactivate() -> None:
    global _active_config
    clear_launch_prepare_hook()
    _active_config = DebuggerConfig()

    from .compiler import set_instrumentation_mode

    set_instrumentation_mode("")


def clear_exported_runs() -> None:
    _exported_runs.clear()


def peek_exported_runs() -> list[dict[str, Any]]:
    return list(_exported_runs)


def take_exported_runs() -> list[dict[str, Any]]:
    exported_runs = list(_exported_runs)
    _exported_runs.clear()
    return exported_runs


def prepare_kernel_launch(
    metadata: Any,
    stream: int,
    launch_metadata: Any = None,
    kernel_args: Optional[Sequence[Any]] = None
) -> Optional[PreparedKernelLaunch]:
    enabled = bool(getattr(metadata, "debug_enabled",
                           False)) or _active_config.enabled
    if not enabled:
        return None

    if _launch_prepare_hook is None:
        raise RuntimeError(
            "debug-enabled kernel launch requires "
            "flagtree.debugger.register_launch_prepare_hook(...)")

    prepared = _launch_prepare_hook(
        metadata,
        int(stream),
        launch_metadata,
        tuple(kernel_args or ()),
    )
    if prepared is None:
        raise RuntimeError(
            "debugger launch prepare hook returned None for a debug-enabled "
            "kernel")
    if isinstance(prepared, PreparedKernelLaunch):
        return PreparedKernelLaunch(
            kernel_args=_normalize_kernel_args(prepared.kernel_args),
            finalize=prepared.finalize,
        )
    return PreparedKernelLaunch(kernel_args=_normalize_kernel_args(prepared))


def finalize_prepared_launch(prepared: Optional[PreparedKernelLaunch],
                             error: Optional[BaseException] = None) -> None:
    if prepared is None or prepared.finalize is None:
        return
    prepared.finalize(error)


@contextmanager
def launch_context(
    backend: str,
    metadata: Any,
    grid: Sequence[int],
    stream: int,
    launch_metadata: Any = None,
    kernel_args: Optional[Sequence[Any]] = None,
):
    backend = str(backend or "unknown").lower()
    metadata_dict = _metadata_to_dict(metadata)
    runtime_launch_metadata = _materialize_launch_metadata(launch_metadata)
    runtime_launch_metadata = dict(runtime_launch_metadata or {})
    runtime_launch_metadata.setdefault("grid", tuple(int(dim) for dim in grid))
    has_hidden_arg = bool(metadata_dict.get("debug_launch_hidden_arg", False))
    prepare = prepare_kernel_launch if has_hidden_arg else prepare_metadata_only_kernel_launch
    prepared = None
    launch_error = None
    try:
        prepared = prepare(
            metadata,
            int(stream),
            runtime_launch_metadata,
            tuple(kernel_args or ()),
        )
        hidden_args = () if prepared is None else prepared.kernel_args
        expected = int(has_hidden_arg)
        if len(hidden_args) != expected:
            raise RuntimeError(
                f"instrumented {backend} kernel requires {expected} debugger hidden "
                f"argument(s), but the debugger prepared {len(hidden_args)}")
        yield hidden_args
        if has_hidden_arg:
            if backend in {"ascend", "cann", "npu"}:
                import torch_npu

                torch_npu.npu.synchronize()
            elif backend in {"tianshu", "corex", "iluvatar"}:
                import torch

                torch.cuda.synchronize()
            elif backend in {"gcu", "enflame"}:
                import torch

                torch.gcu.synchronize()
            elif backend in {"mthreads", "musa"}:
                import torch

                torch.musa.synchronize()
            elif backend in {"cuda", "nvidia"}:
                # FlagPrism: TransferEngine.syncExport synchronizes the exact
                # launch stream after the device-to-host copy. A process-wide
                # torch.cuda.synchronize() would also stall unrelated streams
                # and is unnecessary for the CUDA driver-backed adapter.
                pass
            else:
                raise RuntimeError(
                    f"FlagPrism has no hidden-argument synchronization adapter "
                    f"for backend {backend!r}")
    except BaseException as error:
        launch_error = error
        raise
    finally:
        finalize_prepared_launch(prepared, launch_error)


def ascend_launch_context(
    metadata: Any,
    grid: Sequence[int],
    stream: int,
    launch_metadata: Any = None,
    kernel_args: Optional[Sequence[Any]] = None,
):
    """Compatibility alias for callers predating the generic host gateway."""
    return launch_context("ascend", metadata, grid, stream, launch_metadata,
                          kernel_args)


__all__ = (
    "DebuggerConfig",
    "PreparedKernelLaunch",
    "activate",
    "ascend_launch_context",
    "clear_exported_runs",
    "clear_launch_prepare_hook",
    "configure",
    "current_compile_config",
    "deactivate",
    "finalize_prepared_launch",
    "get_config",
    "get_output_dir",
    "is_active",
    "is_available",
    "launch_context",
    "peek_exported_runs",
    "prepare_kernel_launch",
    "prepare_metadata_only_kernel_launch",
    "register_launch_prepare_hook",
    "reset_config",
    "set_output_dir",
    "take_exported_runs",
)
