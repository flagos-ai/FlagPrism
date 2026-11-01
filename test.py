#!/usr/bin/env python3
"""Self-contained Triton operator acceptance for FlagTree and FlagPrism."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
OPERATORS = ROOT / "tests/operators.py"

ALIASES = {
    "absolute": "abs",
    "arcsinh": "asinh",
    "arctanh": "atanh",
    "negative": "neg",
    "concatenate": "cat",
    "clip": "clamp",
}


def canonical(name):
    for suffix in ("_out", "_"):
        if name.endswith(suffix):
            return ALIASES.get(name[:-len(suffix)],
                               name[:-len(suffix)]) + suffix
    return ALIASES.get(name, name)


def family(name):
    name = canonical(name)
    return name[:-4] if name.endswith("_out") else name.rstrip("_")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def catalog(requested):
    # Listing needs neither a device nor installed Torch/Triton. The literal
    # catalog and actual kernels live together in the repository-owned module.
    source = OPERATORS.read_text()
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "CASES"
                    for t in n.targets))
    cases = ast.literal_eval(node.value)
    known = {case["op"] for case in cases}
    unknown = set(requested or []) - known
    if unknown:
        raise ValueError(f"Unknown operator IDs: {sorted(unknown)}")
    selected = []
    for entry in sorted(cases, key=lambda c: c["op"]):
        if requested and entry["op"] not in requested:
            continue
        entry.update(case_id="small", canonical_op=entry["op"])
        entry["body"] = entry["setup"] + "\n" + entry["operation"]
        entry["body_sha256"] = hashlib.sha256(
            entry["body"].encode()).hexdigest()
        selected.append(entry)
    return selected, []


def load_operators():
    spec = importlib.util.spec_from_file_location("flagprism_test_operators",
                                                  OPERATORS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cpu_namespace(case, torch):
    torch.manual_seed(0)
    namespace = {"torch": torch, "device": "cpu"}
    exec(compile(case["setup"], "<shared-inputs>", "exec"), namespace)
    return namespace


def to_cpu(value, torch):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, (tuple, list)):
        return (type(value)(
            to_cpu(v, torch)
            for v in value) if type(value) in (list, tuple) else tuple(
                to_cpu(v, torch) for v in value))
    if isinstance(value, dict):
        return {k: to_cpu(v, torch) for k, v in value.items()}
    return value


def check_reports(stage, tool, base, launched, level=1):
    if stage == "debugger":
        runs = tool.take_exported_runs()
        assert runs, "No exported debugger runs"
        total = 0
        full_value_records = 0
        summary_sources = set()
        captured = set()
        for run in runs:
            decoded = run["decoded"]
            assert decoded["header"][
                "overflow_count"] == 0, "Debug buffer overflow"
            records = decoded["records"]
            assert records, "Empty dynamic records"
            if level == 2:
                import numpy as np

                full_records = [
                    r for r in records if r["record_kind"] == "FULL_VALUE"
                ]
                assert full_records, "L2 produced no full tensor records"
                index_path = Path(run["full_dump_index_path"])
                artifacts = json.loads(index_path.read_text())["artifacts"]
                assert len(artifacts) == len(
                    full_records), "Incomplete L2 tensor export"
                assert any(a["kind"] == "value"
                           for a in artifacts), "No L2 value tensors"
                for artifact in artifacts:
                    tensor = np.load(artifact["path"], allow_pickle=False)
                    assert tensor.size > 0, "Empty L2 tensor"
                    assert (list(tensor.shape) == artifact["shape"]
                            ), "L2 tensor shape mismatch"
                    assert (tensor.nbytes == artifact["payload_length"]
                            ), "L2 tensor payload size mismatch"
                full_value_records += len(full_records)
            else:
                assert any(r["record_kind"] in (
                    "SUMMARY_COUNT_BUNDLE_U64",
                    "SUMMARY_VALUE_BUNDLE_F32",
                    "SUMMARY",
                ) for r in records), "No numeric records"
            total += len(records)
            for key in ("report_path", "json_report_path",
                        "op_log_report_path"):
                path = Path(run[key])
                assert path.is_file() and path.stat(
                ).st_size, f"Missing report: {key}"
            report = json.loads(Path(run["json_report_path"]).read_text())
            assert report, "Empty debugger JSON report"
            summary_sources.add(report.get("summary_source", "device"))
            captured.add(report["kernel_name"])
        missing = sorted(set(launched) - captured)
        assert not missing, f"Missing debugger captures for Triton kernels: {missing}"
        return {
            "debug_runs": len(runs),
            "dynamic_records": total,
            "captured_kernels": sorted(captured),
            "full_value_records": full_value_records,
            "summary_sources": sorted(summary_sources),
        }
    if stage == "profiler":
        roots = json.loads(base.with_suffix(".hatchet").read_text())
        matches = []

        def visit(node):
            name = node.get("frame", {}).get("name", "")
            metrics = node.get("metrics", {})
            time_ns = float(metrics.get("time (ns)", 0))
            if (any(kernel in name for kernel in launched)
                    and "device_id" in metrics and math.isfinite(time_ns)
                    and time_ns > 0):
                matches.append({"name": name, "time_ns": time_ns})
            for child in node.get("children", []):
                visit(child)

        assert isinstance(roots, list) and roots, "Empty profiler report"
        for root in roots:
            visit(root)
        assert matches, "No measured device kernel matching actual Triton launches"
        missing = sorted({
            kernel
            for kernel in launched
            if not any(kernel in item["name"] for item in matches)
        })
        assert not missing, f"Missing device timings for Triton kernels: {missing}"
        meta = base.with_suffix(".meta.json")
        if meta.exists():
            config = json.loads(meta.read_text()).get("config", {})
            assert (
                str(config.get("runtime_host_timing_fallback",
                               "false")).lower()
                != "true"), "Host timing fallback is not a device measurement"
        return {"profiled_kernels": matches}
    return {}


def worker(args):
    case = json.loads(args.worker.read_text())
    stage = args.stage
    result = {
        "op": case["op"],
        "case_id": case["case_id"],
        "stage": stage,
        "body_sha256": case["body_sha256"],
        "status": "FAIL",
    }
    if stage == "debugger":
        result["debug_level"] = args.level
    try:
        tool = None
        if stage in ("debugger", "profiler"):
            try:
                tool = importlib.import_module(f"flagtree.{stage}")
                if stage == "debugger":
                    if not tool.is_available():
                        raise RuntimeError(
                            "Debugger native bindings unavailable")
                else:
                    importlib.import_module(
                        "flagtree.profiler.native").runtime_binding()
            except Exception as error:
                result.update(status="UNAVAILABLE",
                              error=str(error),
                              traceback=traceback.format_exc())
                return result
        try:
            import torch

            if args.torch_extension:
                importlib.import_module(args.torch_extension)
            else:
                for module in ("torch_npu", "torch_gcu", "torch_musa"):
                    if importlib.util.find_spec(module):
                        importlib.import_module(module)
            import triton

            driver = triton.runtime.driver.active
            interface = driver.get_device_interface()
            if args.device is not None:
                interface.set_device(args.device)
            device = driver.get_active_torch_device()
            target = driver.get_current_target()
            assert torch.device(device).type != "cpu", "Accelerator required"
            result.update(target=str(target), device=str(device))
        except Exception as error:
            result.update(status="SETUP_ERROR",
                          error=str(error),
                          traceback=traceback.format_exc())
            return result
        # Independent native CPU reference, with deterministic inputs.
        reference = cpu_namespace(case, torch)
        exec(compile(case["operation"], "<cpu-reference>", "exec"), reference)
        expected = to_cpu(reference["result"], torch)
        namespace = cpu_namespace(case, torch)
        # Preserve repeated tensor references within the input namespace.
        moved = {}
        for name, value in list(namespace.items()):
            if isinstance(value, torch.Tensor):
                if id(value) not in moved:
                    moved[id(value)] = value.to(device)
                namespace[name] = moved[id(value)]
        namespace["device"] = device
        if stage == "debugger":
            tool.activate(
                auto_collect=True,
                level=args.level,
                addr_level=args.addr_level,
                record_capacity=args.record_capacity,
                output_dir=args.result.parent / "debug",
            )
        operators = load_operators()
        result["operator_source"] = str(OPERATORS)
        result["launch_policy"] = "fixed_config_no_autotuning"
        base = args.result.parent / "profile"
        session = None
        if stage == "profiler":
            backend = args.profiler_backend or {
                "cuda": "cupti",
                "hip": "roctracer",
                "ascend": "cann",
                "npu": "cann",
                "cann": "cann",
                # FlagPrism：run the same acceptance suite on MACA.
                "maca": "metax",
                "metax": "metax",
                # FlagPrism end
                "mthreads": "mthreads",
                "musa": "mthreads",
                "tianshu": "tianshu",
                "iluvatar": "tianshu",
                "corex": "tianshu",
                "gcu": "enflame",
                "tops": "enflame",
                "enflame": "enflame",
            }.get(target.backend)
            assert backend, "Specify --profiler-backend for this backend"
            opts = {
                "backend": backend,
                "hook": "triton",
                "context": "shadow",
                "data": "tree",
            }
            if args.profiler_mode:
                opts["mode"] = args.profiler_mode
            elif backend not in ("cupti", "roctracer"):
                opts["mode"] = (
                    f"runtime_base:runtime_host_timing_fallback=false:device_id={torch.device(device).index or 0}"
                )
            session = tool.start(str(base), **opts)
        launched = []
        previous = triton.knobs.runtime.launch_enter_hook

        def observe(metadata):
            launched.append(metadata.get()["name"])
            if previous:
                previous(metadata)

        triton.knobs.runtime.launch_enter_hook = observe
        try:
            namespace["result"] = operators.run(case, namespace)
            interface.synchronize()
        finally:
            triton.knobs.runtime.launch_enter_hook = previous
            if session is not None:
                tool.finalize(session)
        assert (
            launched
        ), "No Triton launch: a native fallback or view-only op is not a pass"
        torch.testing.assert_close(
            to_cpu(namespace["result"], torch),
            expected,
            rtol=1e-3,
            atol=1e-4,
            equal_nan=True,
        )
        result.update(check_reports(stage, tool, base, launched, args.level))
        result.update(status="PASS", launched_kernels=launched)
    except Exception as error:
        result.update(error=str(error), traceback=traceback.format_exc())
    finally:
        write_json(args.result, result)
    return result


def command(argv, log, timeout):
    with log.open("w") as output:
        proc = subprocess.Popen(argv,
                                stdout=output,
                                stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            return 124
        except KeyboardInterrupt:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait()
            raise


def git_revision(path):
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--devices",
        help="Comma-separated device indices; at most one case per device")
    parser.add_argument("--device", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Concurrent workers; --jobs > 1 requires --devices (default: 1)")
    parser.add_argument("--out", type=Path, default=ROOT / "test-results")
    parser.add_argument("--ops",
                        help="Comma-separated operator IDs; diagnostic subset")
    parser.add_argument(
        "--min-ops",
        type=int,
        default=100,
        help="Required IDs after merging aliases and in-place/out variants",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("execute", "debugger", "profiler"),
        default=["debugger", "profiler"],
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Write the shared manifest without using a device",
    )
    parser.add_argument(
        "--check-cases",
        action="store_true",
        help="Validate all shared CPU references, no device claims",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Override seconds per stage (default: L2 600, others 180)",
    )
    parser.add_argument("--torch-extension")
    parser.add_argument("--profiler-backend")
    parser.add_argument("--profiler-mode")
    parser.add_argument(
        "--level",
        type=int,
        choices=(1, 2),
        help="Run only this debugger level (default: run both L1 and L2)",
    )
    parser.add_argument("--addr-level", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--record-capacity", type=int, default=65536)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--stage",
                        choices=("execute", "debugger", "profiler"),
                        help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.stages = list(dict.fromkeys(args.stages))
    if args.worker:
        if args.level is None:
            parser.error("--worker requires --level")
        return 0 if worker(args)["status"] == "PASS" else 1
    if (args.min_ops < 1 or (args.timeout is not None and args.timeout < 1)
            or args.record_capacity < 1):
        parser.error("min-ops, timeout, and record-capacity must be positive")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.jobs > 1 and not args.devices:
        parser.error(
            "--jobs > 1 requires explicit --devices (for example --devices 0,1)"
        )
    selected_devices = None
    if args.devices:
        try:
            selected_devices = [
                int(value) for value in args.devices.split(",")
            ]
        except ValueError:
            parser.error(
                "--devices must contain comma-separated integer indices")
        if len(set(selected_devices)) != len(selected_devices) or any(
                d < 0 for d in selected_devices):
            parser.error(
                "--devices must contain unique non-negative device indices")
    debug_levels = [args.level] if args.level is not None else [1, 2]
    args.stages = [
        expanded for stage in args.stages for expanded in (
            [f"debugger_l{level}"
             for level in debug_levels] if stage == "debugger" else [stage])
    ]
    stage_tools = {
        stage: "debugger" if stage.startswith("debugger_l") else stage
        for stage in args.stages
    }
    requested = ([s.strip() for s in args.ops.split(",")
                  if s.strip()] if args.ops else None)
    cases, excluded = catalog(requested)
    ops = sorted({c["op"] for c in cases})
    canonical_ops = sorted({c["canonical_op"] for c in cases})
    families = sorted({family(op) for op in ops})
    if len(families) < args.min_ops:
        parser.error(
            f"Only {len(families)} operator families, need {args.min_ops}; use --min-ops 1 only for diagnosis"
        )
    if requested and excluded:
        parser.error(
            f"Requested operators have no shared direct case: {excluded}")
    run = args.out.resolve() / datetime.now(
        timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run.mkdir(parents=True)
    manifest = {
        "operator_source":
        str(OPERATORS),
        "operator_source_sha256":
        hashlib.sha256(OPERATORS.read_bytes()).hexdigest(),
        "flagprism_commit":
        git_revision(ROOT),
        "operator_ids":
        ops,
        "canonical_api_ids":
        canonical_ops,
        "operator_count":
        len(ops),
        "canonical_api_count":
        len(canonical_ops),
        "operator_families":
        families,
        "operator_family_count":
        len(families),
        "case_count":
        len(cases),
        "excluded_no_shared_case":
        excluded,
        "stages":
        args.stages,
        "devices":
        args.devices,
        "debug_levels":
        debug_levels if "debugger" in stage_tools.values() else [],
        "debug_addr_level":
        args.addr_level,
        "debug_record_capacity":
        args.record_capacity,
        "launch_policy":
        "fixed_config_no_autotuning",
        "stage_timeouts": {
            stage: args.timeout or (600 if stage == "debugger_l2" else 180)
            for stage in args.stages
        },
        "baseline_timeout":
        args.timeout or 180,
        "jobs":
        args.jobs,
        "cases":
        cases,
        "failure_policy":
        "Retry failed capture without instrumentation: baseline failure is WARNING (accepted), baseline success is ERROR",
        "counting":
        "minimum uses alias/in-place/out-merged IDs; dimension signatures retained; shapes never counted",
    }
    write_json(run / "manifest.json", manifest)
    print(
        f"{len(ops)} operator IDs / {len(families)} families / {len(cases)} cases",
        flush=True,
    )
    print(f'Manifest: {run / "manifest.json"}', flush=True)
    if args.list:
        return 0
    if args.check_cases:
        import torch

        checks = []
        for case in cases:
            try:
                env = cpu_namespace(case, torch)
                exec(compile(case["operation"], "<reference>", "exec"), env)
                assert "result" in env
                torch.manual_seed(0)
                original = {"torch": torch, "device": "cpu"}
                exec(compile(case["body"], "<original-case>", "exec"),
                     original)
                torch.testing.assert_close(
                    to_cpu(env["result"], torch),
                    to_cpu(original["result"], torch),
                    equal_nan=True,
                )
                checks.append({
                    "op": case["op"],
                    "case": case["case_id"],
                    "status": "PASS"
                })
            except Exception as error:
                checks.append({
                    "op": case["op"],
                    "case": case["case_id"],
                    "status": "FAIL",
                    "error": str(error),
                })
        write_json(run / "cpu-cases.json", checks)
        print(dict(Counter(c["status"] for c in checks)))
        return int(any(c["status"] != "PASS" for c in checks))
    unavailable_stages = {}
    for stage in args.stages:
        if stage == "execute":
            continue
        try:
            probe = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import importlib.util; print(bool(importlib.util.find_spec('flagtree') and importlib.util.find_spec('flagtree."
                    + stage_tools[stage] + "')))",
                ],
                capture_output=True,
                text=True,
                timeout=args.timeout or 180,
            )
        except subprocess.TimeoutExpired:
            unavailable_stages[stage] = "Instrumentation preflight timed out"
            continue
        if probe.returncode != 0 or probe.stdout.strip() != "True":
            detail = (probe.stderr or probe.stdout).strip()
            unavailable_stages[stage] = (
                f"flagtree.{stage_tools[stage]} preflight failed: {detail}")
    results = []

    def summarize():
        passed = {}
        lookup = {
            (r["op"], r["case_id"], r["stage"]): r["status"]
            for r in results
        }
        for stage in args.stages:
            passed[stage] = sorted({
                canonical(op)
                for op in ops if all(
                    lookup.get((op, c["case_id"],
                                stage)) in ("PASS", "WARNING") for c in cases
                    if c["op"] == op)
            })
        passed_families = {
            stage: sorted({family(op)
                           for op in ids})
            for stage, ids in passed.items()
        }
        complete = len(results) == len(cases) * len(args.stages)
        ok = (complete
              and all(r["status"] in ("PASS", "WARNING") for r in results)
              and all(
                  len(x) >= args.min_ops for x in passed_families.values()))
        summary = {
            "status":
            "PASS" if ok else ("FAIL" if complete else "RUNNING"),
            "manifest":
            str(run / "manifest.json"),
            "acceptance_policy":
            "PASS and WARNING count as accepted; WARNING means the uninstrumented retry also failed",
            "warning_count":
            sum(r["status"] == "WARNING" for r in results),
            "error_count":
            sum(r["status"] not in ("PASS", "WARNING") for r in results),
            "passed_api_ids":
            passed,
            "passed_operator_families":
            passed_families,
            "counts":
            dict(Counter(r["status"] for r in results)),
            "counts_by_stage": {
                stage:
                dict(
                    Counter(r["status"] for r in results
                            if r["stage"] == stage))
                for stage in args.stages
            },
            "results":
            results,
        }
        write_json(run / "summary.json", summary)
        return ok

    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, Lock
    from queue import Queue

    summary_lock = Lock()
    device_fault = Event()
    device_queue = Queue()
    for value in selected_devices if selected_devices is not None else [None]:
        device_queue.put(value)

    def run_case(index, case, device_id):
        case_dir = (
            run / "cases" /
            f'{index:04d}_{case["op"].replace(".", "_")}_{case["case_id"]}')
        case_file = case_dir / "case.json"
        write_json(case_file, case)

        def run_stage(stage, out):
            timeout = args.timeout or (600 if stage == "debugger_l2" else 180)
            started = time.monotonic()
            tool_stage = "debugger" if stage.startswith(
                "debugger_l") else stage
            level = int(stage[-1]) if tool_stage == "debugger" else 1
            out.mkdir(parents=True)
            result_file = out / "result.json"
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                str(case_file),
                "--stage",
                tool_stage,
                "--result",
                str(result_file),
                "--level",
                str(level),
                "--addr-level",
                str(args.addr_level),
                "--record-capacity",
                str(args.record_capacity),
            ]
            if device_id is not None:
                cmd += ["--device", str(device_id)]
            for option in ("torch_extension", "profiler_backend",
                           "profiler_mode"):
                if getattr(args, option):
                    cmd += [
                        "--" + option.replace("_", "-"),
                        getattr(args, option)
                    ]
            if device_fault.is_set():
                write_json(
                    result_file,
                    {
                        "op": case["op"],
                        "case_id": case["case_id"],
                        "stage": stage,
                        "status": "BLOCKED",
                        "error":
                        "Batch stopped after a device context failure; device health must be checked",
                        "body_sha256": case["body_sha256"],
                    },
                )
                (out / "worker.log").write_text(
                    "Blocked by device context failure in this batch\n")
                code = 1
            elif stage in unavailable_stages:
                write_json(
                    result_file,
                    {
                        "op": case["op"],
                        "case_id": case["case_id"],
                        "stage": stage,
                        "body_sha256": case["body_sha256"],
                        "status": "UNAVAILABLE",
                        "error": unavailable_stages[stage],
                    },
                )
                (out / "worker.log").write_text(unavailable_stages[stage] +
                                                "\n")
                code = 1
            else:
                code = command(cmd, out / "worker.log", timeout)
            data = (json.loads(result_file.read_text())
                    if result_file.exists() else {
                        "op": case["op"],
                        "case_id": case["case_id"],
                        "stage": stage,
                        "status": "FAIL",
                        "error": "Worker exited without a result",
                    })
            # Python exceptions are saved in JSON and need not appear on stderr.
            # Preserve them before timeout/exit handling can replace the message.
            failure_details = "\n".join(
                str(data.get(key, "")) for key in ("error", "traceback"))
            data["stage"] = stage
            if tool_stage == "debugger":
                data["debug_level"] = level
            data.update(exit_code=code,
                        log=str(out / "worker.log"),
                        device_index=device_id)
            data.update(
                timeout_seconds=timeout,
                elapsed_seconds=round(time.monotonic() - started, 3),
            )
            if code == 124:
                data.update(status="TIMEOUT", error=f"Exceeded {timeout}s")
            if code != 0 and data["status"] == "PASS":
                data.update(status="FAIL",
                            error="Worker did not exit successfully")
            if code != 0:
                log_text = (out / "worker.log").read_text(errors="replace")
                failure_details += "\n" + log_text
                if any(marker in failure_details.lower() for marker in (
                        "detected context error",
                        "sip exception",
                        "receive sip error",
                        "device is out of service",
                        "topserrorinvaliddevice",
                        "printf_display detected the buffer is corrupted",
                        "npu function error",
                        "vector core exception",
                        "illegal memory access",
                        "device-side assert",
                )):
                    device_fault.set()
                    data["device_fault"] = True
            data["body_sha256"] = case["body_sha256"]
            write_json(result_file, data)
            return data

        for stage in args.stages:
            out = case_dir / stage
            data = run_stage(stage, out)
            if stage != "execute" and data["status"] not in ("PASS",
                                                             "BLOCKED"):
                data["capture_status"] = data["status"]
                if data["status"] in ("UNAVAILABLE", "SETUP_ERROR"):
                    data.update(
                        status="ERROR",
                        severity="error",
                        diagnosis="Required tooling or runtime setup failed")
                elif device_fault.is_set():
                    # A failed retry on a poisoned context cannot establish operator support.
                    data.update(
                        status="ERROR",
                        severity="error",
                        diagnosis=
                        "Device context failure; baseline comparison unavailable",
                    )
                else:
                    baseline = run_stage("execute", out / "baseline")
                    data["baseline_result"] = baseline
                    if baseline["status"] in ("UNAVAILABLE", "SETUP_ERROR"):
                        data.update(
                            status="ERROR",
                            severity="error",
                            diagnosis=
                            "Baseline runtime setup failed; operator support cannot be determined"
                        )
                    elif baseline["status"] == "PASS":
                        data.update(
                            status="ERROR",
                            severity="error",
                            diagnosis=
                            "Uninstrumented execution passed; capture failed",
                        )
                    elif device_fault.is_set(
                    ) or baseline["status"] == "BLOCKED" or baseline.get(
                            "device_fault"):
                        data.update(
                            status="ERROR",
                            severity="error",
                            diagnosis=
                            "Device context failure prevented a valid baseline comparison",
                        )
                    else:
                        data.update(
                            status="WARNING",
                            severity="warning",
                            diagnosis=
                            "Uninstrumented execution also failed; accepted by policy",
                        )
            data["accepted"] = data["status"] in ("PASS", "WARNING")
            write_json(out / "result.json", data)
            with summary_lock:
                results.append(data)
                summarize()
            print(
                f'[{index + 1}/{len(cases)}] {case["op"]}/{case["case_id"]} {stage}: {data["status"]}',
                flush=True,
            )

    def scheduled_case(item):
        device_id = device_queue.get()
        try:
            return run_case(*item, device_id)
        finally:
            device_queue.put(device_id)

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(scheduled_case, enumerate(cases)))
    ok = summarize()
    print(f'Summary: {run / "summary.json"}', flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
