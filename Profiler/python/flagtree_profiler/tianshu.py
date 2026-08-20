"""Tianshu/CoreX profiling helpers.

ixKN is a process wrapper rather than an in-process start/stop API. These
helpers keep that lifecycle explicit and provide a small CSV normalizer for
the vendor artifact produced by FlagTree.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence


def find_ixkn_cli(explicit: str | None = None) -> str:
    candidates = [
        explicit,
        os.getenv("FLAGTREE_PROFILER_TIANSHU_IXKN_CLI"),
        os.getenv("IXKN_CLI"),
        "ixkn-cli",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
        if Path(candidate).is_file():
            return str(Path(candidate))
    raise FileNotFoundError(
        "ixkn-cli was not found; install the Tianshu CoreX tools or set "
        "FLAGTREE_PROFILER_TIANSHU_IXKN_CLI")


def build_ixkn_command(
    target: Sequence[str],
    *,
    devices: str = "0",
    sections: str = "all",
    kernel_name: str | None = None,
    launch_count: int | None = None,
    launch_skip: int | None = None,
    export_profile: str | None = None,
    csv_output: bool = True,
    profile_child_processes: bool = False,
    ixkn_cli: str | None = None,
) -> list[str]:
    command = [find_ixkn_cli(ixkn_cli)]
    if devices:
        command.extend(["--devices", str(devices)])
    if sections:
        command.extend(["--section", str(sections)])
    if kernel_name:
        command.extend(["--kernel-name", str(kernel_name)])
    if launch_count is not None:
        command.extend(["--launch-count", str(launch_count)])
    if launch_skip is not None:
        command.extend(["--launch-skip", str(launch_skip)])
    if export_profile:
        command.extend(
            ["--export-profile",
             str(export_profile), "--force-overwrite"])
    if csv_output:
        command.append("--csv")
    if profile_child_processes:
        command.append("--profile-child-processes")
    command.extend(str(item) for item in target)
    return command


def run_ixkn_profile(
    target: Sequence[str],
    *,
    devices: str = "0",
    sections: str = "all",
    kernel_name: str | None = None,
    launch_count: int | None = None,
    launch_skip: int | None = None,
    export_profile: str | None = None,
    csv_output: bool = True,
    profile_child_processes: bool = False,
    ixkn_cli: str | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    command = build_ixkn_command(
        target,
        devices=devices,
        sections=sections,
        kernel_name=kernel_name,
        launch_count=launch_count,
        launch_skip=launch_skip,
        export_profile=export_profile,
        csv_output=csv_output,
        profile_child_processes=profile_child_processes,
        ixkn_cli=ixkn_cli,
    )
    if not csv_output:
        return subprocess.run(command, check=False, text=True, env=env)

    # ixKN 4.4 emits CSV rows on stdout. Capture that stream so the result can
    # be consumed by the vendor importer instead of being lost in the console.
    result = subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )
    stdout = result.stdout or ""
    csv_lines = stdout.splitlines()
    header_index = next(
        (index for index, line in enumerate(csv_lines)
         if line.startswith('"Kernel ID"')),
        None,
    )
    if export_profile and header_index is not None:
        csv_path = Path(export_profile).with_suffix(".csv")
        csv_path.write_text(
            "\n".join(csv_lines[header_index:]) + "\n",
            encoding="utf-8",
        )
        console_lines = csv_lines[:header_index]
    else:
        console_lines = csv_lines
    if console_lines:
        sys.stdout.write("\n".join(console_lines) + "\n")
    if result.stderr:
        sys.stderr.write(result.stderr)

    # With --export-profile ixKN writes only its binary database during the
    # live run. Re-open that database to obtain the structured CSV report.
    if export_profile and result.returncode == 0 and header_index is None:
        import_command = [
            find_ixkn_cli(ixkn_cli), "--import-profile",
            str(export_profile)
        ]
        if sections:
            import_command.extend(["--section", str(sections)])
        import_command.append("--csv")
        imported = subprocess.run(
            import_command,
            check=False,
            text=True,
            capture_output=True,
            env=env,
        )
        imported_lines = (imported.stdout or "").splitlines()
        imported_header = next(
            (index for index, line in enumerate(imported_lines)
             if line.startswith('"Kernel ID"')),
            None,
        )
        if imported_header is not None:
            Path(export_profile).with_suffix(".csv").write_text(
                "\n".join(imported_lines[imported_header:]) + "\n",
                encoding="utf-8",
            )
        if imported.stderr:
            sys.stderr.write(imported.stderr)
    return result


def _normalize(value: str) -> str:
    return "".join(ch.lower() for ch in value.strip() if ch.isalnum())


def _first(row: dict[str, str], names: Iterable[str]) -> str:
    normalized = {_normalize(key): value for key, value in row.items()}
    for name in names:
        value = normalized.get(_normalize(name), "")
        if value.strip():
            return value.strip()
    return ""


def _number(value: str) -> int | float | None:
    if not value.strip():
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return int(number) if number.is_integer() else number


def _time_ns(value: str, header: str) -> int:
    number = _number(value)
    if number is None:
        return 0
    name = _normalize(header)
    if "ns" in name:
        return int(number)
    if "ms" in name:
        return int(number * 1_000_000)
    return int(number * 1_000)


def _find_csv_files(root: Path) -> list[Path]:
    root = root.expanduser()
    if root.is_file():
        if root.suffix.lower() == ".csv":
            return [root]
        sibling = root.with_suffix(".csv")
        return [sibling] if sibling.exists() else []
    if root.is_dir():
        return sorted(path for path in root.rglob("*.csv") if path.is_file())

    # ixKN accepts a profile basename, while the CSV is emitted as a sibling
    # file. Resolve that basename before deciding that no vendor data exists.
    candidates = [root.with_suffix(".csv")]
    if root.suffix.lower() == ".ixkn":
        candidates.append(root.with_suffix(".ixkn").with_suffix(".csv"))
    return [path for path in dict.fromkeys(candidates) if path.is_file()]


def _find_ixkn_profile(root: Path) -> Path | None:
    root = root.expanduser()
    if root.is_file() and root.suffix.lower() == ".ixkn":
        return root
    if root.is_dir():
        return None
    candidates = [root.with_suffix(".ixkn")]
    if root.suffix.lower() == ".ixkn":
        candidates.insert(0, root)
    return next((path for path in dict.fromkeys(candidates) if path.is_file()),
                None)


def _parse_csv(path: Path) -> list[dict]:
    associations = []
    with path.open(newline="", encoding="utf-8", errors="replace") as stream:
        for row in csv.DictReader(stream):
            name = _first(
                row,
                ("kernel_name", "kernelname", "op_name", "opname", "name"))
            start_header = next(
                (key for key in row if _normalize(key) in
                 {"starttimens", "starttimeus", "starttime", "start"}),
                "start_us",
            )
            end_header = next(
                (key for key in row if _normalize(key) in
                 {"endtimens", "endtimeus", "endtime", "end"}),
                "end_us",
            )
            duration_header = next(
                (key for key in row if _normalize(key) in
                 {"durationns", "durationus", "durationms", "duration"}),
                "",
            )
            start = _time_ns(_first(row, (start_header, )), start_header)
            end = _time_ns(_first(row, (end_header, )), end_header)
            if not end and duration_header:
                end = start + _time_ns(row.get(duration_header, ""),
                                       duration_header)

            metrics: dict[str, int | float | str] = {"ixkn_file": str(path)}
            metric_label = _first(row, ("metrics", "metric"))
            metric_value = _first(row, ("value", ))
            if metric_label and metric_label != "INF" and metric_value:
                numeric_value = _number(metric_value)
                metrics[f"tianshu.{_normalize(metric_label)}"] = (
                    numeric_value
                    if numeric_value is not None else metric_value)
            for header, value in row.items():
                if _normalize(header) in {
                        _normalize(start_header),
                        _normalize(end_header),
                        _normalize(duration_header), "metrics", "metric",
                        "value"
                }:
                    continue
                number = _number(value or "")
                if number is not None:
                    metrics[f"tianshu.{_normalize(header)}"] = number
            associations.append({
                "runtime_event": {
                    "scope_id":
                    0,
                    "op_name":
                    name,
                    "task_id":
                    int(
                        _number(_first(row,
                                       ("task_id", "taskid", "kernel_id")))
                        or 0),
                    "correlation_id":
                    int(
                        _number(
                            _first(
                                row,
                                ("correlation_id", "correlationid", "corrid")))
                        or 0),
                    "device_id":
                    int(
                        _number(
                            _first(row, ("device_id", "deviceid", "device")))
                        or 0),
                    "stream_id":
                    int(
                        _number(
                            _first(row, ("stream_id", "streamid", "stream")))
                        or 0),
                    "start_time_ns":
                    start,
                    "end_time_ns":
                    end,
                },
                "state": "collected",
                "source": "ixkn_csv",
                "note":
                "imported from ixKN CSV; runtime correlation is deferred to native importer",
                "metrics": metrics,
            })
    return associations


def merge_ixkn_vendor_artifact(name: str, export_profile: str) -> Path | None:
    """Merge post-process ixKN CSV output into an existing vendor artifact."""
    vendor_path = Path(name).with_suffix(".vendor.json")
    if not vendor_path.exists():
        return None
    artifact = json.loads(vendor_path.read_text(encoding="utf-8"))
    root = Path(export_profile)
    files = _find_csv_files(root)
    ixkn_profile = _find_ixkn_profile(root)
    raw_inputs = artifact.setdefault("raw_inputs", [])
    associations = artifact.setdefault("associations", [])
    existing = {
        str(item.get("metrics", {}).get("ixkn_file", ""))
        for item in associations if isinstance(item, dict)
    }
    for path in files:
        if str(path) not in raw_inputs:
            raw_inputs.append(str(path))
        if str(path) in existing:
            continue
        associations.extend(_parse_csv(path))
    if files and associations:
        enabled_metrics = artifact.setdefault("enabled_metrics", [])
        for association in associations:
            for metric_name in (association.get("metrics") or {}):
                if metric_name.startswith(
                        "tianshu.") and metric_name not in enabled_metrics:
                    enabled_metrics.append(metric_name)
        reasons = artifact.setdefault("degrade_reasons", [])
        artifact["degrade_reasons"] = [
            reason for reason in reasons
            if "No Tianshu ixKN profiling associations could be imported."
            not in str(reason)
            and "use --csv for structured vendor import" not in str(reason)
        ]
        by_source: dict[str, int] = {}
        by_state: dict[str, int] = {}
        timed = 0
        for association in associations:
            by_source[str(association.get(
                "source", "unknown"))] = by_source.get(
                    str(association.get("source", "unknown")), 0) + 1
            by_state[str(association.get("state", "unknown"))] = by_state.get(
                str(association.get("state", "unknown")), 0) + 1
            event = association.get("runtime_event") or {}
            if int(event.get("end_time_ns",
                             0)) > int(event.get("start_time_ns", 0)):
                timed += 1
        summary = artifact.setdefault("summary", {})
        summary.update({
            "association_count": len(associations),
            "counts_by_source": by_source,
            "counts_by_state": by_state,
            "timed_association_count": timed,
            "raw_input_count": len(raw_inputs),
        })
    if ixkn_profile is not None:
        if str(ixkn_profile) not in raw_inputs:
            raw_inputs.append(str(ixkn_profile))
        if not files:
            artifact.setdefault("degrade_reasons", []).append(
                "ixKN binary export retained; use --csv for structured vendor import"
            )
    vendor_path.write_text(
        json.dumps(artifact, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return vendor_path
