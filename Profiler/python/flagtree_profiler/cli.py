import argparse
import sys
import os
from pathlib import Path
from .profile import start, finalize, _select_backend
from .flags import set_command_line
from .mthreads import (DEFAULT_MCU_SECTIONS, MCU_INTEGRATION_ENABLED,
                       merge_mcu_vendor_artifact, run_mcu_profile)
from .tianshu import merge_ixkn_vendor_artifact, run_ixkn_profile


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=
        "FlagTree Profiler command utility for scripts and pytest tests.",
        usage="""
    flagtree-profiler [options] script.py [script_args] [script_options]
    flagtree-profiler [options] pytest [pytest_args] [script_options]
    python -m flagtree.profiler.cli [options] script.py [script_args] [script_options]
""",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-n",
                        "--name",
                        type=str,
                        help="Name of the profiling session")
    parser.add_argument(
        "-b",
        "--backend",
        type=str,
        help="Profiling backend",
        default=None,
        choices=[
            # FlagPrism: expose the NVIDIA adapter through
            # the command-line interface as well as the API.
            "nvidia",
            "cuda",
            "cupti",
            "cupti_pcsampling",
            "roctracer",
            "instrumentation",
            "cann",
            "mthreads",
            "musa",
            "tianshu",
            "corex",
            "iluvatar",
            "enflame",
            "gcu",
            "tops"
        ])
    parser.add_argument("-c",
                        "--context",
                        type=str,
                        help="Profiling context",
                        default="shadow",
                        choices=["shadow", "python"])
    parser.add_argument("-m",
                        "--mode",
                        type=str,
                        help="Profiling mode",
                        default=None)
    parser.add_argument("-d",
                        "--data",
                        type=str,
                        help="Profiling data",
                        default="tree",
                        choices=["tree", "trace"])
    parser.add_argument("-k",
                        "--hook",
                        type=str,
                        help="Profiling hook",
                        default=None,
                        choices=["triton", "instrumentation"])
    parser.add_argument("--ixkn",
                        action="store_true",
                        help="Wrap the target process with Tianshu ixKN")
    parser.add_argument("--ixkn-cli", type=str, default=None)
    parser.add_argument("--ixkn-devices", type=str, default="0")
    parser.add_argument("--ixkn-section", type=str, default="all")
    parser.add_argument("--ixkn-kernel-name", type=str, default=None)
    parser.add_argument("--ixkn-launch-count", type=int, default=None)
    parser.add_argument("--ixkn-launch-skip", type=int, default=None)
    parser.add_argument("--ixkn-export-profile", type=str, default=None)
    parser.add_argument("--no-ixkn-csv", action="store_true")
    parser.add_argument("--ixkn-profile-child-processes", action="store_true")
    # TODO(FlagPrism): Expose the MCU CLI after validation on a compatible
    # Moore Threads MCU, MUSA SDK, and driver test environment.
    if MCU_INTEGRATION_ENABLED:
        parser.add_argument(
            "--mcu",
            action="store_true",
            help="Wrap the target process with Moore Perf Compute")
        parser.add_argument("--mcu-cli", type=str, default=None)
        parser.add_argument("--mcu-devices", type=str, default="0")
        parser.add_argument("--mcu-sections",
                            type=str,
                            default=DEFAULT_MCU_SECTIONS)
        parser.add_argument("--mcu-metrics", type=str, default=None)
        parser.add_argument("--mcu-kernel-name", type=str, default=None)
        parser.add_argument("--mcu-launch-count", type=int, default=None)
        parser.add_argument("--mcu-launch-skip", type=int, default=None)
        parser.add_argument("--mcu-output", type=str, default=None)
        parser.add_argument(
            "--mcu-import-csv",
            type=str,
            default=None,
            help=
            "Structured MCU CSV file or directory to merge after collection",
        )
    parser.add_argument('target_args',
                        nargs=argparse.REMAINDER,
                        help='Subcommand and its arguments')
    args = parser.parse_args()
    return args, args.target_args


def is_pytest(script):
    return os.path.basename(script) == 'pytest'


def execute_as_main(script, args):
    script_path = os.path.abspath(script)
    # Prepare a clean global environment
    clean_globals = {
        "__name__": "__main__",
        "__file__": script_path,
        "__builtins__": __builtins__,
        sys.__name__: sys,
    }

    original_argv = sys.argv
    sys.argv = [script] + args
    # Append the script's directory in case the script uses relative imports
    sys.path.append(os.path.dirname(script_path))

    # Execute in the isolated environment
    try:
        with open(script_path, 'rb') as file:
            code = compile(file.read(), script_path, 'exec')
        exec(code, clean_globals)
    except Exception as e:
        print(f"An error occurred while executing the script: {e}")
        sys.exit(1)
    finally:
        sys.argv = original_argv


def do_setup_and_execute(target_args):
    # Set the command line mode to avoid any `start` calls in the script.
    set_command_line()

    script = target_args[0]
    script_args = target_args[1:] if len(target_args) > 1 else []
    if is_pytest(script):
        import pytest
        pytest.main(script_args)
    else:
        execute_as_main(script, script_args)


def run_profiling(args, target_args):
    backend = args.backend if args.backend else _select_backend()

    if getattr(args, "mcu", False):
        if not MCU_INTEGRATION_ENABLED:
            raise RuntimeError(
                "Moore Perf Compute integration is frozen until it can be "
                "validated on a compatible Moore Threads environment")
        if backend not in {"mthreads", "musa"}:
            raise ValueError("--mcu is only valid with the mthreads backend")
        if not target_args:
            raise ValueError(
                "--mcu requires a target script or pytest command")

        name = args.name or "flagtree_profiler"
        report_path = args.mcu_output or str(
            Path(name).with_suffix(".mcu-rep"))
        log_path = str(Path(name).with_suffix(".mcu.log"))
        child_mode = args.mode or (
            "runtime_base:vendor_metrics=launch_stats,occupancy,resource_usage,"
            "peak_memory_bandwidth,instruction_count,cycles,memory_bandwidth,"
            "sm_utilization,hardware_counters")
        if "mcu_external=" not in child_mode:
            child_mode += ":mcu_external=true"
        child_command = [
            sys.executable,
            "-m",
            f"{__package__}.cli",
            "--backend",
            "mthreads",
            "--name",
            name,
            "--context",
            args.context,
            "--data",
            args.data,
            "--mode",
            child_mode,
        ]
        if args.hook:
            child_command.extend(["--hook", args.hook])
        child_command.extend(target_args)
        result = run_mcu_profile(
            child_command,
            devices=args.mcu_devices,
            sections=args.mcu_sections,
            metrics=args.mcu_metrics,
            kernel_name=args.mcu_kernel_name,
            launch_count=args.mcu_launch_count,
            launch_skip=args.mcu_launch_skip,
            output=report_path,
            mcu_cli=args.mcu_cli,
            env=os.environ.copy(),
            log_path=log_path,
        )
        merge_mcu_vendor_artifact(
            name,
            report_path,
            csv_import_path=args.mcu_import_csv,
            log_path=log_path,
        )
        if result.returncode != 0:
            raise SystemExit(result.returncode)
        return

    if args.ixkn:
        if backend not in {"tianshu", "corex", "iluvatar"}:
            raise ValueError("--ixkn is only valid with the tianshu backend")
        if not target_args:
            raise ValueError(
                "--ixkn requires a target script or pytest command")

        name = args.name or "flagtree_profiler"
        export_profile = args.ixkn_export_profile
        if not export_profile:
            export_profile = str(Path(name).with_suffix(".ixkn"))
        child_mode = args.mode or ""
        import_token = f"ixkn_import_path={export_profile}"
        child_mode = f"{child_mode}:{import_token}" if child_mode else import_token
        child_command = [
            sys.executable,
            "-m",
            f"{__package__}.cli",
            "--backend",
            "tianshu",
            "--name",
            name,
            "--context",
            args.context,
            "--data",
            args.data,
            "--mode",
            child_mode,
        ]
        if args.hook:
            child_command.extend(["--hook", args.hook])
        child_command.extend(target_args)
        # ixKN resolves Python virtualenv symlinks before launching the target.
        # Preserve the active interpreter's site-packages explicitly so the
        # wrapped process keeps access to torch and the selected Triton build.
        child_env = {
            **os.environ, "FLAGTREE_PROFILER_TIANSHU_IMPORT_PATH":
            export_profile
        }
        inherited_pythonpath = [
            item for item in child_env.get("PYTHONPATH", "").split(os.pathsep)
            if item
        ]
        for item in sys.path:
            if item and ("site-packages" in item or "dist-packages"
                         in item) and item not in inherited_pythonpath:
                inherited_pythonpath.append(item)
        if inherited_pythonpath:
            child_env["PYTHONPATH"] = os.pathsep.join(inherited_pythonpath)
        result = run_ixkn_profile(
            child_command,
            devices=args.ixkn_devices,
            sections=args.ixkn_section,
            kernel_name=args.ixkn_kernel_name,
            launch_count=args.ixkn_launch_count,
            launch_skip=args.ixkn_launch_skip,
            export_profile=export_profile,
            csv_output=not args.no_ixkn_csv,
            profile_child_processes=args.ixkn_profile_child_processes,
            ixkn_cli=args.ixkn_cli,
            env=child_env,
        )
        merge_ixkn_vendor_artifact(name, export_profile)
        if result.returncode != 0:
            raise SystemExit(result.returncode)
        return

    start(
        args.name,
        context=args.context,
        data=args.data,
        backend=backend,
        mode=args.mode,
        hook=args.hook,
    )

    do_setup_and_execute(target_args)

    finalize()


def main():
    args, target_args = parse_arguments()
    run_profiling(args, target_args)


if __name__ == "__main__":
    main()
