import argparse
import sys
import os
from pathlib import Path
from .profile import start, finalize, _select_backend
from .flags import set_command_line
from .tianshu import merge_ixkn_vendor_artifact, run_ixkn_profile


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="FlagTree Profiler command utility for scripts and pytest tests.", usage="""
    flagtree-profiler [options] script.py [script_args] [script_options]
    flagtree-profiler [options] pytest [pytest_args] [script_options]
    python -m flagtree.profiler.cli [options] script.py [script_args] [script_options]
""", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-n", "--name", type=str, help="Name of the profiling session")
    parser.add_argument("-b", "--backend", type=str, help="Profiling backend", default=None,
                        choices=["cupti", "cupti_pcsampling", "roctracer", "instrumentation", "cann", "tianshu", "corex", "iluvatar"])
    parser.add_argument("-c", "--context", type=str, help="Profiling context", default="shadow",
                        choices=["shadow", "python"])
    parser.add_argument("-m", "--mode", type=str, help="Profiling mode", default=None)
    parser.add_argument("-d", "--data", type=str, help="Profiling data", default="tree", choices=["tree", "trace"])
    parser.add_argument("-k", "--hook", type=str, help="Profiling hook", default=None,
                        choices=["triton", "instrumentation"])
    parser.add_argument("--ixkn", action="store_true",
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
    parser.add_argument('target_args', nargs=argparse.REMAINDER, help='Subcommand and its arguments')
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

    if args.ixkn:
        if backend not in {"tianshu", "corex", "iluvatar"}:
            raise ValueError("--ixkn is only valid with the tianshu backend")
        if not target_args:
            raise ValueError("--ixkn requires a target script or pytest command")

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
        child_env = {**os.environ, "FLAGTREE_PROFILER_TIANSHU_IMPORT_PATH": export_profile}
        inherited_pythonpath = [item for item in child_env.get("PYTHONPATH", "").split(os.pathsep) if item]
        for item in sys.path:
            if item and ("site-packages" in item or "dist-packages" in item) and item not in inherited_pythonpath:
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
