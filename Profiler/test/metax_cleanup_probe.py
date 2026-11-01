# FlagPrism：a failed vendor cleanup must be reported without breaking the next session.
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap


def run_probe(tmp_path):
    import triton
    if triton.runtime.driver.active.get_current_target().backend != "maca":
        raise RuntimeError("MetaX device required")
    compiler = shutil.which("c++")
    sdk = Path(os.getenv("MACA_PATH", "/opt/maca"))
    if not compiler or not (sdk / "include/mcpti/mcpti.h").exists():
        raise RuntimeError(
            "C++ compiler and MCPTI SDK required for fault injection")
    source = tmp_path / "cleanup.cpp"
    source.write_text(
        textwrap.dedent('''
        #include <mcpti/mcpti.h>
        #include <dlfcn.h>
        extern "C" MCptiResult MCPTIAPI mcptiEventGroupSetsDestroy(MCpti_EventGroupSets *groups) {
          using Destroy = MCptiResult (*)(MCpti_EventGroupSets *);
          auto real = reinterpret_cast<Destroy>(dlsym(RTLD_NEXT, "mcptiEventGroupSetsDestroy"));
          if (!real) return MCPTI_ERROR_UNKNOWN;
          auto result = real(groups);
          static bool injected = false;
          if (result == MCPTI_SUCCESS && !injected) {
            injected = true;
            return MCPTI_ERROR_UNKNOWN; // Actual cleanup succeeded; inject its reported error.
          }
          return result;
        }
    '''))
    library = tmp_path / "cleanup.so"
    subprocess.run([
        compiler, "-shared", "-fPIC",
        str(source), "-I" + str(sdk / "include"),
        "-I" + str(sdk / "include/mcr"), "-I" + str(sdk / "include/mc"),
        "-L" + str(sdk / "lib"), "-Wl,--no-as-needed", "-lmcpti",
        "-Wl,-rpath," + str(sdk / "lib"), "-ldl", "-o",
        str(library)
    ],
                   check=True,
                   capture_output=True,
                   text=True)
    script = tmp_path / "probe.py"
    script.write_text(
        textwrap.dedent('''
        import json
        import sys
        from pathlib import Path
        import torch
        from flagtree import profiler
        base = Path(sys.argv[1])
        x = torch.ones(65536, device="cuda")
        y = torch.empty_like(x)
        for index in range(2):
            path = base / str(index)
            session = profiler.start(str(path), backend="metax",
                                     mode="runtime_base:vendor_metrics=inst_executed")
            try:
                for _ in range(20):
                    torch.add(x, 1, out=y)
            finally:
                profiler.finalize(session)
            report = json.loads(path.with_suffix(".vendor.json").read_text())
            if index == 0:
                assert any("mcptiEventGroupSetsDestroy failed: status 999" in reason
                           for reason in report["degrade_reasons"]), report
                assert "inst_executed" not in report["enabled_metrics"]
            else:
                assert not report["degrade_reasons"], report
                assert "inst_executed" in report["enabled_metrics"]
                ranges = [a for a in report["associations"] if a["source"] == "mcpti_hardware_counter"]
                assert ranges and ranges[0]["metrics"]["metax.hardware.inst_executed"] > 0
        print("cleanup error reported; next session recovered")
    '''))
    env = os.environ.copy()
    env["LD_PRELOAD"] = str(library) + (":" + env["LD_PRELOAD"]
                                        if env.get("LD_PRELOAD") else "")
    result = subprocess.run(
        [sys.executable, str(script),
         str(tmp_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "next session recovered" in result.stdout


# FlagPrism end

# FlagPrism：MACA keeps the perf-counter channel attached to the process after use.
# Run this probe separately, before any other process acquires this GPU's counters.
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run_probe(args.output)
    print("PASS: cleanup error reported and following session recovered")
# FlagPrism end
