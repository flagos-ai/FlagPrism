# Tianshu/CoreX Profiler

FlagPrism exposes Tianshu as `backend="tianshu"` and accepts `corex` and
`iluvatar` as backend aliases. The implementation has two layers:

1. The IR path reuses FlagPrism Debugger records for in-process summary and
   memory collection.
2. The vendor path invokes ixKN outside the target process and imports its CSV
   export into `.vendor.json`.

## In-process IR collection

```python
import flagtree.profiler as profiler

sid = profiler.start(
    "tianshu_run",
    backend="tianshu",
    hook="triton",
    mode="runtime_base",
)
# launch Triton kernels
profiler.finalize(sid)
```

The CoreX transfer adapter loads the CUDA-compatible driver lazily. Set
`FLAGTREE_DEBUGGER_COREX_DRIVER_LIBRARY` when the driver is not available as
`libcuda.so.1` or is installed outside `/usr/local/corex/lib`.

The hidden debug control pointer is passed only to kernels that contain dynamic
records. The default Tianshu IR path records address summaries so it can report
per-op read/write bytes. Device-cycle timeline is disabled until a Tianshu
device timestamp instruction is confirmed; the in-process timeline therefore
uses synchronized host launch bounds and labels them as host timing. Use ixKN
for hardware duration, occupancy, instruction, and memory-throughput metrics.

## ixKN vendor collection

The documented ixKN workflow wraps the target process from startup:

```bash
flagtree-profiler \
  --backend tianshu \
  --ixkn \
  --ixkn-devices 0 \
  --ixkn-section LaunchStats,Occupancy,Instruction,Memory \
  --ixkn-kernel-name _vector_add_kernel \
  --ixkn-launch-count 1 \
  --ixkn-export-profile /tmp/tianshu_ixkn \
  --name /tmp/tianshu_run \
  workload.py
```

The wrapper uses `ixkn-cli`, requests CSV export, and merges CSV rows into the
vendor artifact after the wrapped process exits. For programmatic use, import
`flagtree.profiler.tianshu.build_ixkn_command` or
`run_ixkn_profile`.

Use `--ixkn-kernel-name` when the workload performs framework initialization
or allocation kernels before the Triton launch. The value is passed to ixKN's
kernel-name regex filter, so `--ixkn-launch-count` then counts only matching
kernels.

Supported normalized vendor metric names are:

```text
launch_stats
occupancy
instruction
memory
```

An `.ixkn` binary is retained as a raw input, but structured association needs
the `--csv` output. Missing or unmatched correlations are recorded in
`degrade_reasons` or with `state="unmatched"` rather than inferred silently.
