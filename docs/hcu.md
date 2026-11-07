# Hygon HCU

The HCU integration uses FlagTree's `hcu` compiler backend, the DTK HIP runtime
for debugger transfers, and ROCtracer for device kernel timings. Triton's
runtime target is named `hip`. The tested configuration is BW1101 (`gfx938`,
warp size 64), DTK 26.04, PyTorch 2.10 and Triton 3.6.

## Build

Use the development image and device mounts in the
[FlagTree HCU manual](https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-hcu).
Both repositories need their HCU integration changes. Load DTK in
non-interactive shells as well as interactive ones:

```bash
source /opt/dtk/env.sh
export FLAGTREE_BACKEND=hcu
export FLAGPRISM_BACKEND=hcu
export FLAGPRISM_SOURCE_DIR=/path/to/FlagPrism
export TRITON_BUILD_FLAGPRISM=ON
export TRITON_BUILD_PROTON=OFF
cd /path/to/FlagTree
MAX_JOBS=32 python3 -m pip install . --no-build-isolation -v
```

Use the LLVM package specified by the HCU manual. FlagTree can download it;
`LLVM_SYSPATH` can select an extracted copy.

## Debugger

Instrumented kernels receive a trailing device control pointer. Ordinary
launches retain their original arguments. The HIP runtime selects the launch
device and synchronizes the launch stream before decoding L1 summaries or
L2 value/address records. Missing HIP support fails explicitly.

Kernel-internal instruction timelines are not enabled: stable clock-to-nanosecond
conversion and cross-CU clock alignment have not been established.

## Profiler

The HCU driver automatically selects `backend="hcu"` (alias `hygon`). The legacy
`backend="roctracer"` remains available, but cannot overlap an HCU session because
both collectors use ROCtracer's default pool. The legacy backend rejects
`HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES`.

```python
from flagtree import profiler

session = profiler.start(
    "profile", backend="hcu", hook="triton",
    mode=("runtime_base:vendor_metrics=launch_stats,resource_usage,"
          "kernel_duration,theoretical_occupancy:runtime_events=true"),
)
# Launch kernels here.
profiler.finalize(session)
```

Reports use `.hatchet`, `.timeline.json`, `.meta.json` and `.vendor.json`.
Kernel durations come from device activity timestamps. Session membership is
captured at API entry, excluding paused work even when buffers arrive later.
Overlapping HCU sessions are rejected. Flush synchronizes observed devices and
restores the caller's device.

| Option | Information and limits |
| --- | --- |
| `launch_stats` | Grid/block dimensions for module launches. |
| `resource_usage` | HIP function queries for registers, static shared memory and local memory. |
| `theoretical_occupancy` | HIP module occupancy query using the actual block size and dynamic shared memory. A resource bound, not achieved occupancy or GPU utilization. |
| `runtime_events=true` | Device memory activities and separately labeled host HIP API intervals; API time is excluded from kernel totals. |
| `hook="triton"` or `hook="instrumentation"` | Debugger IR operation counts and logical read/write bytes, not hardware DRAM traffic. Adds instrumentation overhead; use `hook=None` for activity-only timing. |

Missing or unloaded function metadata, including graph replay, produces an
explicit resource/occupancy note. Resource/occupancy metrics without any available
measurements are removed from the enabled list. Unsupported optional metrics have degradation
reasons; `require_metrics=true` rejects unsupported or wholly unavailable metrics.

For `hipMemcpy[Async]` and `hipMemset[Async]`, byte counts come from correlated
API parameters: DTK's copy-record union does not reliably contain lengths.
Other memory activity forms retain their name and interval without an invented
byte count. Invalid kernel timestamps and callback/decoder errors fail explicitly;
a dropped-record query has not been validated for this DTK integration.

## Hardware counters

```bash
flagtree-profiler -b hcu --rocprof \
  --rocprof-counters SQ_WAVES,SQ_INSTS_VALU,SQ_INSTS_SALU \
  --name counters workload.py
```

This runs DTK ROCProfiler with one PMC group and one application execution,
then imports per-dispatch CSV into the four report formats. Raw data and logs
are retained as `.rocprof.csv` and `.rocprof.log`. Invalid or empty captures and
failed applications are errors. Use `--rocprof-cli` for another installation.

Validated metrics include wavefront count (`SQ_WAVES`), VALU/SALU instruction
counts and `FetchSize`/`WriteSize` in SDK kilobyte units. The tested DTK/device
cannot collect `FetchSize` and `WriteSize` in the same PMC group; request each
in a separate explicit application run:

```bash
flagtree-profiler -b hcu --rocprof --rocprof-counters FetchSize -n reads workload.py
flagtree-profiler -b hcu --rocprof --rocprof-counters WriteSize -n writes workload.py
```

Other installed metric identifiers can be requested, subject to SDK/device
availability and group limits. Reports retain vendor names and units. VALU
counts are not per-thread instruction counts. CSV device IDs remain ROCProfiler
HSA-agent indices, not HIP ordinals, and are labeled accordingly.

Hardware profiling may serialize dispatches and perturb timings. It is separate
from in-process start/stop, cannot combine with `--hook` or `--mode`, and does not
automatically replay the application for additional groups.

## Validation

Run the shared operator suite from the FlagPrism checkout:

```bash
python3 test.py --jobs 8 --devices 0,1,2,3,4,5,6,7
```

Use devices available on the host. Coverage, failure policy and output formats
are defined in [Testing](TESTING.md); actual results belong in the generated
`manifest.json` and `summary.json`.

The following regressions additionally check L2 values/addresses across dtypes,
masked streams, non-finite summaries, buffer overflow/recovery, launch ABI,
block pointers, session lifecycle, threaded/multi-device attribution, graph
replay, SDK occupancy agreement, memory/API activities and hardware counters:

```bash
python3 -m pytest -q \
  Debugger/test/python/runtime/test_hcu_debugger_e2e.py \
  Profiler/test/test_hcu.py Profiler/test/test_hcu_extended.py \
  Profiler/test/test_hcu_occupancy.py Profiler/test/test_hcu_devices.py \
  Profiler/test/test_hcu_counters.py
```

Multi-device tests require two visible devices. Counter tests launch ROCProfiler
in child processes; run them without other hardware-counter collectors on that
GPU. The native transfer test checks device/pinned-host buffers, copies, memset
and debug headers on default and non-default streams:

```bash
cmake -S Debugger/lib/Runtime -B build-hip-runtime \
  -DFLAGPRISM_BACKEND=hcu -DFLAGTREE_RUN_HCU_HARDWARE_TESTS=ON
cmake --build build-hip-runtime -j8
ctest --test-dir build-hip-runtime --output-on-failure
```
