<!-- FlagPrism：MetaX adaptation, build requirements and hardware validation. -->
# MetaX / MACA

MetaX uses the Triton target name `maca` and the PyTorch device interface
`torch.cuda`. FlagPrism accepts `backend="metax"` and `backend="maca"` for
profiling. This does not select NVIDIA CUDA libraries.

The Debugger uses the existing TTIR instrumentation and hidden control-pointer
ABI. Its MACA adapter allocates device buffers and pinned host staging memory,
initializes the protocol header, and exports records on the launch stream.
Default-stream header initialization completes before the launcher returns to
user code. Non-default-stream exports synchronize that stream.

The Profiler uses MCPTI concurrent-kernel activity timestamps and API correlation
IDs. It records only kernels launched while the session is active, including
sessions paused and resumed through the public API. Kernel symbols, device IDs,
stream IDs, correlation IDs and nanosecond device intervals appear in the vendor
artifact and are overlaid into Hatchet and timeline reports. The collector is
compiled against the installed MACA SDK headers. Zero/invalid intervals and
lost activity records are errors; host API duration is not substituted for
device duration. Unsupported counter requests are reported in `degrade_reasons`.
Instruction-level device timing and validated achieved occupancy are still unavailable;
theoretical resource occupancy is available explicitly.

<!-- FlagPrism：shared IR route and activity-backed metric capabilities. -->
Both `hook="triton"` and `hook="instrumentation"` also use the shared Debugger
IR collector. Their reports include operation counts and read/write byte counts;
MCPTI remains the source of device kernel durations. IR collection adds overhead
to the measured kernel. Use `hook=None` for activity-only timing. Instruction
cycle timelines are unavailable on MetaX.

Request `vendor_metrics=launch_stats,resource_usage,kernel_duration` in the mode
string for MCPTI grid/block dimensions, registers per thread, shared/local memory
usage and device duration. These are activity-record properties, not hardware
performance counters. Additional real counter ranges are described below;
achieved-occupancy and utilization counters remain unsupported.
<!-- FlagPrism：explicit device selection and per-launch IR attribution. -->
Sessions use the current device for metadata unless `device_id` is explicitly
provided. Hardware counter groups use that selected device and its context,
restoring the caller's device after setup, flush and cleanup. Invalid device
indices are rejected before collection starts. Activity capture can still report
kernels on other devices; counter ranges measure only the selected device.
IR kernel/op associations use the device recorded for each launch. When the
same kernel symbol runs on multiple devices, Hatchet places its IR metrics under
separate `IR device <id>` children of the kernel scope. Native scope aggregation
is unchanged; use the activity associations for individual kernel/device records.
Failed hook/native initialization releases the session for retry.
<!-- FlagPrism end -->

An `all` build registers MetaX when MACA and MCPTI headers/libraries are found;
the SDK remains optional on other machines. This does not remove the other
backends' SDK requirements.
<!-- FlagPrism end -->

<!-- FlagPrism：validated runtime activities and explicit hardware counter ranges. -->
## Theoretical occupancy

Request `vendor_metrics=theoretical_occupancy,launch_stats,resource_usage` to
collect `metax.theoretical_occupancy_pct`, `active_blocks_per_sm`,
`threads_per_block`, `max_threads_per_sm` and the occupancy API source.
The calculation is `100 * active_blocks_per_sm * threads_per_block /
max_threads_per_sm`. It describes maximum resource-limited residency for a
kernel configuration, not achieved occupancy, grid utilization or GPU busy time.
For example, a one-block launch can have 100% theoretical resource occupancy
while using only a small part of the device.

The launch callback copies its function handle, block geometry and dynamic shared
memory. SDK queries run at flush on the kernel's actual device, restoring the
caller's selected device afterwards. Module queries use
`mcModuleOccupancyMaxActiveBlocksPerMultiprocessor`; ordinary runtime launches
use the corresponding runtime occupancy API when their function metadata is
available. The current SDK reports module launches in the driver callback domain
with runtime callback IDs, so callback matching uses the API name and ID together.

Queries are cached only within a flush. If a module unload was observed after a
launch and before its query, unresolved module queries from that earlier epoch
are conservatively skipped, including potentially unrelated modules. Missing
function metadata (for example graph replay) or query failure produces an
association note and degradation reason. When no kernel has an available result,
`theoretical_occupancy` is removed from enabled metrics. No achieved-occupancy
value is substituted.

## Experimental local clock probe

The SDK implements `clock64()` through `__builtin_readcyclecounter`. The matching
`llvm.readcyclecounter` intrinsic can be called through existing Triton
`extern_elementwise` lowering without backend changes. The probe test validates
local monotonicity, workload scaling and retention of a dependency loop between
the two reads in optimized LLVM IR. Run:

```bash
python -m pytest -q Debugger/test/python/runtime/test_metax_clock_probe.py
```

These are local counter ticks. A stable tick-to-nanosecond conversion and
cross-SM clock alignment have not been established. The SDK device clock property
is a reported clock limit, not a sufficient calibration for every execution.
The production internal timeline remains disabled on MetaX. The probe does not
claim cycle-accurate latency for arbitrary individual IR operations.

## Runtime activity and hardware counters

Use `mode="runtime_base:runtime_events=true"` to collect memcpy, memset and
runtime/driver API activities in the vendor report and timeline. Memory events
include byte counts and copy direction. API events carry host thread and return
status; API duration is host time, not GPU execution time. API/memory intervals
are excluded from Hatchet kernel totals. Synchronization API calls are collected;
the MACA 3.7 SDK used here did not emit standalone synchronization activities in
the tested workload, although the collector handles that activity type.

The collector restores the caller's current device after flushing every device
seen in active API contexts. MCPTI API enter/exit pairs are serialized across host
threads to avoid observed MACA 3.7 activity-bookkeeping races. Device launches
remain asynchronous, but profiling can affect host launch timing.

Real hardware counters can be explicitly requested, for example:

```python
session = profiler.start(
    "counter_profile", backend="metax",
    mode="runtime_base:vendor_metrics=inst_executed",
)
# Launch GPU work.
profiler.finalize(session)
```

Validated metric names are `dram_read_bytes`, `dram_write_bytes`,
`dram_read_throughput`, `dram_write_throughput` and `inst_executed`.
They are exported under `metax.hardware.*` in `mcpti_hardware_counter`
associations. Each association covers a session's active range on the device
selected at session start. Pause/resume creates separate ranges and excludes
paused work. These are **not per-kernel counter values**; a counter range on one
device does not measure other devices used by that session.
<!-- FlagPrism：MCPTI counter access requires exclusive process ownership. -->
Use hardware counters from only one profiling process per GPU. MACA may retain
its performance-counter channel until that process exits, even after a session
ends. Concurrent or nested profiling processes on that GPU are not a validated
counter configuration; activity-only collection is independent of this limit.
<!-- FlagPrism end -->

Counter deltas and SDK instance normalization are retained as `raw_event.*`
fields. Throughput is derived explicitly from hardware byte counts divided by
range duration, in bytes/second; the source and numerator are exported. Requests
requiring multiple replay passes degrade with a reason and are removed from
`enabled_metrics`. User kernels are never automatically replayed.

<!-- FlagPrism：cleanup failures remain observable without throwing from destructors. -->
Counter cleanup attempts every release operation and records failed API names and
numeric status codes in the vendor degradation reasons. A cleanup failure removes
the requested counter metrics from `enabled_metrics`; it is not silent success.
Destructors and error-path cleanup do not throw a second exception. Device indices
outside the visible range, malformed indices and integer overflow are rejected
before starting collection.
<!-- FlagPrism end -->

`achieved_occupancy` is deliberately excluded: the installed SDK enumerates it,
but its evaluated result failed the expected ratio bounds in hardware tests.
Enumeration or successful group enable alone is not treated as metric support.
Overlapping MetaX sessions and instruction-level cycle timelines remain
unsupported.
<!-- FlagPrism end -->

## Joint source build

Use the two repositories on `metax-202610-dev`. Keep the FlagPrism checkout
separate and use the source override supported by FlagTree:

```bash
cd FlagTree
FLAGTREE_BACKEND=metax \
FLAGPRISM_BACKEND=metax \
TRITON_BUILD_FLAGPRISM=ON \
TRITON_BUILD_PROTON=OFF \
FLAGPRISM_SOURCE_DIR="$(realpath ../FlagPrism)" \
python -m pip install . --no-build-isolation
```

The MACA SDK is resolved through `MACA_PATH` (or `/opt/maca`):
`include/mcr/mc_runtime_api.h`, `include/mcpti/mcpti.h`, `lib/libmcruntime.so`
and `lib/libmcpti.so` are required. Use the MetaX LLVM and backend plugin
versions selected in FlagTree's `python/setup_tools/utils/metax.py`.
A generic upstream LLVM package is not a replacement for the vendor toolchain.

The only FlagTree source changes are the Python/CMake component allowlists and
the MetaX launcher wrapper. Original replaced statements remain commented;
new blocks use `FlagPrism：` and `FlagPrism end` markers. Triton compiler passes
and the MetaX code generation pipeline are unchanged.

The current upstream MetaX graph can schedule MCTLE header consumers too early
on a first build. Generate its headers before compiling the full graph. For a
wheel build, use that configured directory as `TRITON_BUILD_DIR`; the wheel
validation performed here included this preparation step:

```bash
cmake --build <build-dir> --target McTleTableGen
cmake --build <build-dir> --target triton flagtree_debugger_native flagtree_profiler_native -j 24
```

## Usage

```python
from flagtree import debugger, profiler

debugger.activate(auto_collect=True, level=1, output_dir="metax_debug")
# Launch @triton.jit kernels. Use level=2 for full tensor capture.
debugger.deactivate()

session = profiler.start(
    "metax_profile", backend="metax", hook="triton", context="shadow",
    mode="runtime_base:runtime_host_timing_fallback=false",
)
# Launch @triton.jit kernels.
profiler.finalize(session)
```

## Validation

Run the existing shared operator suite, which checks CPU numerical references,
L1 records, L2 full-value tensor artifacts and real device timings:

```bash
cd FlagPrism
python test.py
```

For an initial diagnostic subset:

```bash
python test.py --ops abs,add,mm,softmax,sum --min-ops 1
python -m pytest -q Profiler/test/test_metax.py Profiler/test/test_metax_extended.py Profiler/test/test_metax_occupancy.py Profiler/test/test_metax_devices.py
python -m pytest -q Debugger/test/python/runtime/test_metax_debugger_e2e.py
```

The MCPTI regression checks exact session membership across pause/resume,
excludes paused launches, and verifies a new session after an overlap rejection.
It also checks hook failure recovery, current-device metadata, both IR hooks,
activity-backed resource metrics and unsupported-counter diagnostics. The L2
regression checks FP16, BF16 and FP32 values together with lane addresses.
Extended tests cover runtime copy/set/API activities, repeated two-thread/two-GPU
launches, graph replay, and hardware counter pause/resume ranges. Two visible
MetaX devices are required for the multi-device tests.
<!-- FlagPrism：regress the device combinations absent from the initial suite. -->
`test_metax_devices.py` checks explicit counter-device selection while another
device is current, caller-device restoration across pause/resume/finalize, both
IR hooks across devices, per-device Hatchet IR children, and invalid-device
startup recovery. `metax_cleanup_probe.py` injects a failed SDK cleanup return and
checks the error artifact and successful collection in the following session.
Run it in a separate process on a GPU with no other process using hardware
counters: `python Profiler/test/metax_cleanup_probe.py /tmp/metax-cleanup-probe`.
MACA can retain the perf-counter channel in a process after its counter groups
are destroyed; launching this probe from a pytest process that already used
hardware counters can deny the child access to that channel.
Debugger MetaX regressions also cover NaN/Inf summaries (FP16/BF16/FP32), an
actual L1 record-buffer overflow and recovery, and repeated enable/disable
transitions with assertions on the hidden-argument ABI.
<!-- FlagPrism end -->
The MACA transfer regression covers device roundtrips, asynchronous memset,
hidden pointers, header initialization/export and release on default and
non-default streams, on device 0 only:

```bash
cmake -S Debugger/lib/Runtime -B /tmp/flagprism-metax-transfer \
  -DFLAGPRISM_BACKEND=metax -DFLAGTREE_RUN_METAX_HARDWARE_TESTS=ON \
  -DFLAGTREE_DEBUGGER_RUNTIME_BUILD_SMOKE_TEST=OFF \
  -DFLAGTREE_DEBUGGER_RUNTIME_BUILD_TRACE_DEMO=OFF \
  -DFLAGTREE_DEBUGGER_RUNTIME_BUILD_ASYNC_LOOP_TEST=OFF
cmake --build /tmp/flagprism-metax-transfer -j 4
ctest --test-dir /tmp/flagprism-metax-transfer --output-on-failure
```

On NFS workspaces whose ownership is mapped to `nobody`, give pytest a fresh
local `--basetemp` directory if its default temporary-directory ownership check
fails. Actual hardware results and their tested source revisions must be read
from the run's `manifest.json` and `summary.json`; a runtime transfer pass alone
does not establish that compiler instrumentation works.
<!-- FlagPrism end -->
