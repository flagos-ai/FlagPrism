# Proton CANN Adapter Acceptance Status

## Purpose

This document records the current validation status for the Proton CANN vendor
adapter work. It is intentionally separate from
`proton_vendor_adapter_minimal_patch.md`.

The minimal patch document describes the target design and the minimum contract
for adding a vendor adapter path. This document explains what was validated on
the server, what the result means, and what remains outside the current
acceptance scope.

## Background

The CANN vendor adapter work is currently scoped as a minimal integration layer
for Proton. The goal is to make the following path usable and reviewable:

- accept `backend="cann"` from the Python profiler API
- parse mode strings such as `runtime_base:vendor_metrics=aicore,bandwidth`
- create a CANN-specific vendor adapter path
- emit the expected artifact set:
  - `<base>.hatchet`
  - `<base>.timeline.json`
  - `<base>.meta.json`
  - `<base>.vendor.json`
- preserve base runtime output even when vendor enhancement is unavailable
- write degradation reasons into metadata and vendor outputs

The current minimal acceptance does not require real Ascend/CANN profiling data
to be collected. Real profiling support is described as follow-up work in the
minimal patch document, including:

- loading and using CANN profiling libraries
- importing `aclprof` or `msprof` output
- collecting real runtime kernel events
- correlating vendor metrics back to runtime events

## Server Environment Used

Validation was performed in the remote FlagTree checkout:

```text
/home/secure/zhaoguoxiang/FlagTree
```

The successful editable build used the conda environment:

```text
/home/secure/miniforge3/envs/flagtree-py310
```

Important build environment variables used during the successful build were:

```bash
export CC="$CONDA_PREFIX/bin/aarch64-conda-linux-gnu-cc"
export CXX="$CONDA_PREFIX/bin/aarch64-conda-linux-gnu-c++"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$CONDA_PREFIX/lib:$LIBRARY_PATH"
export LDFLAGS="-L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib"
export MAX_JOBS=16
```

The editable install was verified to import Triton from the active checkout:

```bash
python -c "import triton; print(triton.__file__)"
```

Expected result:

```text
/home/secure/zhaoguoxiang/FlagTree/python/triton/__init__.py
```

The Proton Python entry point was also verified:

```bash
python -c "import triton.profiler as proton; print(proton.start)"
```

## Minimal Acceptance Result

The CANN smoke test passed:

```bash
python -m pytest -q third_party/FlagTree_DevTools/proton/flagtree_profiler/test/test_cann_smoke.py -s
```

Observed result:

```text
6 passed
```

A manual artifact check was also run. It generated:

```text
/tmp/proton_cann_acceptance_1777884896/profile_run.hatchet
/tmp/proton_cann_acceptance_1777884896/profile_run.timeline.json
/tmp/proton_cann_acceptance_1777884896/profile_run.meta.json
/tmp/proton_cann_acceptance_1777884896/profile_run.vendor.json
```

The manual validation script printed:

```text
ACCEPTANCE OK
```

The checked conditions were:

- all four expected artifacts exist
- `meta.json` reports `backend == "cann"`
- `meta.json` reports `runtime_base_enabled == true`
- requested vendor metrics include `aicore` and `bandwidth`
- `degrade_reasons` is present in metadata
- `degrade_reasons` is present in vendor output
- `vendor.json` contains a readable `associations` list
- `timeline.json` contains a readable `traceEvents` list

## Real CANN 8.5 MSTX Validation

A real CANN 8.5 MSTX smoke probe was validated on the aarch64 server by
running an ACL program under external `msprof`:

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
msprof --msproftx=on --output=/tmp/cann_acl_mstx_real/msprof /tmp/cann_mstx_acl_probe
```

Observed probe result:

```text
mstx_domain=0x3206c720
mstx_range_id=1
exit_code=0
```

The run exported real CANN profiler outputs, including:

```text
mindstudio_profiler_output/msprof_tx_*.csv
mindstudio_profiler_output/task_time_*.csv
mindstudio_profiler_output/api_statistic_*.csv
```

This validates the CANN 8.5 runtime side of the integration:

- `libms_tools_ext.so` is the MSTX implementation used by this installation.
- MSTX domain/range APIs work when AscendCL is initialized, device 0 is set,
  and an ACL runtime stream is passed to the range start call.
- External `msprof --msproftx=on` captures and exports the MSTX data after the
  wrapped process exits.

## Real CANN ACLNN Compute Validation

A native ACLNN workload was validated on the same server without requiring
`torch` or `torch_npu`. The helper compiles a small C++ program that initializes
AscendCL, creates an ACL stream, emits an MSTX range, runs ACLNN Add on device
memory, launches the program under external `msprof`, and post-imports the
exported CSV files through Proton:

```bash
python third_party/FlagTree_DevTools/proton/flagtree_profiler/scripts/cann_native_acl_mstx_validate.py \
  --clean \
  --out /tmp/proton_cann_native_aclnn_add \
  --cann /usr/local/Ascend/cann-8.5.0 \
  --device 0 \
  --elements 1048576 \
  --iters 20 \
  --compute-kind aclnn-add
```

Observed compute result:

```text
mstx_range_id=1
acl_compute_op=aclnnAdd
acl_compute_elements=1048576
acl_compute_iters=20
acl_compute_first_f32=3.0
```

External `msprof` exported five real profiler CSV inputs:

```text
msprof_tx_*.csv
op_summary_*.csv
op_statistic_*.csv
task_time_*.csv
api_statistic_*.csv
```

The Proton post-import step read all five raw inputs and produced real vendor
associations:

```text
raw_inputs 5
association_sources {
  'msprof_mstx': 1,
  'aclprof_op_summary': 20,
  'msprof_api_statistic': 16,
  'msprof_op_statistic': 1
}
```

This validates the real CANN capture/import path for a native NPU compute
workload:

```text
Native ACLNN compute: PASS
MSTX annotation/capture: PASS
External msprof CSV export: PASS
Post-process Proton CSV import: PASS
op_summary association import: PASS
```

The active conda environment still does not contain `torch` or `torch_npu`, so
the Python `torch_npu` workload helper remains unvalidated on this server. This
is no longer a blocker for real CANN capture/import because the native ACLNN
workload provides a real NPU compute path.

## Current Proton Import Timing Limitation

The Proton run no longer indicates an MSTX runtime failure. The remaining
degradation reasons are expected for the current minimal patch shape:

```text
No vendor summary CSV files were found. Expected msprof exports such as summary/op_summary*.csv.
msprof CSV exports were not visible at proton.finalize time. When using external `msprof --msproftx=on`, summaries may only be exported after the wrapped process exits.
```

This is a timing limitation rather than a CANN runtime failure. Proton writes
`.vendor.json` during in-process `proton.finalize()`, while external `msprof`
exports `mindstudio_profiler_output/*.csv` only after the Python process exits.
The later `find "$MSPROF_OUT" -type f -iname '*.csv'` check proves those CSV
files do exist after `msprof` completes.

Therefore the current result should be read as:

```text
Real CANN MSTX runtime integration: PASS
External msprof capture/export: PASS
Post-process Proton import of externally exported CSV: PASS
In-process Proton import of externally exported CSV at finalize time: LIMITED BY EXPORT TIMING
Real native NPU compute capture/import: PASS via ACLNN Add
```

## Acceptance Interpretation

Current status:

```text
Minimal acceptance: PASS
Real CANN MSTX runtime validation: PASS
External msprof CSV export validation: PASS
Post-process msprof CSV import validation: PASS
Real native ACLNN compute validation: PASS
op_summary association validation: PASS
Synchronous import of external msprof CSV during proton.finalize: NOT AVAILABLE
Python torch_npu workload validation: NOT AVAILABLE IN CURRENT ENV
```

The current implementation satisfies the minimal acceptance criteria from
`proton_vendor_adapter_minimal_patch.md`:

1. `proton.start(..., backend="cann", mode="runtime_base:vendor_metrics=...")`
   works.
2. `proton.finalize(...)` emits the four expected artifact files.
3. When vendor enhancement is unavailable, base artifacts still exist and
   degradation reasons are written to `meta.json` and `vendor.json`.

The current validation proves the real CANN MSTX annotation/capture path and a
real native ACLNN Add compute workload. It also proves post-process import of
real `op_summary`, `op_statistic`, `task_time`, `api_statistic`, and
`msprof_tx` CSV files into Proton vendor artifacts. Python `torch_npu` coverage
is still unavailable in the current environment.

## How To Reproduce The Minimal Check

From the remote FlagTree checkout:

```bash
cd /home/secure/zhaoguoxiang/FlagTree

source /home/secure/miniforge3/etc/profile.d/conda.sh
conda activate flagtree-py310

export CC="$CONDA_PREFIX/bin/aarch64-conda-linux-gnu-cc"
export CXX="$CONDA_PREFIX/bin/aarch64-conda-linux-gnu-c++"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$CONDA_PREFIX/lib:$LIBRARY_PATH"
export LDFLAGS="-L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib"
export MAX_JOBS=16

python -m pip install -e . --no-build-isolation
python -m pytest -q third_party/FlagTree_DevTools/proton/flagtree_profiler/test/test_cann_smoke.py -s
```

For a manual artifact check:

```bash
OUT=/tmp/proton_cann_acceptance_$(date +%s)
export OUT
mkdir -p "$OUT/vendor"

python - <<'PY'
import os
import pathlib
import time
import triton.profiler as proton
from flagtree_profiler.native import runtime_binding

libproton = runtime_binding()

base = pathlib.Path(os.environ["OUT"]) / "profile_run"
vendor_output = pathlib.Path(os.environ["OUT"]) / "vendor"

sid = proton.start(
    name=str(base),
    context="shadow",
    data="tree",
    hook="triton",
    backend="cann",
    mode=(
        "runtime_base:"
        "vendor_metrics=aicore,bandwidth:"
        f"aclprof_output_path={vendor_output}:"
        "runtime_host_timing_fallback=true"
    ),
)

scope_id = libproton.record_scope()
libproton.enter_op(scope_id, "cann_acceptance_kernel")
time.sleep(0.001)
libproton.exit_op(scope_id, "cann_acceptance_kernel")

proton.finalize(sid)
print(base)
PY

ls -lh "$OUT"/profile_run.*
```

Then validate the generated JSON:

```bash
python - <<'PY'
import os
import json
import pathlib

base = pathlib.Path(os.environ["OUT"]) / "profile_run"

paths = {
    "hatchet": base.with_suffix(".hatchet"),
    "timeline": base.with_suffix(".timeline.json"),
    "meta": base.with_suffix(".meta.json"),
    "vendor": base.with_suffix(".vendor.json"),
}

missing = [name for name, path in paths.items() if not path.exists()]
assert not missing, f"missing artifacts: {missing}"

meta = json.loads(paths["meta"].read_text())
vendor = json.loads(paths["vendor"].read_text())
timeline = json.loads(paths["timeline"].read_text().splitlines()[0])

assert meta["backend"] == "cann"
assert meta["runtime_base_enabled"] is True
assert "aicore" in meta["vendor_metrics_enabled"]
assert "bandwidth" in meta["vendor_metrics_enabled"]
assert isinstance(meta["degrade_reasons"], list)
assert isinstance(vendor.get("degrade_reasons", []), list)
assert isinstance(vendor.get("associations", []), list)
assert isinstance(timeline.get("traceEvents", []), list)

print("ACCEPTANCE OK")
print("meta degrade_reasons:", meta["degrade_reasons"])
print("vendor degrade_reasons:", vendor.get("degrade_reasons", []))
print("timeline events:", len(timeline.get("traceEvents", [])))
PY
```

## How To Check Real CANN Profiling Availability

For CANN 8.5, real profiling collection should follow the `mstx` + external
`msprof` flow documented by Ascend. Proton writes `mstx` ranges for profiler
scopes, and `msprof --msproftx=on` captures those ranges while it launches the
user program.

First confirm the CANN runtime/profiling environment is available to the
process:

```bash
echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -i Ascend || true
find /usr/local/Ascend -name 'libascendcl.so*' -o -name 'libmsprofiler.so*' -o -name 'libprofapi.so*' -o -name 'msprof' 2>/dev/null
which msprof || true
```

If the libraries exist but are not on the runtime path, source the Ascend
environment script before running validation:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

or, depending on the installation:

```bash
source /usr/local/Ascend/latest/set_env.sh
```

Then run the target program through `msprof`:

```bash
msprof --msproftx=on --output="$OUT/vendor" python your_program.py
```

If custom MSTX domains are used, include the Proton domain:

```bash
msprof --msproftx=on --mstx-domain-include=proton --output="$OUT/vendor" python your_program.py
```

Real vendor profiling validation should check for:

- no `Failed to load CANN mstx APIs` degradation reason
- no `mstx range start returned an invalid range id` degradation reason
- exported `mindstudio_profiler_output/msprof_tx_*.csv` files after `msprof`
  exits
- real `aclprof` or `msprof` summary CSV inputs
- non-empty imported vendor metrics in `vendor.json`
- non-empty runtime or kernel events in `timeline.json`
- meaningful associations between runtime events and vendor metrics

When exported CSV files already exist before Proton finalization, the importer
can be pointed at them with either a mode option such as
`msprof_import_path=/path/to/msprof` or the environment variable:

```bash
export PROTON_CANN_IMPORT_PATH=/path/to/msprof
```

For end-to-end real NPU validation, use the repository helper scripts after
building the editable install on the Ascend server:

```bash
cd /home/secure/zhaoguoxiang/FlagTree
source /usr/local/Ascend/cann-8.5.0/set_env.sh

python third_party/FlagTree_DevTools/proton/flagtree_profiler/scripts/cann_real_msprof_validate.py \
  --clean \
  --out /tmp/proton_cann_real \
  --device 0 \
  --size 1024 \
  --iters 20
```

The driver does three things:

1. runs `cann_real_npu_workload.py` under
   `msprof --msproftx=on`
2. waits for external `msprof` to export `mindstudio_profiler_output/*.csv`
3. runs `cann_post_import_msprof.py` to import those exported CSV files into a
   second Proton vendor artifact

Successful output should include a nonzero `exported_csv_count`, nonempty
`raw_inputs`, and association sources such as `aclprof_task_time`,
`aclprof_op_summary`, `msprof_mstx`, or supplemental `msprof_*` sources,
depending on what CANN exports for the workload.

If the active environment does not provide `torch`/`torch_npu`, use the native
ACLNN Add + MSTX validation helper instead. It compiles a small C++ AscendCL
program, runs real ACLNN Add compute under external `msprof`, and post-imports
the exported CSV files:

```bash
python third_party/FlagTree_DevTools/proton/flagtree_profiler/scripts/cann_native_acl_mstx_validate.py \
  --clean \
  --out /tmp/proton_cann_native_aclnn_add \
  --cann /usr/local/Ascend/cann-8.5.0 \
  --device 0 \
  --elements 1048576 \
  --iters 20 \
  --compute-kind aclnn-add
```

This validates the real CANN capture/import flow available on the current
server without requiring Python NPU packages. It proves ACL runtime
initialization, MSTX range capture, native ACLNN compute, `msprof` CSV export,
and Proton post-import of real `op_summary` associations.

The legacy in-process `aclprofStart/aclprofStop` path is intentionally disabled
by default for CANN 8.5 because some installations expose only stub
`libacl_prof.so` libraries. It can still be enabled explicitly with:

```text
aclprof_runtime_enabled=true
```

## Reproducibility Notes

The successful build was not independent of the configured environment. A new
user, a new conda environment, or a clean clone may fail unless the same build
toolchain and runtime variables are configured.

In particular, the successful server build depended on:

- Python 3.10 from the `flagtree-py310` conda environment
- conda-provided aarch64 GCC/G++
- conda-provided CMake and Ninja
- conda-provided zlib
- `LD_LIBRARY_PATH`, `LIBRARY_PATH`, and `LDFLAGS` pointing at the conda prefix

Also note that the remote build disabled the new C++ Vendor unittest directory
to avoid unrelated link failures in the test executable. The Python CANN smoke
test does not depend on that C++ unittest target.

## Follow-Up Work

The next code changes should stay aligned with
`proton_vendor_adapter_minimal_patch.md` and avoid changing behavior for
`cupti`, `roctracer`, or `instrumentation`.

Recommended next steps:

1. Add an explicit post-process import path for already exported
   `mindstudio_profiler_output/*.csv` files. This can reuse
   `CannProfiler::importMsprofOutput(...)`; it should run after external
   `msprof` exits, not during in-process `proton.finalize()`.
2. Keep the current finalize-time degradation reason. It accurately explains
   why external `msprof` CSV files are not yet visible.
3. Keep the native ACLNN Add validation helper as the no-`torch_npu` real
   workload path. Add optional `torch`/`torch_npu` validation only when such an
   environment is available.
4. If a post-process importer is added, document it as an optional follow-up
   workflow rather than a replacement for the minimal artifact contract.

## Recommended Status Wording

Use this wording when reporting the current result:

```text
Minimal CANN adapter acceptance passed. The Python API accepts backend="cann"
and finalization emits .hatchet/.timeline.json/.meta.json/.vendor.json. Real
CANN 8.5 MSTX runtime annotation was validated with external
msprof --msproftx=on. A native ACLNN Add workload also ran successfully under
msprof and exported msprof_tx/op_summary/op_statistic/task_time/api_statistic
CSV files. Proton post-import read all five raw inputs and produced
aclprof_op_summary associations. The remaining limitation is only synchronous
import during in-process proton.finalize(), because external msprof exports CSV
files after the wrapped process exits. Python torch_npu validation is still
unavailable in the current conda environment.
```
