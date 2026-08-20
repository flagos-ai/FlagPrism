# FlagTree Profiler - A Profiler for Triton

## Introduction

FlagTree Profiler is a lightweight profiler for Triton, designed to be used for code written in Python and to invoke underlying GPU kernels. FlagTree Profiler provides insightful information about the program context, metadata, and hardware performance metrics of the GPU kernels invoked.

## Installation

FlagTree Profiler is maintained in the `FlagPrism` submodule and bundled into the
main FlagTree wheel. Initialize the submodule and build FlagTree once:

```bash
git submodule update --init --recursive
FLAGTREE_BACKEND=ascend TRITON_BUILD_FLAGPRISM=ON MAX_JOBS=16 \
python -m pip install . --no-build-isolation
```

The main CMake graph builds `libtriton`, `flagtree.profiler._native`, and the
Profiler compiler plugin with one LLVM/MLIR configuration. The resulting wheel
installs `Profiler/python/flagtree_profiler` as `flagtree.profiler`.
`TRITON_BUILD_FLAGPRISM=OFF` produces a
core-only wheel; Debugger and Profiler cannot be enabled independently. On
Ascend, the default CANN `hook="triton"` IR path reuses the bundled Debugger
runtime, so the two tools are built as one suite.

To build a Tianshu-only variant, select the vendor backend explicitly. This
excludes the Ascend/CANN profiler sources and uses the Tianshu/CoreX runtime
path:

```bash
FLAGPRISM_BACKEND=tianshu TRITON_BUILD_FLAGPRISM=ON \
python -m pip install . --no-build-isolation
```

## Usage

### Basic usage

More examples can be found in the [tutorials](tutorials) directory.

FlagTree Profiler can be used to profile *functions* and *regions* in Python code.

- The following examples demonstrate how to use FlagTree Profiler to profile a simple Python function.

```python
import flagtree.profiler as profiler

# name: The path to the profile data
# context: The method used to annotate the context of each GPU kernel. Currently, "shadow" and "python" are supported.
session_id = profiler.profile(func, name="profile_name", context="python")(args)
```

- The following examples demonstrate how to use FlagTree Profiler to profile a region in Python code.

```python
session_id = profiler.start(name="profile_name", context="python")
...
# Skip a region
profiler.deactivate(session_id)
...
# Restart profiling
profiler.activate(session_id)
...
# Write out the profile data and finalize the profiler
profiler.finalize()
```

### Scope

Unlike the *python* context that provide users with files, functions, and lines where the GPU kernels are invoked, the *shadow* context provides users with the annotated regions in the code. The following example demonstrates how to use the *shadow* context.

```python
import flagtree.profiler as profiler


session_id = profiler.start(name="profile_name", context="shadow")

with profiler.scope("test0"):
    with profiler.scope("test1"):
        foo[1,](x, y)
with profiler.scope("test2"):
    foo[1,](x, y)

...
profiler.finalize()
```

The *scope* utility also accepts flexible metrics, provided with a dictionary that maps from a string (metric name) to a value (int or float).
FlagTree Profiler will aggregate the metrics for each scope and write them to the profile data.
It is useful for users to understand the performance of the model at a high level.

```python
with profiler.scope("test0", {"bytes": 1000}):
    with profiler.scope("test1", {"bytes": 2000}):
        foo[1,](x, y)
with profiler.scope("test2", {"bytes": 3000}):
    foo[1,](x, y)
```

### Backend and mode

FlagTree Profiler supports `cupti`, `roctracer`, `instrumentation`, `cann`, and
`tianshu` backends.

- **`cupti`**: Used for NVIDIA GPUs. It supports both the default profiling mode and `pcsampling` (instruction sampling).
- **`roctracer`**: Used for AMD GPUs. It supports only the default profiling mode.
- **`instrumentation`**: Available on both NVIDIA and AMD GPUs, this backend enables collection of custom metrics and advanced instrumentation.
- **`cann`**: Uses the Ascend vendor adapter and CANN runtime/import path.
- **`tianshu`**: Reuses Debugger instrumentation through the CoreX-compatible
  driver and imports ixKN CSV output. Use the `flagtree-profiler --ixkn` CLI
  wrapper because ixKN profiles the target process from startup.

By default, FlagTree Profiler automatically selects `cupti`, `roctracer`,
`cann`, or `tianshu` based on the active target backend. The `instrumentation`
backend offers a wide range of mode options for fine-grained profiling, as
detailed in the `mode.py` file.

#### Instruction Sampling

FlagTree Profiler supports instruction sampling on NVIDIA GPUs.
You may experience ~20x end-to-end overhead when using instruction sampling, although the overhead for each individual GPU kernel is negligible.
The overhead is mostly caused by data transfer and processing on the CPU.
Additionally, the flagtree-profiler-viewer options `-i <regex> -d <depth> -t <threshold>` can be helpful for filtering out GPU kernels that are not of interest.
The following example demonstrates how to use instruction sampling:

```python
import flagtree.profiler as profiler

profiler.start(name="profile_name", context="shadow", backend="cupti_pcsampling")
```

#### Instrumentation

The instrumentation backend allows for detailed, fine-grained profiling of intra-kernel behavior, generating trace or tree views similar to those produced by coarse-grained profiling.
By default, if no `mode` is specified, FlagTree Profiler profiles kernel cycles, which may require shared memory. If there is insufficient shared memory, profiling will abort and a warning will be displayed. Future releases will introduce additional instrumentation modes.

**Host-side usage:**

```python
import flagtree.profiler as profiler

profiler.start(
    name="profile_name",
    backend="instrumentation",
    mode="<mode0>=<option0>:<mode1>=<option1>:..."
)
```

**Kernel-side usage:**

**Caution**: For DSL level instrumentation, **only Gluon** semantic is enabled by default.
Instrumenting kernels written in Triton DSL is disable because Triton's higher-level IR undergoes
aggressive compiler rewrites (loop pipelining, instruction re-ordering, IR duplication, etc.).
These transformations can invalidate naïve instrumentation and lead to misleading results.

```python
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

import flagtree.profiler.language as pl

@gluon.jit
def kernel(...):
    pl.enter_scope("scope0")
    for i in range(iters):
        gl.load(...)
    pl.exit_scope("scope0")
    with pl.scope("scope1"):
        for i in range(iters):
            gl.load(...)
```

Advanced users can instrument either the `ttir` or `ttgir` intermediate representations for even finer-grained measurement. The relevant instructions are `proton.record start` and `proton.record end`; `proton` remains the internal MLIR dialect name. This can be combined with the environment variable `TRITON_KERNEL_OVERRIDE=1` for custom kernel overrides. For detailed steps, refer to the Triton [documentation](https://github.com/triton-lang/triton?tab=readme-ov-file#tips-for-hacking) under the **Kernel Override Steps** section. We have also assembled a [tutorial](tutorials/ttgir_override) that demonstrates how to use the IR-based instrumentation.

#### Merging profiles for postmortem analysis

We could use concurrent sessions to profile the same code region using different backends, and then merge the profiles using hatchet for postmortem analysis. In the following example, the `cupti` backend obtains different metrics than the `instrumentation` backend, and thus it makes sense to merge them using `GraphFrame.add` directly. Otherwise, if there are duplicate metrics, we could customize the `merge` logic or manipulate the dataframes.

```python

import flagtree.profiler as profiler

profiler.start(name="profile_name0", context="shadow", backend="cupti")
profiler.start(name="profile_name1", context="shadow", backend="instrumentation")

...

profiler.finalize()
```

### Hook

```python
import flagtree.profiler as profiler
from typing import NamedTuple

# hook: When hook="triton", it enables profiler to invoke launch_metadata function before launching the GPU kernel
profiler.start("profile_name", hook="triton")

def metadata_fn(
    grid: tuple,
    metadata: NamedTuple,
    args: dict
):
    return {"name": "<kernel_name>", "flops8": 1.0}

@triton.jit(launch_metadata=metadata_fn)
def foo(x, y):
    tl.store(y, tl.load(x))
```

The `metadata_fn` function is called before launching the GPU kernel to provide metadata for the GPU kernel, which returns a dictionary that maps from a string (metadata name) to a value (int or float).

Currently, **only the launch hook is supported**. In the dictionary returned by the `metadata_fn` function, we can supply the following keys:

```python
name: str  # The name of the kernel
flops8: float  # The number of 8-bit floating-point operations
flops16: float  # The number of 16-bit floating-point operations
flops32: float  # The number of 32-bit floating-point operations
flops64: float  # The number of 64-bit floating-point operations
bytes: int  # The number of bytes expected to be transferred
```

### Command line

FlagTree Profiler can be used as a command-line tool to profile Python scripts and Pytest tests.
The following examples demonstrate how to use FlagTree Profiler command-line.

```bash
flagtree-profiler [options] script.py [script_args] [script_options]
flagtree-profiler [options] pytest [pytest_args] [script_options]
python -m flagtree.profiler.cli [options] script.py [script_args] [script_options]
```

When profiling in the command line mode, the `profiler.start` and `profiler.finalize` functions are automatically called before and after the script execution. Any `profiler.start` and `profiler.finalize` functions in the script are ignored. Also, in the command line mode, only a single *session* is supported. Therefore, `profiler.deactivate(session_id=1)` is invalid, while `profiler.deactivate(session_id=0)` is valid.

### Visualizing the profile data

By default, profiler profiles are in the *json* format and can be read by *Hatchet*. The following command visualizes the profile data on terminal.

```bash
pip install llnl-hatchet
flagtree-profiler-viewer -m time/s <profile.hatchet>
```

NOTE: `pip install hatchet` does not work because the API is slightly different.

If you want to dump the entire trace but not just the aggregated data, you should set the data option to `trace` when starting the profiler.

```python
import flagtree.profiler as profiler

profiler.start(name="profile_name", data="trace")
```

The dumped trace will be in the chrome trace format and can be visualized using the `chrome://tracing` tool in Chrome or the [perfetto](https://perfetto.dev) tool.

### Visualizing sorted profile data

In addition visualizing the profile data on terminal through Hatchet. A sorted list of the kernels by the first metric can be done using the --print-sorted flag with flagtree-profiler-viewer

```bash
flagtree-profiler-viewer -m time/ns,time/% <profile.hatchet> --print-sorted
```

prints the sorted kernels by the time/ns since it is the first listed.

More options can be found by running the following command.

```bash
flagtree-profiler-viewer -h
```

## Advanced features and knowledge

### Thread management

We guarantee that calls to `flagtree.profiler._native`, such as `enter_scope`, are synchronized using explicit locks.
For operations that do not trigger native calls, including callbacks to CUDA/HIP APIs, we use separate locks to protect data structures accessed concurrently by multiple threads.
For example, the `enter_op` method in `OpInterface` can be invoked by the main thread that involves triton operators, as well as by helper threads that invoke torch operators.

### `cpu_timed_scope`

`cpu_timed_scope` is a utility that wraps `scope` to measure the CPU time of a scope along with other metrics.
The following example demonstrates how to use `cpu_timed_scope`:

```python
import flagtree.profiler as profiler

with profiler.cpu_timed_scope("test"):
    foo[1,](x, y)
```

The `cpu_timed_scope` output metric is referred to as `cpu_time`, while `time` represents accelerator (e.g., GPU) time.
The key distinction between `cpu_time` and `time` lies in their inclusivity: `cpu_time` is exclusive, whereas `time` is inclusive.
This difference arises because the time spent on individual kernels represents the smallest measurable time granularity, and each kernel is mutually exclusive.
This exclusivity allows time to be accurately accumulated across parent scopes for `time`.
In contrast, `cpu_time` measures the time within a specific scope.
Since a parent scope encompasses the time spent in its child scopes, summing `cpu_time` from child scope into parent scope would result in double counting.
To visualize both the CPU and GPU time, we can use the following command:

```bash
flagtree-profiler-viewer -m time/ns,cpu_time/ns <profiler.hatchet>
```

### Metrics naming

Custom metrics should follow this format: `metric_name (unit) (type)`.
We prefer no space within the metric name.
`unit` and `type` are optional fields.

There are three types of metrics in profiler: inclusive, exclusive, and property metrics.
By default, a metric is inclusive.
The metric types are distinguished by the suffix of their names.
The following table shows the suffix for each type and its meaning:

| Suffix | Name | Meaning |
| --- | --- | --- |
| (inc) or "" | Inclusive metric | The metric is accumulated at a scope and can be propagated to the parent scope. |
| (exc) | Exclusive metric | The metric is accumulated at a scope and cannot be propagated to the parent scope. |
| (pty) | Property metric | The metric is a property of the scope and cannot be accumulated or propagated. |

### State annotation

In addition to `profiler.scope`, we can also customize the call path of each GPU operation using `profiler.state`.

`state` is different from `scope` in several ways:

1. State is not recursive; each operation can have only a single state. Inner most state will overwrite the outer most state.
2. A states is a suffix, meaning that the original call path will append a state above the name of each kernel.
3. State is compatible with both Python and shadow contexts.

The following example demonstrates a basic use of state:

```python
with profiler.scope("test"):
    with profiler.state("state0"):
        with profiler.scope("test0"):
            foo0[1,](x, y)
        with profiler.scope("test1"):
            foo1[1,](x, y)
```

The call path of `foo1` will be `test->test1->state0`.

## FlagTree Profiler *vs* nsys

- Runtime overhead (up to 1.5x)

FlagTree Profiler has a lower profiling overhead than nsys. Even for workload with a large number of small GPU kernels, profiler triggers less than ~1.5x overhead.

For GPU-bound workload, both profiler and nsys has similar overhead, with little impact on the workload.

The lower overhead of profiler is due to its less profiling metrics and callbacks compared to nsys.

- Profile size (significantly smaller than nsys)

nsys traces and records every GPU kernel, while profiler aggregates the metrics of GPU kernels under the same calling context.

As a result, profiler's profile size can be up to thousands of times smaller than nsys's profile size, depending on the running time.

- Portability (support different GPUs)

FlagTree Profiler is designed to be portable and can be used on AMD GPUs. nsys only supports NVIDIA GPUs.

- Insights (more insightful than nsys on triton kernels)

FlagTree Profiler can register hooks to analyze the metadata of triton kernels, while nsys cannot. **Note** that the hooks do add additional overhead to profiler.

## FlagTree Profiler *vs* ncu

Similar to the comparison between FlagTree Profiler and Nsight Systems (Nsys), FlagTree Profiler has a lower profiling overhead than Nsight Compute (NCU). We also plan to support instruction sampling on AMD GPUs.
However, Nsight Compute supports the collection of more detailed metrics than FlagTree Profiler, such as memory access patterns, memory transactions, and other instruction-level metrics.
In contrast, FlagTree Profiler only supports instruction sampling and is designed to be lightweight and portable.

## Known issues

- CUDA graph

`hooks` cannot be used to accurately accumulate the number of FLOPs in CUDA graph mode profiling because kernels are captured and launched separately; metrics are not accumulated when kernels are launched in graph mode. This issue can be circumvented by using `scope` to supply FLOPs.

If profiling is initiated after CUDA graph capturing, there may be minor memory leak issues.
This is because the number of kernels in a graph instance (i.e., `cuGraphExec`) is unknown, preventing the deletion of mappings between the kernel ID and the graph ID.

- Instruction sampling

If you encounter permission related problems when using instruction sampling, you can lookup this [page](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters) for help.

The overhead of instruction sampling on NVIDIA GPUs is about 20x using FlagTree Profiler because we haven't enabled continuous sampling yet.
Continuous sampling can allow for more runtime optimizations, but it makes it more challenging to attribute performance data back to the GPU kernels because: (1) it enables profiling of concurrent kernels, (2) it doesn't allow profiling of time and instruction samples simultaneously, and (3) it works best if we have a separate thread dedicated to attributing instruction samples to the GPU kernels

- Visible devices on AMD GPUs

Environment variables such as `HIP_VISIBLE_DEVICES`, and `CUDA_VISIBLE_DEVICES` are not supported on AMD GPUs. Once it's set, we cannot find a valid mapping between the device ID returned by RocTracer and the physical device ID. Instead, `ROCR_VISIBLE_DEVICES` is recommended to be used.
