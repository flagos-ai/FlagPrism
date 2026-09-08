# FlagPrism

English | [简体中文](README_CN.md)

<p align="center">
  <img src="docs/assets/flagprism-architecture.png"
       alt="FlagPrism architecture"
       width="100%" />
</p>

FlagPrism is a multi-backend debugging and performance-analysis toolkit for
Triton programs, designed to provide a consistent observability workflow across
NVIDIA GPUs and diverse AI accelerators. Built for the
[FlagTree](https://github.com/flagos-ai/FlagTree) ecosystem, it observes Triton
kernels at both compile time and runtime, connects source-level context with
Triton IR operations and device events, and turns the collected data into
reports that help developers understand correctness, memory behavior, and
performance across heterogeneous accelerator backends.

FlagPrism contains two user-facing components:

- **[Debugger](Debugger/README.md)** captures values, numerical summaries,
  memory-address summaries, complete tensor data, and statement/operation
  metadata from selected regions inside `@triton.jit` kernels. See the
  Debugger README for configuration, collection levels, report formats,
  examples, and current limitations.
- **[Profiler](Profiler/README.md)** records execution context, timelines,
  operation counts, estimated bytes, hardware metrics, and vendor-profiler
  data. It aggregates these records into tree, timeline, Hatchet, metadata,
  and vendor-specific outputs. See the Profiler README for APIs, modes,
  command-line tools, visualization, and backend-specific usage.

## Backend Support

The table below tracks the FlagPrism enablement roadmap. It describes Debugger
and Profiler integration, not the availability of the corresponding FlagTree
compiler backend. We are rapidly expanding FlagPrism to support more devices
and accelerator backends.

| NVIDIA | Huawei Ascend | T-Head | Hygon | Moore Threads | MetaX | ILUVATAR | Enflame |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| In progress<br>Target: Sep 2026 | ✅ | — | — | Merging | Not started<br>Target: Oct 2026 | ✅ | In progress<br>Target: Sep 2026 |

## Relationship with FlagTree

[FlagTree](https://github.com/flagos-ai/FlagTree) is a unified, multi-backend
compiler that enables Triton programs to run across diverse AI accelerators.
FlagPrism complements FlagTree with observability tools that help developers
debug kernel correctness, analyze runtime behavior, identify performance
bottlenecks, and optimize Triton workloads. FlagPrism is integrated and
maintained as the `third_party/FlagPrism` submodule of FlagTree.

## Architecture Design

### Overview

FlagPrism is designed around a unified observability experience across
heterogeneous accelerators. It separates what developers want to observe from
how each backend collects that information, allowing Debugger and Profiler to
offer consistent concepts and workflows across devices. The design emphasizes
low intrusion into user programs, reuse between debugging and profiling,
predictable results, and incremental support for new accelerator backends.

### Debugger

The Debugger instruments selected regions inside Triton kernels and combines
static source and IR metadata with runtime value and memory information. It
helps developers inspect numerical results, memory access, and kernel data flow
through statement-level reports, operation-level reports, and tensor artifacts.
See the [Debugger README](Debugger/README.md) for details.

### Profiler

The Profiler combines program context, kernel events, instrumentation data, and
vendor performance metrics to analyze how Triton workloads execute on a device.
It correlates and aggregates the collected information into call trees,
timelines, and portable performance reports. See the
[Profiler README](Profiler/README.md) for details.

## Developing with FlagTree

FlagPrism is developed and built together with FlagTree. Download FlagTree,
initialize the FlagPrism submodule, and build the complete project from the
FlagTree repository root:

```bash
git clone https://github.com/flagos-ai/FlagTree.git
cd FlagTree
git submodule update --init --recursive third_party/FlagPrism
python3 -m pip install . --no-build-isolation
```

## Repository Layout

```text
FlagPrism/
├── Debugger/                 # Compiler passes, runtime, decoder, and Python API
├── Profiler/                 # Profiler dialect, runtime, adapters, and Python API
├── cmake/FlagPrism.cmake     # Unified CMake integration policy
├── python/flagprism_build.py # FlagTree wheel and package integration
└── docs/                     # Architecture and backend adaptation documents
```

## License

FlagPrism is licensed under the [MIT License](LICENSE), using the same license
as [FlagTree](https://github.com/flagos-ai/FlagTree/blob/main/LICENSE).
