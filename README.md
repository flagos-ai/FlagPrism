# FlagPrism

English | [简体中文](README_CN.md)

This repository centrally maintains FlagTree's optional debugging and profiling
components:

- `Debugger/`: provides the Python and native implementation of
  `flagtree.debugger`.
- `Profiler/`: provides the Python package, native runtime, and CLI for
  `flagtree.profiler`.
- `cmake/FlagPrism.cmake`: centralizes the CMake build policy and target
  integration for both components.
- `python/flagprism_build.py`: defines the package, CLI, and CMake argument
  policy for the unified wheel.

FlagTree consumes this repository as the `third_party/FlagPrism` submodule.
The `flagtree-debugger` and `flagtree-profiler` wheels are no longer published
separately. Running `pip wheel .` from the FlagTree repository builds the core,
Debugger, and Profiler in one CMake graph and packages them into a single
FlagTree wheel. The Debugger sources are installed as `flagtree.debugger`, and
the sources under `Profiler/python/flagtree_profiler` are installed as
`flagtree.profiler`. Both components also place their `_native` extensions in
the same wheel.

The Debugger compiler plugin is linked into `libtriton`. Its runtime transport
and decoding interfaces live in the standalone `flagtree.debugger._native`
extension, which does not link against `libtriton`. The Profiler exposes
`flagtree.profiler` as its only public Python entry point. Its native module is
`flagtree.profiler._native`, its command-line tools are `flagtree-profiler` and
`flagtree-profiler-viewer`, and its runtime environment variables use the
`FLAGTREE_PROFILER_*` prefix.

From the FlagTree repository root, run:

```bash
git submodule update --init --recursive
python -m pip wheel . --no-build-isolation
```

The Python wheel supports two build modes only: the default combined Debugger
and Profiler build, and a core-only build enabled with
`TRITON_BUILD_FLAGPRISM=OFF`. The two components cannot be enabled or disabled
independently.

`TRITON_BUILD_FLAGPRISM` is the only component build switch. The public Python
entry points `flagtree.debugger` and `flagtree.profiler`, as well as the existing
`flagtree_debug` ABI, remain stable. Internally, the Profiler compiler continues
to use the `proton` and `proton_gpu` MLIR dialect names, C++ namespaces, and
existing target names. These internal names are not public Python or CLI
interfaces.

On Ascend, the default `backend="cann", hook="triton"` IR collection path reuses
the Debugger instrumentation runtime. Debugger and Profiler are therefore built
and distributed together as one tool suite.
