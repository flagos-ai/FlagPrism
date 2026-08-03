# FlagPrism

此目录统一维护 FlagTree 的可选调试与性能分析组件：

- `Debugger/`：提供 `flagtree.debugger` 的 Python 与 native 实现。
- `proton/`：提供 `flagtree.profiler` 的 Python 实现、Proton runtime 与 CLI。
- `cmake/FlagPrism.cmake`：集中维护两个组件的 CMake 构建策略和 target 接入。
- `python/flagprism_build.py`：维护统一 wheel 的 package、CLI 和 CMake 参数策略。

本仓库作为 `third_party/FlagPrism` submodule 维护源码，不再分别发布
`flagtree-debugger` 和 `flagtree-profiler` wheel。FlagTree 主仓库的
`pip wheel .` 会在同一 CMake graph 中构建 core、Debugger 和 Proton，并将
Debugger 源码安装为 `flagtree.debugger`，将 `proton/proton` 源码安装为
`flagtree.profiler`，并把 `libproton` 与 Debugger native extension 写入同一个
FlagTree wheel。Debugger 的编译期插件编入 `libtriton`，运行期传输和解码接口位于
独立的 `flagtree.debugger._native`，后者不链接 `libtriton`。Profiler 只在 Python
公共入口使用 `flagtree.profiler`；内部继续使用 `proton/`、Proton dialect、
`libproton`、`PROTON_*` 和 Proton CLI。

```bash
git submodule update --init --recursive
python -m pip wheel . --no-build-isolation
```

Python wheel 只提供两种构建模式：默认的 Debugger + Profiler 联合构建，以及
`TRITON_BUILD_FLAGPRISM=OFF` 的 core-only 构建。两个组件不能独立启停。

旧环境变量 `TRITON_BUILD_DEVTOOLS` 和 `TRITON_BUILD_PROTON` 仅作为兼容输入保留，并且
同时出现时取值必须一致；新脚本和文档统一使用 `TRITON_BUILD_FLAGPRISM`。
Python 公共入口 `flagtree.debugger`、`flagtree.profiler`，以及既有
`flagtree_debug`/Proton IR ABI 保持不变，避免品牌重命名破坏用户代码和编译缓存。

Ascend 上默认的 `backend="cann", hook="triton"` IR 采集复用 Debugger 的插桩 runtime，
因此 Debugger 与 Profiler 始终作为一个工具套件构建和发布。
