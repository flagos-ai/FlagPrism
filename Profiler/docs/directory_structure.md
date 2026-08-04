# FlagTree Profiler 目录结构

Profiler 的公开实现集中在 `third_party/FlagPrism/Profiler/`，不再使用独立的
`proton/` 或 `flagtree_profiler/` 源码分区。

```text
third_party/FlagPrism/Profiler/
  CMakeLists.txt
  README.md
  python/flagtree_profiler/       # 安装为 flagtree.profiler
  csrc/                            # 通用 runtime 与 vendor adapter
    include/Profiler/Vendor/
    include/Driver/Ascend/
    lib/Profiler/Vendor/
    lib/Driver/Ascend/
  common/                          # trace 解码与公共数据结构
  Dialect/                         # 内部 FlagTree Profiler/ProtonGPU MLIR dialect
  cmake/ProtonDialectObjects.cmake # 内部 dialect object 列表
  examples/
  scripts/
  test/
  tutorials/
  docs/
```

## 公开边界

- Python：`flagtree.profiler`
- native：`flagtree.profiler._native`
- CLI：`flagtree-profiler`、`flagtree-profiler-viewer`
- 环境变量：`FLAGTREE_PROFILER_*`
- CMake 组件 target：`flagtree_profiler_native`

`python/flagtree_profiler/native.py` 将 runtime 调用转发到
`flagtree.profiler._native`。编译器 pass 仍通过
`triton._C.libtriton.proton` 获取；这里的 `proton` 是内部 compiler plugin 名，
不是兼容入口。

## 内部编译器名称

为避免修改 MLIR bytecode、pass pipeline 和既有后端链接契约，以下名称保留：

- `proton`、`proton_gpu` dialect 文本名
- `mlir::triton::proton` C++ namespace
- `ProtonIR`、`ProtonGPUIR`、`TritonProton`、`TritonTestProton` target
- `__PROTON__` 编译宏

这些名称只出现在 compiler/runtime 实现和内部测试中。产品文档、Python API、CLI、
native 文件名以及运行时环境变量均使用 Profiler 名称。

## Vendor 适配

通用 `Session` 和数据模型位于 `csrc/include`、`csrc/lib`。CANN 通过
`Profiler/Vendor/Adapter` 接口接入，Ascend runtime shim 位于
`Driver/Ascend`。新增后端应实现相同 adapter 契约，不应在父仓增加新的公开 API。

## 构建边界

`Profiler/CMakeLists.txt` 生成 `_native`，并由
`cmake/FlagPrism.cmake` 接入 FlagTree 的统一 CMake graph。父仓只负责包含 FlagPrism
构建策略、映射 Python package 和提供稳定的 compiler/runtime hook；Profiler 源码、
测试、示例和文档均留在 submodule。
