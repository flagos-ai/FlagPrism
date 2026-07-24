# FlagTree Tools

此目录统一维护 FlagTree 的可选调试与性能分析组件：

- `Debugger/`：构建 `flagtree-debugger` wheel。
- `proton/`：构建 `flagtree-profiler` wheel。

两个组件保持各自的源码目录、构建入口和 Python wheel，不通过软链接组织。
构建时依赖已编译的 FlagTree；在 FlagTree 源码树外单独构建时，需要设置
`FLAGTREE_SOURCE_DIR` 和对应的构建环境变量。FlagTree 主 wheel 仅保留
`triton.debugger`、`triton.profiler` 门面及必要的组件接入点。
