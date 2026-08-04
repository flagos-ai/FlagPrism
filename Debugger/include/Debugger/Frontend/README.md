# Frontend

本目录对应 A 模块：Python 前端与参数穿透。
负责人：华师。

这个目录就是 A 模块的主要开发目录。A 的公共接口在这里定义，A 的实现入口在 `third_party/FlagPrism/Debugger/lib/Frontend/`。

核心文件：

- `Bridge.h`

模块目标：

- 把 Python 前端选项、compile request、launch request 统一成 debugger 可消费的结构。
- 把 B 输出的 `KernelDebugMetadata` 挂到编译产物上。
- 对 metadata 明确启用 hidden-argument ABI 的后端，把 F 提供的 control block
  指针作为 `__debug_ctrl_ptr` 透传到 kernel launch。
- 把运行时 dynamic tensor/buffer 信息整理成 `DebugRuntimeMetadata`，供 F 和 D 使用。

上游输入：

- Python 侧 debug 选项
- kernel 名称、backend、target
- B 输出的 `KernelDebugMetadata`
- F 提供的 `TransferEngine`
- launch 时的 tensor / buffer 实参

对下游输出：

- `DebugCompileRequest`
- `DebugKernelArtifacts`
- `DebugLaunchRequest`
- `PreparedDebugLaunch`
- 隐藏参数值 `hiddenArgValue`

本模块负责接线和整理的字段：

- 编译/运行开关：
  - `enabled`
  - `recordLevel`
  - `exportMode`
  - `recordCapacity`
  - `captureMemoryEvents`
  - `captureFullValues`
- kernel 级上下文：
  - `kernelName`
  - `backendName`
  - `targetName`
- launch 期上下文：
  - `kernelId`
  - `hiddenArgValue`
  - `DebugBufferPlan`
  - `TransferEngineOptions`
  - `streamHandle`
- 运行期 dynamic tensor / buffer 元数据：
  - tensor `argumentIndex`
  - tensor `logicalName`
  - `dtype`
  - `shape`
  - `stride`
  - `layout`
  - `bufferId`
  - `baseAddress`
  - `sizeBytes`
  - buffer `bufferName`
  - buffer `alignment`

上述 runtime tensor/buffer 类型是 A/F/D 间的扩展契约。当前 Python 默认 launch
path 只自动补充 grid、record plan 等 debugger metadata，尚未自动从 PyTorch
kernel 实参填充 tensor `shape / stride / layout`；调用方可通过
`runtime_metadata_builder` 显式提供，自动采集仍是后续工作。

推荐真实 launcher 入口：

- `prepareOwnedLaunch()`
  - 由 A 按 `BufferMeta.backendKind + streamHandle` 直接创建
    `TransferEngine`
  - 在 caller 未填全 `BufferMeta` 时，自动从 `DebugKernelArtifacts` 补齐
    `protocolVer / recordLevel / exportMode / kernelId / backendKind`
