# FlagPrism Debugger

FlagPrism Debugger 用于观察 Triton kernel 内部的数值、内存访问和 operation
执行状态。它将编译期静态 metadata 与 device 运行期记录关联，导出 Triton
语句级报告、IR op 级报告和 level 2 NumPy artifact，用于定位数值异常、异常
访存和 kernel 内部数据流问题。当前动态采集和 hidden-argument launch 路径已在
Ascend/CANN9 与 Tianshu/CoreX 4.4 LLVM 22 后端验证；其他后端的接入边界见
[Backend Support](#backend-support)。

## Public API

Debugger 唯一的公开 Python 导入路径是：

```python
import flagtree.debugger as debugger
```

不再提供 `triton.debugger` 公开命名空间。Triton JIT kernel 内的采集边界由
FlagTree language 扩展提供：

```python
ftl.debug_collect_start(level=1, addr_level=1)
# Triton operations to collect
ftl.debug_collect_end()
```

`flagtree.debugger` 负责 Python 配置、编译模式和导出；
`ftl.debug_collect_start/end` 只负责界定 `@triton.jit` 内需要采集的 IR 区域。

core-only wheel 不包含 `flagtree.debugger`。需要在通用代码中探测工具是否已随
FlagTree 构建时，可通过 host gateway 加载组件；正常用户代码直接导入公开 API
即可。Debugger 已成功导入后，也可用 `debugger.is_available()` 检查 compiler 和
runtime native binding 是否同时可用。

## Build And Install

Debugger 位于 `third_party/FlagPrism/Debugger`，由 FlagTree 主工程统一
编译和打包，不再单独发布 wheel。

```bash
cd /path/to/FlagTree
git submodule update --init --recursive

export FLAGTREE_BACKEND=ascend
export TRITON_BUILD_FLAGPRISM=ON
export MAX_JOBS=16

python3 -m pip install . --no-build-isolation
```

开发期可以使用 editable install：

```bash
python3 -m pip install -e . --no-build-isolation --no-deps
```

`TRITON_BUILD_FLAGPRISM=ON` 会在同一 CMake graph 中编译 Debugger、Profiler 及其
native runtime，并写入同一个 FlagTree wheel。使用
`TRITON_BUILD_FLAGPRISM=OFF` 时构建 core-only FlagTree，该 wheel 不包含两个工具
package。当前只支持“FlagPrism 联合构建”和“core-only”两种发布模式，不提供
Debugger-only、Profiler-only 或独立工具 wheel。

构建纯天数版本时设置 `FLAGPRISM_BACKEND=tianshu`。该模式只编译 Tianshu/CoreX
适配，不会探测或链接昇腾 CANN：

```bash
FLAGPRISM_BACKEND=tianshu TRITON_BUILD_FLAGPRISM=ON \
python3 -m pip install . --no-build-isolation
```

## Quick Start

```python
from pathlib import Path

import torch
import torch_npu
import triton
import flagtree.debugger as debugger
import flagtree.language as ftl
import triton.language as tl


debugger.configure(
    output_dir=Path("/tmp/flagtree_debugger_example"),
    record_capacity=4096,
    export_raw_records=False,
)
debugger.activate(level=1, addr_level=1)


@triton.jit
def debug_abs_kernel(x_ptr, y_ptr, n: tl.constexpr,
                     BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    ftl.debug_collect_start(level=1, addr_level=1)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.abs(x)
    z = y + 1.0
    tl.store(y_ptr + offsets, z, mask=mask)
    ftl.debug_collect_end()


n = 16
x = torch.linspace(-8, 7, n, dtype=torch.float32, device="npu")
y = torch.empty_like(x)

debug_abs_kernel[(1,)](x, y, n, BLOCK_SIZE=16)
torch_npu.npu.synchronize()

print("output_allclose=", torch.allclose(y.cpu(), (torch.abs(x) + 1).cpu()))
for run in debugger.take_exported_runs():
    print("report_path=", run.get("report_path"))
```

`debugger.activate()` 应在编译需要调试的 kernel 之前调用。它开启进程级
Debugger pipeline，但不会记录 Python、PyTorch 或 torch_npu operation。只有
`@triton.jit` 内位于 collect marker 之间的 Triton operation 才会进入采集。

## Configuration

### Persistent Defaults

`debugger.configure()` 修改后续 `activate()` 使用的默认配置。未传入的字段
保持当前值。

```python
debugger.configure(
    output_dir="/tmp/flagtree_debugger_manual",
    record_capacity=4096,
    export_mode="POST_KERNEL_EXPORT",
    export_on_error=False,
    export_raw_records=False,
)
```

| Option | Default | Meaning |
| --- | --- | --- |
| `output_dir` | `/tmp/flagtree_debugger_manual` | 报告输出目录；`None` 关闭自动文件导出 |
| `record_capacity` | `1024` | device record slot 容量，必须为正整数 |
| `export_mode` | `POST_KERNEL_EXPORT` | kernel 完成后导出；协议也接受 `STREAMING_EXPORT` |
| `export_on_error` | `False` | kernel 报错后是否仍尝试导出 |
| `export_raw_records` | `False` | 是否额外写出 decoded raw-record sidecar |

查询和恢复配置：

```python
print(debugger.get_config())
debugger.reset_config()
```

### Activation

`level` 和 `addr_level` 属于采集策略，通过 `activate()` 配置：

```python
debugger.activate(level=1, addr_level=1)
```

- `level=1`：采集数值 summary。
- `level=2`：采集 summary 并导出支持的完整 tensor value。
- `addr_level=0`：不插入动态地址采集。
- `addr_level=1`：采集 load/store 的地址摘要。
- `addr_level=2`：当 `level=2` 且后端支持当前 pointer/mask pattern 时，额外
  导出完整 lane address；与 `level=1` 组合时仍只生成地址摘要。

`ftl.debug_collect_start()` 可以为当前 region 指定 level。`addr_level=None`
时继承 `debugger.activate()` 的地址采集等级：

```python
ftl.debug_collect_start(level=1)
```

长进程、notebook 或测试套件可在不再编译 debug kernel 时关闭 pipeline：

```python
debugger.deactivate()
```

一次性脚本通常无需调用 `deactivate()`。

## Collected Data

### Static Metadata

IR op 级编译期 metadata 通常包含：

- kernel id 和 kernel name
- `scope_id`、`op_id` 和 statement id
- MLIR operation 名称
- Python source location 和 Triton statement
- 编译器可确定的 operand/result dtype 和 logical shape
- load/store 的 address space、access type、element bytes、alignment 和 mask

分配了 `op_id` 但没有 runtime record 的 operation 会作为 static-only context
保留，例如部分 `tt.splat` 和 `tt.addptr`。

`addr_space` 和 `access_type` 描述一次访存的静态语义，例如
`addr_space=global access_type=load`；它们位于 `_op_log.txt/.json`，statement
主报告不会重复展示这些 IR 细节。当前 op log 对未捕获 compiler encoding 的
Triton SSA register/pointer value 仍可能输出 `stride: unknown` 和
`layout: unknown`。这些字段不是 PyTorch runtime tensor 的 stride/layout，不能用来
判断输入是否 contiguous、transpose 或 view；相关清理与 runtime tensor 补充方案记录
在 [todo.md](todo.md)。

### Level 1 Summary

对支持的 tensor result，level 1 可记录：

- `element_count`
- `nan_count`
- `inf_count`
- `zero_count`
- `mean`
- `min`
- `max`
- `l2_norm`

`addr_level=1` 可为支持的 memory operation 记录：

- `first_addr`
- `last_addr`
- `min_addr`
- `max_addr`
- `active_lane_count`
- `address_span_bytes`

地址摘要会应用 memory mask。报告中的 `status` 会指示摘要是否完整；
不支持的 pointer pattern 不会伪造 lane 地址。

`element_count` 是被观察值的逻辑 lane 数，不等于 mask 后的有效访存 lane 数；
后者由 `active_lane_count` 表示。

### Statement-Level View

主报告按 Triton 源语句组织，并把结果、operand 和访存地址放在各自语义位置。
以下是当前 Level 2 输出的精简片段：

```text
statement: x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
  [result x]:
    instances: [0]
    summary:
      element_count: [16 (U64)]
      nan_count    : [0 (U64)]
      ...
    full_value_file: op3_inst0_rec10_value.npy
    address_summary(load from):
      status            : [complete]
      first_addr        : [0x12c041200000]
      last_addr         : [0x12c04120003c]
      active_lane_count : [16]
      address_span_bytes: [64]
    memory_address_file: op3_inst0_rec11_memory_address.npy
  <operand mask>:
    instances: [0]
    summary:
      element_count: [16 (U64)]
    full_value_file: op9_inst0_rec13_value.npy
  <operand other>:
    constant_value: dense<0.000000e+00> : tensor<16xf32>

statement: tl.store(y_ptr + offsets, z, mask=mask)
  memory_access:
    instances: [0]
    address_summary(store to):
      status            : [complete]
      first_addr        : [0x12c041200200]
      last_addr         : [0x12c04120023c]
      active_lane_count : [16]
      address_span_bytes: [64]
  <operand z>: [result z]
```

- `[result x]` 的 `summary` 描述 `tl.load` 得到的值。
- `address_summary(load from)` 描述 `x_ptr + offsets` 经 mask 后实际读取的源地址，
  不是 register result `x` 自身的地址。
- `address_summary(store to)` 描述 `y_ptr + offsets` 经 mask 后写入的目标地址；
  `<operand z>: [result z]` 表示写入值引用前面已经展示的 `z`，不重复打印其 summary。
- operand 来自前序结果时使用 `[result ...]` 引用；有独立捕获结果且不能引用已有
  result 时才展示自己的 summary/artifact；编译期常量显示为 `constant_value`。
- statement 报告不展示 `capture_op_id`、`access_op_id` 等内部关联字段。需要检查
  `op_id`、完整静态 metadata 或所有底层 artifact 时使用 `_op_log` 和
  `tensor_index.json`。

### Level 2 Full Dump

level 2 将完整 payload 写入 artifact 目录：

```text
<report-stem>_artifacts/
  tensor_index.json
  op<id>_inst<id>_rec<id>_value.npy
  op<id>_inst<id>_rec<id>_memory_address.npy
```

对于能关联到 statement value/access 的 artifact，statement 文本报告只显示文件名，
并紧跟其对应的数据块：`full_value_file` 位于 `summary` 后，
`memory_address_file` 位于 `address_summary(...)` 后。文件名相对于同 stem 的
`_artifacts/` 目录，不输出机器相关的完整路径。`tensor_index.json` 保存所有
artifact 的完整索引；`_op_log` 也保留按 `op_id` 组织的引用。

artifact 按插桩计划生成，并不表示每个源码 operand 都会单独保存。来自前序
statement result 的 operand 继续引用该 result；store 的 value 也引用已展示的
result，避免在 statement 主视图重复输出相同数据。pointer 链等 IR-only artifact
可只出现在 `tensor_index.json` 和 `_op_log` 中。

level 2 要求 `output_dir` 不为 `None`。完整 value/address dump 仅支持
编译期可确定 shape 且当前后端能合法 lowering 的数值与 pointer pattern。
强制对不支持的 pattern 做 level 2 dump 会在编译期报错，不会生成
看似成功但数据不完整的报告。

## Reports

报告文件名由脚本名、kernel 名、时间戳和 run id 组成：

```text
<script>_<kernel>_<timestamp>_run<N>.txt
<script>_<kernel>_<timestamp>_run<N>.json
<script>_<kernel>_<timestamp>_run<N>_op_log.txt
<script>_<kernel>_<timestamp>_run<N>_op_log.json
<script>_<kernel>_<timestamp>_run<N>_raw_records.txt
```

- 主 `.txt` / `.json`：按 Triton source statement 组织，用于日常调试。
- `_op_log.txt` / `_op_log.json`：按 MLIR `op_id` 组织，包含更完整的
  静态 metadata 和 runtime record。
- `_raw_records.txt`：只在 `export_raw_records=True` 时生成，用于协议
  和 decoder 调试。
- `_artifacts/`：level 2 完整 tensor/address dump。

当 kernel 有多个 program instance 时，文本报告以 `instances: [...]` 为对齐轴，
各 summary 和 address summary 按相同顺序输出数组。JSON 报告保留同等
结构，便于批处理。

在 Python 中取得导出结果：

```python
runs = debugger.peek_exported_runs()  # 不清空内部列表
runs = debugger.take_exported_runs()  # 返回并清空内部列表
debugger.clear_exported_runs()
```

关键 run 字段包括 `report_path`、`json_report_path`、
`op_log_report_path`、`op_log_json_report_path`、`raw_records_path`、`meta`、
`runtime_metadata` 和 `decoded`。Level 2 run 还包含
`full_dump_artifact_dir`、`full_dump_index_path`；artifact 列表位于
`runtime_metadata["full_dump_artifacts"]`。只有对应输出被启用或生成时，相应
字段才存在。

如果 `overflow_count` 非零，说明 `record_capacity` 不足，报告可能不完整。
应增大 capacity 并重新运行。level 2 不允许从 overflowed buffer 导出
artifact。

## Ascend Runtime

在已安装 FlagTree wheel 的 Ascend 环境中，从 FlagTree 源码 checkout 运行示例：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export FLAGTREE_BACKEND=ascend
# 多卡机器可按需选择可用物理设备，例如 NPU 1：
export ASCEND_RT_VISIBLE_DEVICES=1

python3 third_party/FlagPrism/Debugger/examples/abs_level1.py
```

`ASCEND_RT_VISIBLE_DEVICES=1` 把物理 NPU 1 暴露给当前进程；进程内通常仍以逻辑
device 0 访问它。不要再同时用物理编号作为进程内 device index。

如果后端无法从设备查询 SoC，可显式指定：

```bash
export TRITON_ASCEND_ARCH=Ascend910B4
```

可直接运行的示例：

| Example | Configuration | Output directory |
| --- | --- | --- |
| `examples/abs_level1.py` | `level=1, addr_level=1` | `/tmp/flagtree_debugger_level1_example` |
| `examples/abs_level2.py` | `level=2, addr_level=2` | `/tmp/flagtree_debugger_level2_example` |
| `examples/softmax_dim1_level1.py` | `level=1, addr_level=1` | `/tmp/flagtree_debugger_softmax_level1_example` |
| `examples/softmax_dim1_level2.py` | `level=2, addr_level=2` | `/tmp/flagtree_debugger_softmax_level2_example` |

## Backend Support

| Capability | Current status |
| --- | --- |
| Host/component integration | Host API 2.x 与 capability 协商，不按 FlagTree 3.5/3.6 版本号硬编码 |
| Statement annotation and static metadata | 通过通用 compiler/frontend callback 接入 |
| Summary/full-value instrumentation | 依赖目标后端可 lowering 的 TTIR operation |
| Hidden control pointer and post-kernel export | 当前接入并验证 Ascend/CANN9 与 Tianshu/CoreX 4.4 LLVM 22 |
| CUDA/HIP/MUSA runtime collection | 协议枚举和 adapter 接口已预留，尚未接通 launcher、同步和 transfer 实现 |
| Tianshu/CoreX runtime collection | 复用协议和 hidden pointer；通过 CUDA-compatible driver API 动态加载 CoreX transfer，实现 summary/memory/full dump；device-cycle timeline 暂未启用 |

Debugger 激活时，只有 metadata 明确设置 `debug_launch_hidden_arg=True` 的 Ascend/CANN
或 Tianshu/CoreX kernel 才会附加 hidden control pointer。未接入的后端不会因为全局 Debugger 状态而
改变 kernel launch ABI。

## Current Limitations

- summary 插桩主要依赖通用 TTIR arithmetic/reduce/store。
- memory address 采集使用 Debugger 专用
  `flagtree_debug.capture_memory_address` operation，需要后端提供 lowering。
- 当前 CANN9 与 Tianshu/CoreX 4.4 LLVM 22 路径支持对可证明的
  `tt.addptr(tt.splat(base), offsets)` 指针链和 prefix mask 生成 lane-aware
  address summary。Tianshu/CoreX 对连续地址摘要使用标量 i64 地址计算；这是因为
  当前 CoreX TTIR 到 TTGIR 转换无法 legalize encoded `tensor<i64>` 上的
  `tensor.extract`，并非硬件地址宽度受限。无法完整分析时只报告可证明的地址信息。
- `addr_level=2` 只能用于后端支持完整 lane address lowering 的
  pointer/mask pattern；当前 CANN9 与 Tianshu/CoreX 4.4 LLVM 22 支持的 pattern 会生成
  `*_memory_address.npy`。
- Debug hidden-argument ABI 尚未穿透任意 Triton call graph。含不可安全
  改写 call signature 的 helper/callee 会保持 metadata-only，避免 Debugger 改变
  原 kernel 语义。
- 当前报告可按 `logical_instance_id` 对比 program instance，并在 Level 2 地址
  artifact 中查看 lane 地址；尚未直接输出 warp id、异常 lane 聚类或“可疑访存
  上下文”结论。
- fp16/bf16/fp32 精度转换误差、除数接近零和负数开方等专项异常诊断尚未实现。
  当前可先用 Level 2 value artifact 离线比较；计划见 [todo.md](todo.md)。

新增设备后端时，必须验证 summary writer、hidden argument、transfer engine
和 `capture_memory_address` lowering。

## Architecture

FlagTree 主仓库仅保留必要的集成点：

- `flagtree._flagprism`：Host API 2.x、capability 校验、Debugger 组件注册和
  编译/运行时生命周期边界。
- `flagtree.language.debug_collect_start/end`：FlagTree language 的采集 marker。
- Ascend compiler/launcher hook：传递 Debugger metadata 和 hidden control pointer。
- Tianshu/CoreX compiler/launcher hook：使用同一 metadata/hidden pointer 契约；runtime transfer
  通过 `libcuda.so.1` 兼容层动态解析。

Debugger 的主要实现位于当前目录：

- `include/Debugger/`：跨模块协议与公共 C++ 接口。
- `lib/Frontend/`：前端配置和 launch/ABI 接线。
- `lib/Metadata/`：collect region 解析、`op_id` 分配和静态 metadata。
- `lib/Instrumentation/`：summary、memory event、full dump 和 timeline 插桩。
- `lib/Runtime/`：control block、device buffer、transfer engine 和 post-kernel export。
- `lib/Decode/`：record decode、statement/op 组织和文本/JSON 报告。

内部 compiler binding 位于 `triton._C.libtriton.debugger`，runtime binding 位于
`flagtree.debugger._native`。它们是实现细节，用户不应直接导入。

组件兼容性由 Host API major/minor 和 required capabilities 判断，不直接绑定
FlagTree 的 3.5.x 或 3.6.x release series。同一 FlagPrism revision 只有在目标
FlagTree 提供组件声明的全部 capability 时才会加载；缺失 capability 会在 import
阶段给出明确错误。

跨模块协议和公共 C++ 契约以 `include/Debugger/` 为准。
