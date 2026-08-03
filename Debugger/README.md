# FlagPrism Debugger

FlagPrism Debugger 用于观察 Triton kernel 内部的数值、内存访问和 operation
执行状态。它将编译期静态 metadata 与 device 运行期记录关联，导出文本、
JSON 和 level 2 tensor artifact，用于定位数值异常、越界访问和 kernel 内部
数据流问题。

## Public API

Debugger 唯一的公开 Python 导入路径是：

```python
import flagtree.debugger as debugger
```

不再提供 `triton.debugger` 公开命名空间。Triton JIT kernel 内的采集边界仍然是
Triton language operation：

```python
tl.debug_collect_start(level=1, addr_level=1)
# Triton operations to collect
tl.debug_collect_end()
```

`flagtree.debugger` 负责 Python 配置、编译模式和导出；
`tl.debug_collect_start/end` 只负责界定 `@triton.jit` 内需要采集的 IR 区域。

## Build And Install

Debugger 位于 `third_party/FlagPrism/Debugger`，由 FlagTree 主工程统一
编译和打包，不再单独发布 wheel。

```bash
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

`TRITON_BUILD_FLAGPRISM=ON` 会在同一 CMake graph 中编译 Debugger、Profiler 及其 native
runtime。设为 `OFF` 时构建 core-only FlagTree，该构建不包含两个工具 package。

## Quick Start

```python
from pathlib import Path

import torch
import torch_npu
import triton
import triton.language as tl
import flagtree.debugger as debugger


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

    tl.debug_collect_start(level=1, addr_level=1)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.abs(x) + 1.0
    tl.store(y_ptr + offsets, y, mask=mask)
    tl.debug_collect_end()


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
- `addr_level=2`：在后端支持的 pointer/mask pattern 上导出完整 lane
  address。

`tl.debug_collect_start()` 可以为当前 region 指定 level。`addr_level=None`
时继承 `debugger.activate()` 的地址采集等级：

```python
tl.debug_collect_start(level=1)
```

长进程、notebook 或测试套件可在不再编译 debug kernel 时关闭 pipeline：

```python
debugger.deactivate()
```

一次性脚本通常无需调用 `deactivate()`。

## Collected Data

### Static Metadata

编译期 metadata 通常包含：

- kernel id 和 kernel name
- `scope_id`、`op_id` 和 statement id
- MLIR operation 名称
- Python source location 和 Triton statement
- operand/result dtype
- shape、layout 和可推导的 memory semantics
- load/store 的 address space、access type、element bytes、alignment 和 mask

分配了 `op_id` 但没有 runtime record 的 operation 会作为 static-only context
保留，例如部分 `tt.splat` 和 `tt.addptr`。

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

### Level 2 Full Dump

level 2 将完整 payload 写入 artifact 目录：

```text
<report-stem>_artifacts/
  tensor_index.json
  op<id>_inst<id>_rec<id>_value.npy
  op<id>_inst<id>_rec<id>_memory_address.npy
```

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
`op_log_report_path`、`raw_records_path`、`meta`、`runtime_metadata` 和
`decoded`。只有对应输出被启用或生成时，相应 path 字段才存在。

如果 `overflow_count` 非零，说明 `record_capacity` 不足，报告可能不完整。
应增大 capacity 并重新运行。level 2 不允许从 overflowed buffer 导出
artifact。

## Ascend Runtime

在已安装 FlagTree 的 Ascend 环境中：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export FLAGTREE_BACKEND=ascend
# 多卡机器可按需选择可用物理设备，例如 NPU 1：
export ASCEND_VISIBLE_DEVICES=1

python3 third_party/FlagPrism/Debugger/examples/abs_level1.py
```

如果后端无法从设备查询 SoC，可显式指定：

```bash
export TRITON_ASCEND_ARCH=Ascend910B4
```

可直接运行的示例：

- `examples/abs_level1.py`
- `examples/abs_level2.py`
- `examples/softmax_dim1_level1.py`
- `examples/softmax_dim1_level2.py`

## Backend Limitations

- summary 插桩主要依赖通用 TTIR arithmetic/reduce/store。
- memory address 采集使用 Debugger 专用
  `flagtree_debug.capture_memory_address` operation，需要后端提供 lowering。
- 当前 CANN9 `addr_level=1` 支持对可证明的
  `tt.addptr(tt.splat(base), offsets)` 指针链和 prefix mask 生成 lane-aware
  address summary。无法完整分析时只报告可证明的地址信息。
- `addr_level=2` 只能用于后端支持完整 lane address lowering 的
  pointer/mask pattern。
- Debug hidden-argument ABI 尚未穿透任意 Triton call graph。含不可安全
  改写 call signature 的 helper/callee 会保持 metadata-only，避免 Debugger 改变
  原 kernel 语义。

新增设备后端时，必须验证 summary writer、hidden argument、transfer engine
和 `capture_memory_address` lowering。

## Architecture

FlagTree 主仓库仅保留必要的集成点：

- `triton._flagprism`：Debugger 组件注册和编译/运行时生命周期边界。
- `tl.debug_collect_start/end`：Triton language 的采集 marker。
- Ascend compiler/launcher hook：传递 Debugger metadata 和 hidden control pointer。

Debugger 的主要实现位于当前目录：

- `Frontend/`：前端 marker、配置和 launch/ABI 接线。
- `Metadata/`：collect region 解析、`op_id` 分配和静态 metadata。
- `Instrumentation/`：summary、memory event、full dump 和 timeline 插桩。
- `Runtime/`：control block、device buffer、transfer engine 和 post-kernel export。
- `Decode/`：record decode、statement/op 组织和文本/JSON 报告。

内部 compiler binding 位于 `triton._C.libtriton.debugger`，runtime binding 位于
`flagtree.debugger._native`。它们是实现细节，用户不应直接导入。

跨模块协议和公共 C++ 契约以 `include/Debugger/` 为准。
