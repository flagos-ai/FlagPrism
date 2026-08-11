# Debugger Include Tree

`third_party/FlagPrism/Debugger/include/Debugger` 目录保存 debugger 的公共契约。这里定义的是跨模块都要遵守的接口，而不是某个人的临时实现。

这些头文件由 FlagTree 主 CMake graph 从 FlagPrism submodule 编译。Debugger native
extension 与 `libtriton` 一起写入同一个 FlagTree wheel；主仓库通过 Host API 2.x
和 capability 驱动的 `flagtree._flagprism` 组件接口调用该模块。

目录划分：

- `Common/`：统一协议、record 布局、buffer header、运行时主键
- `Frontend/`：A 模块，Python 前端与 launch/ABI 接线，负责人华师
- `Metadata/`：B 模块，编译期作用域解析、`op_id` 分配、静态元数据，负责人华师
- `Instrumentation/`：C 模块，device 插桩、summary/memory event 记录，负责人颜臻
- `Runtime/`：F 模块，control block、ring buffer、导出与运行时上下文，负责人闫明
- `Decode/`：D 模块，解码与报告，负责人玉珏

对齐原则：

- 协议主键由 `kernel_id + op_id + logical_instance_id` 组成。
- 编译期静态信息统一由 B 输出到 `KernelDebugMetadata / TrackedOpTable`。
- 运行期动态数据统一由 C 写入 `Record`，由 F 导出，由 D 解码。
- 运行期 host 上下文和动态 tensor/buffer 信息统一由 A/F 通过 `DebugRuntimeMetadata` 传递。
- 不要在各模块内部各自重新定义一套 record 布局、buffer header 或 metadata schema。

并行开发入口：

- 统一公共头：`Debugger.h`
- 真实后端入口：`createTransferEngine()`
- 按 backend 选择后端入口：`createTransferEngine(BackendKind, streamHandle)`

Python 调试接口：

Debugger 默认通过 Python 侧接口开启，编译期会进入 debugger instrumentation
mode。只有编译 metadata 明确启用 hidden-argument ABI 的 Ascend/CANN kernel，
运行期才会为 launch 准备 `__debug_ctrl_ptr`，并在 kernel 结束后同步和导出报告；
metadata-only kernel 不改变原 launch ABI。

基本用法：

```python
import triton
import triton.language as tl
import flagtree.debugger as debugger
import flagtree.language as ftl

# 通常在 import 后配置并开启一次。后续哪些 Triton IR op 被记录，
# 由 @triton.jit 内部的 ftl.debug_collect_start/end 控制。
debugger.configure(
    output_dir="/tmp/flagtree_debugger_manual",
    record_capacity=4096,
    export_raw_records=False,
)
debugger.activate(level=1, addr_level=0)


@triton.jit
def kernel(x_ptr, y_ptr, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    ftl.debug_collect_start(level=1, addr_level=1)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.abs(x)
    tl.store(y_ptr + offsets, y, mask=mask)
    ftl.debug_collect_end()


kernel[(grid,)](...)
```

容器内构建命令：

先初始化 FlagPrism submodule，再直接构建 FlagTree。Debugger 的 Python、dialect、
passes 和 runtime native binding 会由同一次构建统一打包。编译期 binding 位于
`triton._C.libtriton.debugger`，runtime binding 位于
`flagtree.debugger._native`，后者不链接 `libtriton`。

构建只保留两种模式：`TRITON_BUILD_FLAGPRISM=ON` 联合构建 Debugger 和 Profiler，
`TRITON_BUILD_FLAGPRISM=OFF` 构建不含两个工具 package 的 core-only wheel。

从 host 侧触发容器内完整 rebuild：

```bash
docker exec flagtree-cann9-quan /bin/sh -c '
cd "${FLAGTREE_SOURCE_DIR:-/workspace/FlagTree}"
git submodule update --init --recursive
FLAGTREE_BACKEND=ascend TRITON_BUILD_FLAGPRISM=ON MAX_JOBS=16 \
python3 -m pip install . --no-build-isolation
'
```

如果已经 attach 到容器内部，执行等价命令：

```bash
cd "${FLAGTREE_SOURCE_DIR:-/workspace/FlagTree}"
git submodule update --init --recursive
FLAGTREE_BACKEND=ascend TRITON_BUILD_FLAGPRISM=ON MAX_JOBS=16 \
python3 -m pip install . --no-build-isolation
```

常用接口：

- `debugger.configure(...)`：设置 debugger 默认配置；未传字段保持当前值。
  支持字段包括：
  - `output_dir`：报告输出目录；传 `None` 可关闭文件导出。
  - `record_capacity`：ring buffer record 容量。
  - `export_mode`：导出模式，默认 `POST_KERNEL_EXPORT`。
  - `export_on_error`：kernel 报错时是否仍尝试导出。
  - `export_raw_records`：是否把 decoded raw records 额外写到 sidecar 文件。
- `debugger.get_config()`：查询当前默认配置。
- `debugger.reset_config()`：恢复默认配置。
- `debugger.activate(level=1, addr_level=0)`：开启进程级 debugger 模式。通常
  在 import 后调用一次；`level` 控制数值采集等级，`addr_level` 控制动态地址
  采集，默认 `0` 表示不插入地址采集。
- `ftl.debug_collect_start/end`：在 `@triton.jit` 内界定实际采集范围。Python
  侧 `activate` 只开启 debugger pipeline，不会记录普通 PyTorch/torch_npu 语句。
  `ftl.debug_collect_start(level=..., addr_level=...)` 可覆盖当前 region 的地址
  采集等级；不传 `addr_level` 时继承 `debugger.activate(...)` 的配置。
- `debugger.deactivate()`：关闭 debugger，并清理 launch hook。普通一次性脚本
  通常不需要调用；长进程、notebook 或测试套件中可用它避免影响后续 kernel。
- `debugger.take_exported_runs()`：取回本进程内已导出的 run 信息。
- `debugger.clear_exported_runs()`：清空本进程内已缓存的导出结果。

后端适配注意事项：

- summary record 的 device lowering 主要依赖通用 TTIR arithmetic/reduce/store。
- memory address event 依赖 debugger 专用
  `flagtree_debug.capture_memory_address` lowering；只有 `addr_level > 0` 才会插入
  该动态地址采集。当前 CANN9 与 Tianshu/CoreX 4.4 LLVM 22 路径在 `addr_level=1` 时会对可反向切片的
  `tt.addptr(tt.splat(base), offsets)` 指针链生成地址摘要：
  `first_addr / last_addr / min_addr / max_addr / active_lane_count /
  address_span_bytes`。该路径要求 offset 可证明为连续 lane offset，mask 为空、
  全 true，或形如 `offsets < limit` 的 prefix mask。无法匹配的指针/掩码形态会
  退回到单条 base/last aligned address 事件，保证 debugger 不破坏正常编译。
  新增后端时需要验证或重写
  `flagtree_debug.capture_memory_address` lowering。`level=2, addr_level=2` 会在
  CANN9 与 Tianshu/CoreX 4.4 LLVM 22 支持的 pointer/mask pattern 上额外导出完整 lane address `.npy`；不支持的
  pattern 在编译期报错，不生成不完整 artifact。

导出文件：

- 默认输出目录：`/tmp/flagtree_debugger_manual`。
- 主报告文件名包含脚本名、kernel 名、时间戳和 run id，例如
  `test_debug_abs_kernel_aiv_20260617_150006_507_run1.txt`。
- 主 `.txt/.json` 是 Triton statement 视图；同 stem 的
  `_op_log.txt/.json` 是 IR op 视图。
- Level 2 额外生成同 stem 的 `_artifacts/`，其中包含 `tensor_index.json` 和
  value/memory-address `.npy`。statement 文本只显示 artifact 文件名，完整路径
  由 index 和 Python run metadata 保存。
- 主报告默认不直接 dump decoded raw records。
- 需要调试 raw record 时，先使用
  `debugger.configure(export_raw_records=True)`，再在进程初始化阶段
  `debugger.activate(level=1)`，会额外生成
  `*_raw_records.txt`。

查看报告：

```bash
cat /tmp/flagtree_debugger_manual/<report-file-stem>.txt
cat /tmp/flagtree_debugger_manual/<report-file-stem>_raw_records.txt
```

用户语义、statement 报告格式、Ascend 运行环境和当前后端边界以
[`Debugger/README.md`](../../README.md) 为准。
