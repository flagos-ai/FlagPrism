# FlagPrism Profiler 与 Debugger 芯片适配需求

## 1. 目标

芯片适配完成后，需要支持：

- Profiler 启动和停止芯片性能采集，获取 kernel 执行信息和硬件指标；
- Debugger 分配设备调试内存，将调试指针传给 kernel，并在 kernel 执行后取回调试数据；
- 未开启 Profiler 或 Debugger 时，不改变 kernel 的编译和执行行为。

其中，芯片硬件性能信息依赖厂商提供的 Profiler 能力。厂商需要提供性能采集 API 或命令行工具，并提供可解析的结果。FlagPrism 负责调用厂商 Profiler、关联 Triton 算子，以及将结果转换为统一格式。

### 1.1 信息来源

当前设计中的信息分为三类：

| 类型 | 可以获得的信息 | 是否依赖厂商 Profiler |
| --- | --- | --- |
| 基础插桩信息 | kernel 和算子名称、源码位置、算子 ID、数据类型、shape、layout、访存类型 | 否 |
| 数值摘要 | 元素数量、NaN 数量、Inf 数量、零值数量、有限值均值、最小值、最大值、L2 Norm | 否 |
| 芯片性能信息 | kernel 设备执行时间、计算单元利用率、带宽、Cache、指令统计、硬件计数器 | 是 |

只要芯片支持 Debugger 的隐藏参数和调试记录写入，基础插桩信息和数值摘要就可以由 FlagPrism 自己获得。

芯片性能指标由厂商决定支持范围。厂商未提供的指标会标记为不支持或不可用，不影响基础插桩信息和数值摘要的采集。

## 2. Profiler 适配接口

Profiler 适配分为三个步骤：

```text
1. 声明厂商支持哪些指标，并生成本次采集配置
                         ↓
2. 启动厂商 Profiler，在程序运行期间采集数据
                         ↓
3. 解析厂商输出，生成统一的 kernel 性能结果
```

三个步骤分别对应 2.1、2.2 和 2.3。完整调用顺序为：

```text
makePlan → doSetMode → doStart → startOp/stopOp → doStop → import
```

### 2.1 声明能力并生成采集配置

`VendorAdapter` 负责说明厂商支持哪些指标，并把用户请求转换成本次实际使用的采集配置。

实现以下 6 个接口：

```cpp
class VendorAdapter {
public:
  // 返回后端名称，例如 "cann"。
  virtual std::string getName() const = 0;

  // 返回设备类型。
  virtual DeviceType getDeviceType() const = 0;

  // 返回支持的硬件指标名称。
  virtual std::vector<std::string>
  getSupportedVendorMetrics() const = 0;

  // 根据用户请求生成实际采集方案。
  virtual VendorProfilePlan
  makePlan(const VendorProfileOptions &options) const = 0;

  // 返回芯片 Profiler 实例。
  virtual Profiler *getRuntimeProfiler() const = 0;

  // 创建性能数据导入器。
  virtual std::unique_ptr<VendorMetricsImporter>
  createImporter() const = 0;
};
```

其中，`getRuntimeProfiler()` 返回 2.2 中的采集对象，`createImporter()` 返回 2.3 中的解析对象。

#### 采集计划（VendorProfilePlan）

`VendorProfilePlan` 不是 kernel 执行计划，而是本次 Profiler 实际使用的采集配置。

生成过程如下：

```text
用户请求的指标和参数
        ↓
makePlan() 检查厂商 Profiler 的能力
        ↓
生成实际启用的指标、未启用的指标和原因
```

用户请求 `VendorProfileOptions` 包含：

- 是否采集 kernel 基础运行信息；
- 希望采集的硬件指标；
- device ID、输出位置等厂商参数。

`makePlan()` 返回的 `VendorProfilePlan` 包含：

- 本次实际启用的指标；
- 本次无法启用的指标；
- 传给厂商 Profiler 的配置；
- 指标无法启用的原因。

例如，用户请求 `aicore`、`bandwidth` 和 `cache`，而昇腾当前只支持前两项，则生成的 plan 为：

```text
enabled  = [aicore, bandwidth]
disabled = [cache]
reason   = [当前 Profiler 不支持 cache 指标]
```

后续采集和结果解析都使用同一个 plan。

### 2.2 启动采集并标记算子

这一阶段负责调用厂商 Profiler，并在 Triton 算子开始和结束时添加标记。

实现以下接口：

```cpp
class Profiler {
protected:
  // 启动芯片性能采集。
  virtual void doStart() = 0;

  // 刷新当前采集数据。
  virtual void doFlush() = 0;

  // 停止采集并释放资源。
  virtual void doStop() = 0;

  // 设置设备、指标、输出目录等采集参数。
  virtual void doSetMode(
      const std::vector<std::string> &options);
};
```

为了将 Triton 算子与芯片侧性能事件对应，还需要实现：

```cpp
class OpInterface {
public:
  // 算子开始执行。
  virtual void startOp(const Scope &scope) = 0;

  // 算子执行结束。
  virtual void stopOp(const Scope &scope) = 0;
};
```

运行过程为：设置采集参数、启动厂商 Profiler、标记算子开始和结束、停止厂商 Profiler。停止后需要保证原始性能数据已经完整生成。

### 2.3 解析厂商性能数据

该接口的作用是把厂商 Profiler 的原始结果转换成 FlagPrism 可以读取的统一结果。

实现以下接口：

```cpp
class VendorMetricsImporter {
public:
  virtual std::string getName() const = 0;

  virtual VendorProfileArtifact import(
      const SessionProfileMetadata &metadata,
      const VendorProfilePlan &plan) const = 0;
};
```

这三个结构体由 FlagPrism 定义，含义如下：

| 参数 | 含义 |
| --- | --- |
| `metadata` | 本次采集的基本信息，例如后端名称、设备信息、session 名称和运行配置 |
| `plan` | 本次需要采集的指标、厂商参数和原始结果位置 |
| 返回值 | 解析后的 kernel 事件、硬件指标、原始文件列表和错误原因 |

返回结果中的每条 kernel 记录包含：kernel 名称、device ID、stream ID、开始时间、结束时间、指标列表、采集状态和错误说明。

其中 `metrics` 保存厂商能够提供的性能数据，例如：

```text
ai_core_utilization = 85.2
memory_bandwidth_gbps = 720.5
cache_hit_rate = 0.91
```

`import()` 内需要完成：

- 读取芯片 Profiler 输出；
- 将时间统一转换为纳秒；
- 将指标与 kernel 或算子关联；
- 标记指标是否采集成功；
- 对不支持、未采集到或无法关联的数据给出原因。

解析结果至少包含：

- kernel 或算子名称；
- device ID 和 stream ID；
- 开始时间和结束时间，单位为纳秒；
- 芯片支持的硬件指标；
- 算子与芯片性能事件之间的关联信息。

厂商可以提供的性能信息包括 kernel 执行时间、计算单元利用率、内存带宽、Cache、指令统计和硬件计数器。厂商没有提供的数据会标记为不支持或不可用。

### 2.4 昇腾实现思路

昇腾上的实现方式如下：

1. 后端名称为 `cann`，设备类型为 `ASCEND`；
2. 通过 `aclprofInit`、`aclprofCreateConfig` 和 `aclprofStart` 启动性能采集；
3. 通过 MSTX Range 标记算子的开始和结束，并保存算子 ID 与 Range ID 的关系；
4. 通过 `aclprofStop`、`aclprofDestroyConfig` 和 `aclprofFinalize` 停止采集；
5. 通过 `msprof` 导出性能数据；
6. 解析导出的数据，将 AI Core、带宽、执行时间等指标关联到对应算子；
7. CANN 或 MSTX 接口不可用时，在结果中说明相应指标未采集成功。

## 3. Debugger 适配接口

### 3.1 设备运行时接口

实现以下 11 个接口：

```cpp
class RuntimeBackendAdapter {
public:
  // 返回驱动类型、后端名称和当前可用状态。
  virtual TransferDriverKind driverKind() const = 0;
  virtual const char *name() const = 0;
  virtual bool isAvailable() const = 0;

  // 分配和释放设备内存。
  virtual void *allocateDevice(size_t bytes) = 0;
  virtual void freeDevice(void *ptr) = 0;

  // 分配和释放 Host 内存。
  virtual void *allocateHost(size_t bytes) = 0;
  virtual void freeHost(void *ptr) = 0;

  // 初始化设备内存。
  virtual void memsetDevice(
      void *ptr,
      int value,
      size_t bytes,
      uint64_t streamHandle) = 0;

  // Host 到设备的数据传输。
  virtual void copyHostToDevice(
      void *deviceDst,
      const void *hostSrc,
      size_t bytes,
      uint64_t streamHandle) = 0;

  // 设备到 Host 的数据传输。
  virtual void copyDeviceToHost(
      void *hostDst,
      const void *deviceSrc,
      size_t bytes,
      uint64_t streamHandle) = 0;

  // 等待指定 stream 完成。
  virtual void synchronize(uint64_t streamHandle) = 0;
};
```

接口需要满足：

- 设备内存可以被 kernel 直接读写；
- Host 内存支持设备与 Host 之间的数据传输；
- 传入有效 stream 时，数据传输和 kernel 在同一 stream 上保持正确顺序；
- 同步完成后，Host 才开始读取调试数据；
- 所有内存和运行时资源在调试结束后释放。

### 3.2 Kernel 编译和启动要求

除上述 C++ 接口外，芯片编译器和 launcher 还需要支持：

1. 为开启 Debugger 的 kernel 增加一个设备指针参数 `__debug_ctrl_ptr`；
2. launcher 在 kernel 启动前分配并初始化调试缓冲区；
3. launcher 将调试缓冲区的设备地址作为最后一个参数传给 kernel；
4. kernel 将调试记录写入该缓冲区；
5. kernel 执行完成后，将缓冲区复制回 Host；
6. 未开启 Debugger 时，不增加该参数；
7. 调试缓冲区满时记录溢出状态，不能越界写入。

第一阶段只需要支持数值摘要采集，包括 NaN、Inf、最大值、最小值、均值等统计信息。

如果需要支持访存地址采集，还需要将 kernel 中的指针转换为实际设备地址，并输出有效线程的地址摘要或完整地址列表。

### 3.3 昇腾实现思路

昇腾上的 Debugger Runtime 使用以下 CANN 接口：

| 功能 | 昇腾实现 |
| --- | --- |
| 分配设备内存 | `aclrtMalloc` |
| 释放设备内存 | `aclrtFree` |
| 分配 Host 内存 | `aclrtMallocHost` |
| 释放 Host 内存 | `aclrtFreeHost` |
| 初始化设备内存 | `aclrtMemset` 或 `aclrtMemsetAsync` |
| Host 复制到设备 | `aclrtMemcpy` 或 `aclrtMemcpyAsync` |
| 设备复制到 Host | `aclrtMemcpy` 或 `aclrtMemcpyAsync` |
| 等待 stream 完成 | `aclrtSynchronizeStream` |

整体流程如下：

1. 使用 `aclrtMalloc` 分配调试缓冲区；
2. 初始化缓冲区头部和记录区域；
3. 将缓冲区设备地址作为 `__debug_ctrl_ptr` 传给 kernel；
4. kernel 将数值摘要、地址信息等调试记录写入缓冲区；
5. kernel 结束后，在同一 CANN stream 上将缓冲区复制到 Host；
6. 调用 `aclrtSynchronizeStream` 等待复制完成；
7. Host 解析调试记录并释放设备和 Host 内存。

## 4. 错误处理要求

- 每个芯片 Runtime 或 Profiler API 调用都要检查返回值；
- 错误信息需要包含失败的接口名称和错误码；
- 不支持的性能指标需要明确标记为不支持；
- 无法关联到 kernel 的指标需要明确标记为未关联；
- 不得使用全 0 数据表示采集成功；
- 内存分配、数据传输或 kernel 参数不一致时，应停止本次调试并返回错误。

## 5. 接口与代码文件对应关系

### 5.1 Profiler

| 内容 | 接口定义或公共结构 | 厂商实现及接线位置 |
| --- | --- | --- |
| 厂商能力与采集计划 | `Profiler/csrc/include/Profiler/Vendor/Adapter.h`、`Profiler/csrc/include/Profiler/Vendor/Mode.h` | 新增 `<Vendor>Adapter.h/.cpp`；昇腾对应 `CannAdapter.h/.cpp` |
| 启动、刷新和停止采集 | `Profiler/csrc/include/Profiler/Profiler.h` | 新增 `<Vendor>Profiler.h/.cpp`；昇腾对应 `CannProfiler.h/.cpp` |
| 算子开始和结束标记 | `Profiler/csrc/include/Context/Context.h` 中的 `OpInterface` | 在 `<Vendor>Profiler.cpp` 中实现 `startOp()` 和 `stopOp()` |
| 厂商数据导入 | `Profiler/csrc/include/Profiler/Vendor/Adapter.h` 中的 `VendorMetricsImporter` | 在 `<Vendor>Adapter.cpp` 或独立 Importer 文件中实现；昇腾实现在 `CannAdapter.cpp` 和 `CannProfiler.cpp` |
| 统一结果结构 | `Profiler/csrc/include/Data/Artifacts.h` | Importer 填充 `VendorProfileArtifact` |
| 设备类型 | `Profiler/common/include/Device.h` | 增加厂商设备类型，并在设备查询实现中接入 |
| Adapter 注册 | `Profiler/csrc/include/Profiler/Vendor/Adapter.h` 中的 `VendorAdapterRegistry` | 修改 `Profiler/csrc/lib/Profiler/Vendor/Adapter.cpp` |
| 自动选择 Profiler 后端 | Python `backend` 选择逻辑 | 修改 `Profiler/python/flagtree_profiler/profile.py` |
| 编译接入 | CMake 源文件列表 | 修改 `Profiler/csrc/CMakeLists.txt` |

### 5.2 Debugger

| 内容 | 接口定义或公共结构 | 厂商实现及接线位置 |
| --- | --- | --- |
| 后端类型和调试记录格式 | `Debugger/include/Debugger/Common/Protocol.h` | 增加厂商 `BackendKind` |
| 传输驱动类型 | `Debugger/include/Debugger/Runtime/TransferEngine.h` | 增加厂商 `TransferDriverKind` |
| 设备内存与数据传输接口 | `Debugger/lib/Runtime/BackendAdapter.h` | 实现 `<Vendor>RuntimeBackendAdapter`；昇腾实现位于 `Debugger/lib/Runtime/TransferEngine.cpp` |
| Runtime Adapter 创建 | `createRuntimeBackendAdapter()` | 在 `Debugger/lib/Runtime/TransferEngine.cpp` 中增加厂商分支 |
| 隐藏参数和调试记录 lowering | Debugger instrumentation passes | 修改 `Debugger/lib/Instrumentation/Passes.cpp` |
| 编译阶段接入 | Debugger compiler callback | 修改 `Debugger/python/flagtree_debugger/compiler.py`；需要新增 C++ binding 时修改 `Debugger/python/CompilerBindings.cpp` |
| 调试缓冲区准备和导出 | Debugger Python Runtime | 修改 `Debugger/python/flagtree_debugger/api.py` |
| FlagTree 通用 launch 接口 | FlagTree Host gateway | `FlagTree/python/triton/_flagprism.py` |
| 芯片 launcher 接线 | 各芯片 backend driver | 在芯片 backend 的 `driver.py` 中调用 Debugger launch context；昇腾对应 `FlagTree/third_party/ascend/backend/driver.py` |
| 厂商 SDK 探测和链接 | Debugger Runtime CMake | 新增厂商 CMake 文件，并修改 `Debugger/lib/Runtime/CMakeLists.txt` 和 `Debugger/lib/CMakeLists.txt`；昇腾对应 `Debugger/lib/Runtime/CANN.cmake` |
