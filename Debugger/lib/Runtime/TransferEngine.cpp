#include "Debugger/Runtime/TransferEngine.h"
#include "BackendAdapter.h"

#if FLAGTREE_DEBUGGER_HAS_CANN_RUNTIME
#include "acl/acl.h"
#include "acl/acl_rt.h"
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#if defined(__linux__)
#include <dlfcn.h>
#endif
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mlir {
namespace flagtree {
namespace debugger {
namespace {

const char *getDriverKindName(TransferDriverKind driverKind) {
  switch (driverKind) {
  case TransferDriverKind::HOST:
    return "host";
  case TransferDriverKind::CANN:
    return "cann";
  case TransferDriverKind::COREX:
    return "corex";
  }
  return "unknown";
}

uint64_t resolveStreamHandle(const DebugLaunchContext &ctx,
                             const TransferEngineOptions &options) {
  return ctx.streamHandle != 0 ? ctx.streamHandle : options.streamHandle;
}

uint64_t saturatingMul(uint64_t lhs, uint64_t rhs) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
    return std::numeric_limits<uint64_t>::max();
  return lhs * rhs;
}

uint32_t saturatingU32(uint64_t value) {
  return static_cast<uint32_t>(
      std::min<uint64_t>(value, std::numeric_limits<uint32_t>::max()));
}

[[noreturn]] void failRuntime(const std::string &message) {
  std::fprintf(stderr, "FlagPrism debugger runtime fatal error: %s\n",
               message.c_str());
  std::abort();
}

[[noreturn]] void
throwUnavailableBackend(const RuntimeBackendAdapter &adapter) {
  failRuntime(std::string("runtime backend adapter '") + adapter.name() +
              "' is not available in this build");
}

#if FLAGTREE_DEBUGGER_HAS_CANN_RUNTIME
[[noreturn]] void throwAclError(const char *call, int errorCode) {
  std::string message =
      std::string(call) + " failed with aclError=" + std::to_string(errorCode);
  if (const char *recent = aclGetRecentErrMsg(); recent && recent[0] != '\0') {
    message += ", recent_err=\"";
    message += recent;
    message += "\"";
  }
  failRuntime(message);
}

void checkAcl(aclError error, const char *call) {
  if (error != ACL_SUCCESS) {
    throwAclError(call, error);
  }
}

aclrtStream toAclrtStream(uint64_t streamHandle) {
  return reinterpret_cast<aclrtStream>(streamHandle);
}
#endif

class HostRuntimeBackendAdapter final : public RuntimeBackendAdapter {
public:
  TransferDriverKind driverKind() const override {
    return TransferDriverKind::HOST;
  }

  const char *name() const override { return "host"; }

  bool isAvailable() const override { return true; }

  void *allocateDevice(size_t bytes) override { return std::malloc(bytes); }

  void freeDevice(void *ptr) override { std::free(ptr); }

  void *allocateHost(size_t bytes) override { return std::malloc(bytes); }

  void freeHost(void *ptr) override { std::free(ptr); }

  void memsetDevice(void *ptr, int value, size_t bytes,
                    uint64_t streamHandle) override {
    (void)streamHandle;
    if (!ptr || bytes == 0) {
      return;
    }
    std::memset(ptr, value, bytes);
  }

  void copyHostToDevice(void *deviceDst, const void *hostSrc, size_t bytes,
                        uint64_t streamHandle) override {
    (void)streamHandle;
    if (!deviceDst || !hostSrc || bytes == 0) {
      return;
    }
    std::memcpy(deviceDst, hostSrc, bytes);
  }

  void copyDeviceToHost(void *hostDst, const void *deviceSrc, size_t bytes,
                        uint64_t streamHandle) override {
    (void)streamHandle;
    if (!hostDst || !deviceSrc || bytes == 0) {
      return;
    }
    std::memcpy(hostDst, deviceSrc, bytes);
  }

  void synchronize(uint64_t streamHandle) override { (void)streamHandle; }
};

class CannRuntimeBackendAdapter final : public RuntimeBackendAdapter {
public:
  TransferDriverKind driverKind() const override {
    return TransferDriverKind::CANN;
  }

  const char *name() const override { return "cann"; }

  bool isAvailable() const override {
#if FLAGTREE_DEBUGGER_HAS_CANN_RUNTIME
    return true;
#else
    return false;
#endif
  }

  void *allocateDevice(size_t bytes) override {
#if FLAGTREE_DEBUGGER_HAS_CANN_RUNTIME
    void *ptr = nullptr;
    checkAcl(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST),
             "aclrtMalloc");
    return ptr;
#else
    (void)bytes;
    throwUnavailableBackend(*this);
#endif
  }

  void freeDevice(void *ptr) override {
#if FLAGTREE_DEBUGGER_HAS_CANN_RUNTIME
    if (!ptr) {
      return;
    }
    checkAcl(aclrtFree(ptr), "aclrtFree");
#else
    (void)ptr;
    throwUnavailableBackend(*this);
#endif
  }

  void *allocateHost(size_t bytes) override {
#if FLAGTREE_DEBUGGER_HAS_CANN_RUNTIME
    void *ptr = nullptr;
    checkAcl(aclrtMallocHost(&ptr, bytes), "aclrtMallocHost");
    return ptr;
#else
    (void)bytes;
    throwUnavailableBackend(*this);
#endif
  }

  void freeHost(void *ptr) override {
#if FLAGTREE_DEBUGGER_HAS_CANN_RUNTIME
    if (!ptr) {
      return;
    }
    checkAcl(aclrtFreeHost(ptr), "aclrtFreeHost");
#else
    (void)ptr;
    throwUnavailableBackend(*this);
#endif
  }

  void memsetDevice(void *ptr, int value, size_t bytes,
                    uint64_t streamHandle) override {
#if FLAGTREE_DEBUGGER_HAS_CANN_RUNTIME
    if (!ptr || bytes == 0) {
      return;
    }
    if (streamHandle != 0) {
      checkAcl(aclrtMemsetAsync(ptr, bytes, value, bytes,
                                toAclrtStream(streamHandle)),
               "aclrtMemsetAsync");
      return;
    }
    checkAcl(aclrtMemset(ptr, bytes, value, bytes), "aclrtMemset");
#else
    (void)ptr;
    (void)value;
    (void)bytes;
    (void)streamHandle;
    throwUnavailableBackend(*this);
#endif
  }

  void copyHostToDevice(void *deviceDst, const void *hostSrc, size_t bytes,
                        uint64_t streamHandle) override {
#if FLAGTREE_DEBUGGER_HAS_CANN_RUNTIME
    if (!deviceDst || !hostSrc || bytes == 0) {
      return;
    }
    if (streamHandle != 0) {
      checkAcl(aclrtMemcpyAsync(deviceDst, bytes, hostSrc, bytes,
                                ACL_MEMCPY_HOST_TO_DEVICE,
                                toAclrtStream(streamHandle)),
               "aclrtMemcpyAsync(H2D)");
      return;
    }
    checkAcl(aclrtMemcpy(deviceDst, bytes, hostSrc, bytes,
                         ACL_MEMCPY_HOST_TO_DEVICE),
             "aclrtMemcpy(H2D)");
#else
    (void)deviceDst;
    (void)hostSrc;
    (void)bytes;
    (void)streamHandle;
    throwUnavailableBackend(*this);
#endif
  }

  void copyDeviceToHost(void *hostDst, const void *deviceSrc, size_t bytes,
                        uint64_t streamHandle) override {
#if FLAGTREE_DEBUGGER_HAS_CANN_RUNTIME
    if (!hostDst || !deviceSrc || bytes == 0) {
      return;
    }
    if (streamHandle != 0) {
      checkAcl(aclrtMemcpyAsync(hostDst, bytes, deviceSrc, bytes,
                                ACL_MEMCPY_DEVICE_TO_HOST,
                                toAclrtStream(streamHandle)),
               "aclrtMemcpyAsync(D2H)");
      return;
    }
    checkAcl(aclrtMemcpy(hostDst, bytes, deviceSrc, bytes,
                         ACL_MEMCPY_DEVICE_TO_HOST),
             "aclrtMemcpy(D2H)");
#else
    (void)hostDst;
    (void)deviceSrc;
    (void)bytes;
    (void)streamHandle;
    throwUnavailableBackend(*this);
#endif
  }

  void synchronize(uint64_t streamHandle) override {
#if FLAGTREE_DEBUGGER_HAS_CANN_RUNTIME
    if (streamHandle == 0) {
      return;
    }
    checkAcl(aclrtSynchronizeStream(toAclrtStream(streamHandle)),
             "aclrtSynchronizeStream");
#else
    (void)streamHandle;
    throwUnavailableBackend(*this);
#endif
  }
};

class CoreXRuntimeBackendAdapter final : public RuntimeBackendAdapter {
public:
  CoreXRuntimeBackendAdapter() { load(); }

  ~CoreXRuntimeBackendAdapter() override {
#if defined(__linux__)
    releaseRetainedPrimaryContext();
    if (library_) {
      dlclose(library_);
    }
#endif
  }

  TransferDriverKind driverKind() const override {
    return TransferDriverKind::COREX;
  }

  const char *name() const override { return "corex"; }

  bool isAvailable() const override { return loaded_; }

  void setDevice(uint32_t deviceId) override {
#if defined(__linux__)
    require(ctxGetCurrent_ != nullptr && ctxGetDevice_ != nullptr &&
                primaryCtxRetain_ != nullptr && primaryCtxRelease_ != nullptr &&
                ctxSetCurrent_ != nullptr,
            "cuCtxSetCurrent");

    void *currentContext = nullptr;
    int currentDevice = -1;
    if (ctxGetCurrent_(&currentContext) == 0 && currentContext != nullptr &&
        ctxGetDevice_(&currentDevice) == 0 &&
        currentDevice == static_cast<int>(deviceId)) {
      if (retainedPrimaryDevice_ >= 0 &&
          retainedPrimaryDevice_ != static_cast<int>(deviceId)) {
        releaseRetainedPrimaryContext();
      }
      return;
    }

    if (retainedPrimaryDevice_ == static_cast<int>(deviceId) &&
        retainedPrimaryContext_ != nullptr) {
      check(ctxSetCurrent_(retainedPrimaryContext_), "cuCtxSetCurrent");
      return;
    }

    void *primaryContext = nullptr;
    check(primaryCtxRetain_(&primaryContext, static_cast<int>(deviceId)),
          "cuDevicePrimaryCtxRetain");
    const Result setResult = ctxSetCurrent_(primaryContext);
    if (setResult != 0) {
      (void)primaryCtxRelease_(static_cast<int>(deviceId));
      failRuntime("cuCtxSetCurrent failed with CoreX driver error=" +
                  std::to_string(setResult) + " (library=" + loadedFrom_ + ")");
    }

    releaseRetainedPrimaryContext();
    retainedPrimaryDevice_ = static_cast<int>(deviceId);
    retainedPrimaryContext_ = primaryContext;
#else
    (void)deviceId;
    throwUnavailableBackend(*this);
#endif
  }

  void *allocateDevice(size_t bytes) override {
#if defined(__linux__)
    require(allocateDevice_ != nullptr, "cuMemAlloc");
    uint64_t ptr = 0;
    check(allocateDevice_(&ptr, bytes), "cuMemAlloc");
    return reinterpret_cast<void *>(static_cast<uintptr_t>(ptr));
#else
    (void)bytes;
    throwUnavailableBackend(*this);
#endif
  }

  void freeDevice(void *ptr) override {
#if defined(__linux__)
    if (!ptr) {
      return;
    }
    require(freeDevice_ != nullptr, "cuMemFree");
    check(freeDevice_(reinterpret_cast<uint64_t>(ptr)), "cuMemFree");
#else
    (void)ptr;
    throwUnavailableBackend(*this);
#endif
  }

  void *allocateHost(size_t bytes) override {
#if defined(__linux__)
    require(allocateHost_ != nullptr, "cuMemHostAlloc");
    void *ptr = nullptr;
    check(allocateHost_(&ptr, bytes, 0), "cuMemHostAlloc");
    return ptr;
#else
    (void)bytes;
    throwUnavailableBackend(*this);
#endif
  }

  void freeHost(void *ptr) override {
#if defined(__linux__)
    if (!ptr) {
      return;
    }
    require(freeHost_ != nullptr, "cuMemFreeHost");
    check(freeHost_(ptr), "cuMemFreeHost");
#else
    (void)ptr;
    throwUnavailableBackend(*this);
#endif
  }

  void memsetDevice(void *ptr, int value, size_t bytes,
                    uint64_t streamHandle) override {
#if defined(__linux__)
    if (!ptr || bytes == 0) {
      return;
    }
    const uint64_t devicePtr = reinterpret_cast<uint64_t>(ptr);
    if (streamHandle != 0 && memsetAsync_) {
      check(memsetAsync_(devicePtr, static_cast<unsigned char>(value), bytes,
                         reinterpret_cast<void *>(streamHandle)),
            "cuMemsetD8Async");
      return;
    }
    if (memset_) {
      check(memset_(devicePtr, static_cast<unsigned char>(value), bytes),
            "cuMemsetD8");
      return;
    }
    require(memsetAsync_ != nullptr, "cuMemsetD8");
    check(memsetAsync_(devicePtr, static_cast<unsigned char>(value), bytes,
                       reinterpret_cast<void *>(streamHandle)),
          "cuMemsetD8Async");
    if (streamHandle == 0) {
      check(synchronize_(nullptr), "cuStreamSynchronize(default)");
    }
#else
    (void)ptr;
    (void)value;
    (void)bytes;
    (void)streamHandle;
    throwUnavailableBackend(*this);
#endif
  }

  void copyHostToDevice(void *deviceDst, const void *hostSrc, size_t bytes,
                        uint64_t streamHandle) override {
#if defined(__linux__)
    if (!deviceDst || !hostSrc || bytes == 0) {
      return;
    }
    const uint64_t devicePtr = reinterpret_cast<uint64_t>(deviceDst);
    if (streamHandle != 0 && copyHostToDeviceAsync_) {
      check(copyHostToDeviceAsync_(devicePtr, hostSrc, bytes,
                                   reinterpret_cast<void *>(streamHandle)),
            "cuMemcpyHtoDAsync");
      return;
    }
    if (copyHostToDevice_) {
      check(copyHostToDevice_(devicePtr, hostSrc, bytes), "cuMemcpyHtoD");
    } else {
      require(copyHostToDeviceAsync_ != nullptr, "cuMemcpyHtoD");
      check(copyHostToDeviceAsync_(devicePtr, hostSrc, bytes, nullptr),
            "cuMemcpyHtoDAsync(default)");
      check(synchronize_(nullptr), "cuStreamSynchronize(default)");
    }
#else
    (void)deviceDst;
    (void)hostSrc;
    (void)bytes;
    (void)streamHandle;
    throwUnavailableBackend(*this);
#endif
  }

  void copyDeviceToHost(void *hostDst, const void *deviceSrc, size_t bytes,
                        uint64_t streamHandle) override {
#if defined(__linux__)
    if (!hostDst || !deviceSrc || bytes == 0) {
      return;
    }
    const uint64_t devicePtr = reinterpret_cast<uint64_t>(deviceSrc);
    if (streamHandle != 0 && copyDeviceToHostAsync_) {
      check(copyDeviceToHostAsync_(hostDst, devicePtr, bytes,
                                   reinterpret_cast<void *>(streamHandle)),
            "cuMemcpyDtoHAsync");
      return;
    }
    if (copyDeviceToHost_) {
      check(copyDeviceToHost_(hostDst, devicePtr, bytes), "cuMemcpyDtoH");
    } else {
      require(copyDeviceToHostAsync_ != nullptr, "cuMemcpyDtoH");
      check(copyDeviceToHostAsync_(hostDst, devicePtr, bytes, nullptr),
            "cuMemcpyDtoHAsync(default)");
      check(synchronize_(nullptr), "cuStreamSynchronize(default)");
    }
#else
    (void)hostDst;
    (void)deviceSrc;
    (void)bytes;
    (void)streamHandle;
    throwUnavailableBackend(*this);
#endif
  }

  void synchronize(uint64_t streamHandle) override {
#if defined(__linux__)
    if (streamHandle == 0) {
      return;
    }
    require(synchronize_ != nullptr, "cuStreamSynchronize");
    check(synchronize_(reinterpret_cast<void *>(streamHandle)),
          "cuStreamSynchronize");
#else
    (void)streamHandle;
    throwUnavailableBackend(*this);
#endif
  }

private:
#if defined(__linux__)
  using Result = int;
  using Init = Result (*)(unsigned int);
  using PrimaryCtxRetain = Result (*)(void **, int);
  using PrimaryCtxRelease = Result (*)(int);
  using CtxSetCurrent = Result (*)(void *);
  using CtxGetCurrent = Result (*)(void **);
  using CtxGetDevice = Result (*)(int *);
  using MemAlloc = Result (*)(uint64_t *, size_t);
  using MemFree = Result (*)(uint64_t);
  using HostAlloc = Result (*)(void **, size_t, unsigned int);
  using HostFree = Result (*)(void *);
  using Memset = Result (*)(uint64_t, unsigned char, size_t);
  using MemsetAsync = Result (*)(uint64_t, unsigned char, size_t, void *);
  using CopyH2D = Result (*)(uint64_t, const void *, size_t);
  using CopyH2DAsync = Result (*)(uint64_t, const void *, size_t, void *);
  using CopyD2H = Result (*)(void *, uint64_t, size_t);
  using CopyD2HAsync = Result (*)(void *, uint64_t, size_t, void *);
  using Synchronize = Result (*)(void *);

  template <typename Function>
  Function loadSymbol(std::initializer_list<const char *> names) {
    if (!library_) {
      return nullptr;
    }
    for (const char *name : names) {
      if (void *symbol = dlsym(library_, name)) {
        return reinterpret_cast<Function>(symbol);
      }
    }
    return nullptr;
  }

  void load() {
    const char *env = std::getenv("FLAGTREE_DEBUGGER_COREX_DRIVER_LIBRARY");
    std::vector<std::string> candidates;
    if (env && *env) {
      candidates.emplace_back(env);
    }
    candidates.emplace_back("libcuda.so.1");
    candidates.emplace_back("libcuda.so");
    candidates.emplace_back("/usr/local/corex-4.4.0/lib64/libcuda.so.1");
    candidates.emplace_back("/usr/local/corex-4.4.0/lib64/libcuda.so");
    candidates.emplace_back("/usr/local/corex/lib/libcuda.so.1");
    candidates.emplace_back("/usr/local/corex/lib/libcuda.so");

    for (const auto &candidate : candidates) {
      library_ = dlopen(candidate.c_str(), RTLD_LOCAL | RTLD_LAZY);
      if (library_) {
        loadedFrom_ = candidate;
        break;
      }
    }
    if (!library_) {
      return;
    }

    init_ = loadSymbol<Init>({"cuInit"});
    primaryCtxRetain_ =
        loadSymbol<PrimaryCtxRetain>({"cuDevicePrimaryCtxRetain"});
    primaryCtxRelease_ =
        loadSymbol<PrimaryCtxRelease>({"cuDevicePrimaryCtxRelease"});
    ctxSetCurrent_ = loadSymbol<CtxSetCurrent>({"cuCtxSetCurrent"});
    ctxGetCurrent_ = loadSymbol<CtxGetCurrent>({"cuCtxGetCurrent"});
    ctxGetDevice_ = loadSymbol<CtxGetDevice>({"cuCtxGetDevice"});
    allocateDevice_ = loadSymbol<MemAlloc>({"cuMemAlloc_v2", "cuMemAlloc"});
    freeDevice_ = loadSymbol<MemFree>({"cuMemFree_v2", "cuMemFree"});
    allocateHost_ = loadSymbol<HostAlloc>({"cuMemHostAlloc"});
    freeHost_ = loadSymbol<HostFree>({"cuMemFreeHost"});
    memset_ = loadSymbol<Memset>({"cuMemsetD8"});
    memsetAsync_ = loadSymbol<MemsetAsync>({"cuMemsetD8Async"});
    copyHostToDevice_ =
        loadSymbol<CopyH2D>({"cuMemcpyHtoD_v2", "cuMemcpyHtoD"});
    copyHostToDeviceAsync_ =
        loadSymbol<CopyH2DAsync>({"cuMemcpyHtoDAsync_v2", "cuMemcpyHtoDAsync"});
    copyDeviceToHost_ =
        loadSymbol<CopyD2H>({"cuMemcpyDtoH_v2", "cuMemcpyDtoH"});
    copyDeviceToHostAsync_ =
        loadSymbol<CopyD2HAsync>({"cuMemcpyDtoHAsync_v2", "cuMemcpyDtoHAsync"});
    synchronize_ = loadSymbol<Synchronize>({"cuStreamSynchronize"});

    loaded_ = init_ && primaryCtxRetain_ && primaryCtxRelease_ &&
              ctxSetCurrent_ && ctxGetCurrent_ && ctxGetDevice_ &&
              allocateDevice_ && freeDevice_ && allocateHost_ && freeHost_ &&
              (memset_ || memsetAsync_) &&
              (copyHostToDevice_ || copyHostToDeviceAsync_) &&
              (copyDeviceToHost_ || copyDeviceToHostAsync_) && synchronize_;
    if (loaded_) {
      check(init_(0), "cuInit");
    }
  }

  [[noreturn]] void unavailable(const char *call) const {
    failRuntime(std::string("CoreX driver symbol unavailable: ") + call);
  }

  void require(bool present, const char *call) const {
    if (!present) {
      unavailable(call);
    }
  }

  void check(Result result, const char *call) const {
    if (result != 0) {
      failRuntime(std::string(call) + " failed with CoreX driver error=" +
                  std::to_string(result) + " (library=" + loadedFrom_ + ")");
    }
  }

  void releaseRetainedPrimaryContext() {
    if (retainedPrimaryDevice_ >= 0 && primaryCtxRelease_) {
      (void)primaryCtxRelease_(retainedPrimaryDevice_);
    }
    retainedPrimaryDevice_ = -1;
    retainedPrimaryContext_ = nullptr;
  }

  void *library_ = nullptr;
  std::string loadedFrom_;
  Init init_ = nullptr;
  PrimaryCtxRetain primaryCtxRetain_ = nullptr;
  PrimaryCtxRelease primaryCtxRelease_ = nullptr;
  CtxSetCurrent ctxSetCurrent_ = nullptr;
  CtxGetCurrent ctxGetCurrent_ = nullptr;
  CtxGetDevice ctxGetDevice_ = nullptr;
  int retainedPrimaryDevice_ = -1;
  void *retainedPrimaryContext_ = nullptr;
  MemAlloc allocateDevice_ = nullptr;
  MemFree freeDevice_ = nullptr;
  HostAlloc allocateHost_ = nullptr;
  HostFree freeHost_ = nullptr;
  Memset memset_ = nullptr;
  MemsetAsync memsetAsync_ = nullptr;
  CopyH2D copyHostToDevice_ = nullptr;
  CopyH2DAsync copyHostToDeviceAsync_ = nullptr;
  CopyD2H copyDeviceToHost_ = nullptr;
  CopyD2HAsync copyDeviceToHostAsync_ = nullptr;
  Synchronize synchronize_ = nullptr;
#endif
#if !defined(__linux__)
  void load() {}
#endif
  bool loaded_ = false;
};

struct BackendTransferAllocation {
  void *deviceBuffer = nullptr;
  void *hostBuffer = nullptr;
  size_t bytes = 0;
  bool asyncExportPending = false;
  bool asyncCopySubmitted = false;
};

void synthesizeExportHeader(const DebugLaunchContext &ctx,
                            BackendTransferAllocation &allocation) {
  if (!ctx.runtimeMetadata.hasLaunchGrid ||
      ctx.runtimeMetadata.recordsPerInstance == 0 || !allocation.hostBuffer ||
      allocation.bytes < sizeof(RingBufferHeader)) {
    return;
  }

  auto *header = reinterpret_cast<RingBufferHeader *>(allocation.hostBuffer);
  uint64_t totalSlots =
      saturatingMul(ctx.runtimeMetadata.gridX, ctx.runtimeMetadata.gridY);
  totalSlots = saturatingMul(totalSlots, ctx.runtimeMetadata.gridZ);
  totalSlots =
      saturatingMul(totalSlots, ctx.runtimeMetadata.recordsPerInstance);
  const uint64_t capacity = header->capacity;
  const uint64_t overflow = totalSlots > capacity ? totalSlots - capacity : 0;

  header->writeIdx = saturatingU32(totalSlots);
  header->overflowCount = saturatingU32(overflow);
  if (overflow != 0)
    header->flags |= RB_FLAG_OVERFLOW;
  else
    header->flags &= ~static_cast<uint32_t>(RB_FLAG_OVERFLOW);
}

class RealTransferEngine final : public TransferEngine {
public:
  explicit RealTransferEngine(const TransferEngineOptions &options)
      : options_(options), adapter_(createRuntimeBackendAdapter(options)) {}

  DebugLaunchContext
  prepare(const BufferMeta &meta, const DebugBufferPlan &plan,
          const DebugRuntimeMetadata &runtimeMetadata) override {
    ensureAdapterReady(meta);

    DebugLaunchContext ctx;
    ctx.meta = meta;
    ctx.bufferPlan = plan;
    ctx.runtimeMetadata = runtimeMetadata;
    ctx.recordCapacity = plan.recordCapacity;
    ctx.streamHandle = options_.streamHandle;
    ctx.layout = computeBufferLayout(plan.recordCapacity, plan.recordSize,
                                     plan.payloadBytes);
    ctx.bufferSize = ctx.layout.totalBytes;

    auto allocation = std::make_unique<BackendTransferAllocation>();
    allocation->bytes = ctx.bufferSize;
    allocation->deviceBuffer = adapter_->allocateDevice(ctx.bufferSize);
    if (!allocation->deviceBuffer) {
      failRuntime("failed to allocate runtime device buffer");
    }

    allocation->hostBuffer = adapter_->allocateHost(ctx.bufferSize);
    if (!allocation->hostBuffer) {
      adapter_->freeDevice(allocation->deviceBuffer);
      failRuntime("failed to allocate runtime host buffer");
    }

    std::memset(allocation->hostBuffer, 0, ctx.bufferSize);
    ctx.deviceCtrlPtr = allocation->deviceBuffer;
    ctx.hostBufferPtr = allocation->hostBuffer;

    std::lock_guard<std::mutex> lock(mutex_);
    allocations_[ctx.deviceCtrlPtr] = std::move(allocation);
    return ctx;
  }

  uint64_t hiddenArg(const DebugLaunchContext &ctx) override {
    return reinterpret_cast<uint64_t>(ctx.deviceCtrlPtr);
  }

  void initHeader(const DebugLaunchContext &ctx) override {
    std::lock_guard<std::mutex> lock(mutex_);
    auto *allocation = findAllocationLocked(ctx);
    if (!allocation) {
      return;
    }

    std::memset(allocation->hostBuffer, 0, allocation->bytes);
    // Header initialization is part of launch setup, not payload capture. Keep
    // it synchronous so the instrumented kernel never observes stale allocation
    // contents when different backend launch APIs are mixed in one process.
    adapter_->memsetDevice(allocation->deviceBuffer, 0, allocation->bytes, 0);

    RingBufferHeader header{};
    header.writeIdx = 0;
    header.capacity = ctx.recordCapacity;
    header.overflowCount = 0;
    header.flags = RB_FLAG_NONE;
    header.recordSize = ctx.bufferPlan.recordSize;
    header.payloadOffset = static_cast<uint32_t>(ctx.layout.payloadOffset);
    header.reserved0 = 0;
    header.reserved1 = 0;

    std::memcpy(allocation->hostBuffer, &header, sizeof(header));
    adapter_->copyHostToDevice(allocation->deviceBuffer, &header,
                               sizeof(header), 0);
    allocation->asyncExportPending = false;
    allocation->asyncCopySubmitted = false;
  }

  DebugExportedRun syncExport(const DebugLaunchContext &ctx) override {
    DebugExportedRun run;
    run.meta = ctx.meta;
    run.runtimeMetadata = ctx.runtimeMetadata;

    std::lock_guard<std::mutex> lock(mutex_);
    auto *allocation = findAllocationLocked(ctx);
    if (!allocation) {
      return run;
    }

    copyDeviceToHostLocked(ctx, *allocation);
    auto *begin = reinterpret_cast<const uint8_t *>(allocation->hostBuffer);
    run.rawBuffer.assign(begin, begin + allocation->bytes);
    return run;
  }

  void asyncExport(const DebugLaunchContext &ctx) override {
    std::lock_guard<std::mutex> lock(mutex_);
    auto *allocation = findAllocationLocked(ctx);
    if (!allocation) {
      return;
    }
    const uint64_t streamHandle = resolveStreamHandle(ctx, options_);
    allocation->asyncCopySubmitted = false;
    if (options_.driverKind != TransferDriverKind::HOST && streamHandle != 0) {
      adapter_->copyDeviceToHost(allocation->hostBuffer,
                                 allocation->deviceBuffer, allocation->bytes,
                                 streamHandle);
      allocation->asyncCopySubmitted = true;
    }
    allocation->asyncExportPending = true;
  }

  void waitAsyncExport(const DebugLaunchContext &ctx) override {
    std::lock_guard<std::mutex> lock(mutex_);
    auto *allocation = findAllocationLocked(ctx);
    if (!allocation || !allocation->asyncExportPending) {
      return;
    }
    if (!allocation->asyncCopySubmitted) {
      copyDeviceToHostLocked(ctx, *allocation);
      return;
    }
    finalizeAsyncCopyLocked(ctx, *allocation);
  }

  void release(DebugLaunchContext &ctx) override {
    std::unique_ptr<BackendTransferAllocation> allocation;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = allocations_.find(ctx.deviceCtrlPtr);
      if (it != allocations_.end()) {
        allocation = std::move(it->second);
        allocations_.erase(it);
      }
    }

    if (allocation) {
      adapter_->freeHost(allocation->hostBuffer);
      adapter_->freeDevice(allocation->deviceBuffer);
    }

    ctx.deviceCtrlPtr = nullptr;
    ctx.hostBufferPtr = nullptr;
    ctx.bufferSize = 0;
    ctx.recordCapacity = 0;
    ctx.streamHandle = 0;
    ctx.layout = {};
  }

private:
  void ensureAdapterReady(const BufferMeta &meta) const {
    if (!adapter_ || !adapter_->isAvailable()) {
      failRuntime(std::string("transfer engine driver '") +
                  getDriverKindName(options_.driverKind) +
                  "' is not available");
    }
    if (options_.driverKind == TransferDriverKind::CANN &&
        meta.backendKind != BackendKind::CANN) {
      failRuntime("cann transfer driver requires BufferMeta.backendKind == "
                  "CANN");
    }
    if (options_.driverKind == TransferDriverKind::COREX &&
        meta.backendKind != BackendKind::TIANSHU) {
      failRuntime("corex transfer driver requires BufferMeta.backendKind == "
                  "TIANSHU");
    }
    if (options_.driverKind == TransferDriverKind::COREX) {
      adapter_->setDevice(meta.deviceId);
    }
#if FLAGTREE_DEBUGGER_HAS_CANN_RUNTIME
    if (options_.driverKind == TransferDriverKind::CANN) {
      aclrtContext context = nullptr;
      checkAcl(aclrtGetCurrentContext(&context), "aclrtGetCurrentContext");
      if (context == nullptr) {
        failRuntime("cann transfer driver requires a current ACL context; "
                    "caller must set device/context before prepare()");
      }
    }
#endif
  }

  BackendTransferAllocation *
  findAllocationLocked(const DebugLaunchContext &ctx) {
    auto it = allocations_.find(ctx.deviceCtrlPtr);
    if (it == allocations_.end()) {
      return nullptr;
    }
    return it->second.get();
  }

  void copyDeviceToHostLocked(const DebugLaunchContext &ctx,
                              BackendTransferAllocation &allocation) {
    const uint64_t streamHandle = resolveStreamHandle(ctx, options_);
    adapter_->copyDeviceToHost(allocation.hostBuffer, allocation.deviceBuffer,
                               allocation.bytes, streamHandle);
    allocation.asyncCopySubmitted = streamHandle != 0;
    finalizeAsyncCopyLocked(ctx, allocation);
  }

  void finalizeAsyncCopyLocked(const DebugLaunchContext &ctx,
                               BackendTransferAllocation &allocation) {
    const uint64_t streamHandle = resolveStreamHandle(ctx, options_);
    adapter_->synchronize(streamHandle);
    synthesizeExportHeader(ctx, allocation);
    allocation.asyncCopySubmitted = false;
    allocation.asyncExportPending = false;
  }

  TransferEngineOptions options_;
  std::unique_ptr<RuntimeBackendAdapter> adapter_;
  std::mutex mutex_;
  std::unordered_map<void *, std::unique_ptr<BackendTransferAllocation>>
      allocations_;
};

} // namespace

TransferDriverKind resolveTransferDriverKind(BackendKind backendKind) {
  switch (backendKind) {
  case BackendKind::CANN:
    return TransferDriverKind::CANN;
  case BackendKind::TIANSHU:
    return TransferDriverKind::COREX;
  case BackendKind::UNKNOWN:
  case BackendKind::CUDA:
  case BackendKind::HIP:
  case BackendKind::MUSA:
    return TransferDriverKind::HOST;
  }
  return TransferDriverKind::HOST;
}

TransferEngineOptions makeTransferEngineOptions(BackendKind backendKind,
                                                uint64_t streamHandle) {
  TransferEngineOptions options;
  options.driverKind = resolveTransferDriverKind(backendKind);
  options.streamHandle = streamHandle;
  return options;
}

std::unique_ptr<RuntimeBackendAdapter>
createRuntimeBackendAdapter(const TransferEngineOptions &options) {
  switch (options.driverKind) {
  case TransferDriverKind::HOST:
    return std::make_unique<HostRuntimeBackendAdapter>();
  case TransferDriverKind::CANN:
    return std::make_unique<CannRuntimeBackendAdapter>();
  case TransferDriverKind::COREX:
    return std::make_unique<CoreXRuntimeBackendAdapter>();
  }
  failRuntime(std::string("unsupported transfer driver '") +
              getDriverKindName(options.driverKind) + "'");
}

std::unique_ptr<TransferEngine>
createTransferEngine(const TransferEngineOptions &options) {
  return std::make_unique<RealTransferEngine>(options);
}

std::unique_ptr<TransferEngine> createTransferEngine(BackendKind backendKind,
                                                     uint64_t streamHandle) {
  return createTransferEngine(
      makeTransferEngineOptions(backendKind, streamHandle));
}

} // namespace debugger
} // namespace flagtree
} // namespace mlir
