// FlagPrism：HIP device buffers, pinned host staging and stream-ordered
// transfers.
#if FLAGTREE_DEBUGGER_HAS_HIP_RUNTIME
class HipRuntimeBackendAdapter final : public RuntimeBackendAdapter {
  static void check(hipError_t status, const char *operation) {
    if (status != hipSuccess)
      failRuntime(std::string(operation) + " failed (" +
                  std::to_string(static_cast<int>(status)) +
                  "): " + hipGetErrorString(status));
  }
  static hipStream_t stream(uint64_t handle) {
    return reinterpret_cast<hipStream_t>(handle);
  }

public:
  TransferDriverKind driverKind() const override {
    return TransferDriverKind::HIP;
  }
  const char *name() const override { return "hip"; }
  bool isAvailable() const override {
    int count = 0;
    return hipGetDeviceCount(&count) == hipSuccess && count > 0;
  }
  void setDevice(uint32_t device) override {
    check(hipSetDevice(device), "hipSetDevice");
  }
  void *allocateDevice(size_t bytes) override {
    void *ptr = nullptr;
    check(hipMalloc(&ptr, bytes), "hipMalloc");
    return ptr;
  }
  void freeDevice(void *ptr) override { check(hipFree(ptr), "hipFree"); }
  void *allocateHost(size_t bytes) override {
    void *ptr = nullptr;
    check(hipHostMalloc(&ptr, bytes, 0), "hipHostAlloc");
    return ptr;
  }
  void freeHost(void *ptr) override { check(hipHostFree(ptr), "hipFreeHost"); }
  void memsetDevice(void *ptr, int value, size_t bytes,
                    uint64_t handle) override {
    check(hipMemsetAsync(ptr, value, bytes, stream(handle)), "hipMemsetAsync");
    if (handle == 0)
      synchronize(handle);
  }
  void copyHostToDevice(void *dst, const void *src, size_t bytes,
                        uint64_t handle) override {
    check(
        hipMemcpyAsync(dst, src, bytes, hipMemcpyHostToDevice, stream(handle)),
        "hipMemcpyAsync H2D");
    // initHeader passes a stack header and requires completion before launch.
    if (handle == 0)
      synchronize(handle);
  }
  void copyDeviceToHost(void *dst, const void *src, size_t bytes,
                        uint64_t handle) override {
    check(
        hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToHost, stream(handle)),
        "hipMemcpyAsync D2H");
  }
  void synchronize(uint64_t handle) override {
    check(hipStreamSynchronize(stream(handle)), "hipStreamSynchronize");
  }
};
#endif
// FlagPrism end
