// FlagPrism：MACA device buffers, pinned host staging and stream-ordered
// transfers.
#if FLAGTREE_DEBUGGER_HAS_MACA_RUNTIME
class MacaRuntimeBackendAdapter final : public RuntimeBackendAdapter {
  static void check(mcError_t status, const char *operation) {
    if (status != mcSuccess)
      failRuntime(std::string(operation) + ": " + mcGetErrorString(status));
  }
  static mcStream_t stream(uint64_t handle) {
    return reinterpret_cast<mcStream_t>(handle);
  }

public:
  TransferDriverKind driverKind() const override {
    return TransferDriverKind::MACA;
  }
  const char *name() const override { return "maca"; }
  bool isAvailable() const override { return true; }
  void setDevice(uint32_t device) override {
    check(mcSetDevice(device), "mcSetDevice");
  }
  void *allocateDevice(size_t bytes) override {
    void *ptr = nullptr;
    check(mcMalloc(&ptr, bytes), "mcMalloc");
    return ptr;
  }
  void freeDevice(void *ptr) override { check(mcFree(ptr), "mcFree"); }
  void *allocateHost(size_t bytes) override {
    void *ptr = nullptr;
    check(mcMallocHost(&ptr, bytes, 0), "mcHostAlloc");
    return ptr;
  }
  void freeHost(void *ptr) override { check(mcFreeHost(ptr), "mcFreeHost"); }
  void memsetDevice(void *ptr, int value, size_t bytes,
                    uint64_t handle) override {
    check(mcMemsetAsync(ptr, value, bytes, stream(handle)), "mcMemsetAsync");
    if (handle == 0)
      synchronize(handle);
  }
  void copyHostToDevice(void *dst, const void *src, size_t bytes,
                        uint64_t handle) override {
    check(mcMemcpyAsync(dst, src, bytes, mcMemcpyHostToDevice, stream(handle)),
          "mcMemcpyAsync H2D");
    // initHeader passes a stack header and requires completion before launch.
    if (handle == 0)
      synchronize(handle);
  }
  void copyDeviceToHost(void *dst, const void *src, size_t bytes,
                        uint64_t handle) override {
    check(mcMemcpyAsync(dst, src, bytes, mcMemcpyDeviceToHost, stream(handle)),
          "mcMemcpyAsync D2H");
  }
  void synchronize(uint64_t handle) override {
    check(mcStreamSynchronize(stream(handle)), "mcStreamSynchronize");
  }
};
#endif
// FlagPrism end
