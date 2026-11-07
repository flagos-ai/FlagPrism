// FlagPrism：HIP transfer lifecycle on default and non-default streams.
#include "BackendAdapter.h"
#include "Debugger/Runtime/TransferEngine.h"
#include <cstring>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <stdexcept>

using namespace mlir::flagtree::debugger;

static void check(hipError_t error) {
  if (error != hipSuccess)
    throw std::runtime_error(hipGetErrorString(error));
}
static void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}

int main() {
  int count = 0;
  check(hipGetDeviceCount(&count));
  require(count > 0, "No HIP devices; hardware test cannot pass");
  for (int device = 0; device < 1; ++device) {
    check(hipSetDevice(device));
    for (bool nonDefault : {false, true}) {
      hipStream_t stream = nullptr;
      if (nonDefault)
        check(hipStreamCreate(&stream));
      const auto handle = reinterpret_cast<uint64_t>(stream);
      auto options = makeTransferEngineOptions(BackendKind::HIP, handle);
      require(options.driverKind == TransferDriverKind::HIP,
              "Wrong backend mapping");
      auto adapter = createRuntimeBackendAdapter(options);
      require(adapter->isAvailable(), "HIP adapter unavailable");
      adapter->setDevice(device);
      constexpr size_t bytes = 4097;
      auto *src = static_cast<unsigned char *>(adapter->allocateHost(bytes));
      auto *dst = static_cast<unsigned char *>(adapter->allocateHost(bytes));
      void *dev = adapter->allocateDevice(bytes);
      for (size_t i = 0; i < bytes; ++i)
        src[i] = (i * 17 + 3) % 251;
      adapter->copyHostToDevice(dev, src, bytes, handle);
      adapter->copyDeviceToHost(dst, dev, bytes, handle);
      adapter->synchronize(handle);
      require(std::memcmp(src, dst, bytes) == 0,
              "Device roundtrip corrupted data");
      adapter->memsetDevice(dev, 0x5a, bytes, handle);
      adapter->copyDeviceToHost(dst, dev, bytes, handle);
      adapter->synchronize(handle);
      for (size_t i = 0; i < bytes; ++i)
        require(dst[i] == 0x5a, "Memset mismatch");
      adapter->freeDevice(dev);
      adapter->freeHost(src);
      adapter->freeHost(dst);

      auto engine = createTransferEngine(options);
      BufferMeta meta{};
      meta.backendKind = BackendKind::HIP;
      meta.deviceId = device;
      DebugBufferPlan plan;
      plan.recordCapacity = 17;
      auto ctx = engine->prepare(meta, plan, {});
      engine->initHeader(ctx);
      require(engine->hiddenArg(ctx) ==
                  reinterpret_cast<uint64_t>(ctx.deviceCtrlPtr),
              "Hidden argument pointer mismatch");
      engine->asyncExport(ctx);
      engine->waitAsyncExport(ctx);
      auto run = engine->syncExport(ctx);
      require(run.rawBuffer.size() == ctx.bufferSize, "Export size mismatch");
      RingBufferHeader header{};
      std::memcpy(&header, run.rawBuffer.data(), sizeof(header));
      require(header.capacity == 17 && header.writeIdx == 0 &&
                  header.overflowCount == 0,
              "Header initialization/export mismatch");
      engine->release(ctx);
      require(ctx.deviceCtrlPtr == nullptr && ctx.hostBufferPtr == nullptr,
              "Release did not clear context");
      if (nonDefault)
        check(hipStreamDestroy(stream));
      std::cout << "PASS device=" << device
                << " non_default_stream=" << nonDefault << '\n';
    }
  }
}

// FlagPrism end
