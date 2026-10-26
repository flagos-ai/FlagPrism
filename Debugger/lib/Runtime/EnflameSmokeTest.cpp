#include "BackendAdapter.h"
#include "Debugger/Runtime/TransferEngine.h"
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <tops_runtime.h>

using namespace mlir::flagtree::debugger;

static void check(topsError_t error) {
  if (error != topsSuccess)
    throw std::runtime_error(topsGetErrorString(error));
}
static void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}

int main() {
  int count = 0;
  check(topsGetDeviceCount(&count));
  require(count > 0, "No GCU devices; hardware test cannot pass");
  for (int device = 0; device < count; ++device) {
    check(topsSetDevice(device));
    for (bool nonDefault : {false, true}) {
      topsStream_t stream = nullptr;
      if (nonDefault)
        check(topsStreamCreate(&stream));
      const auto handle = reinterpret_cast<uint64_t>(stream);
      auto options = makeTransferEngineOptions(BackendKind::ENFLAME, handle);
      require(options.driverKind == TransferDriverKind::TOPS,
              "Wrong backend mapping");
      auto adapter = createRuntimeBackendAdapter(options);
      require(adapter->isAvailable(), "TOPS adapter unavailable");
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
      meta.backendKind = BackendKind::ENFLAME;
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
      DebugRuntimeMetadata grid;
      grid.hasLaunchGrid = true;
      grid.gridX = 2;
      grid.recordsPerInstance = 10;
      plan.payloadBytes = 32;
      bool rejected = false;
      try {
        auto invalid = engine->prepare(meta, plan, grid);
        engine->release(invalid);
      } catch (const std::invalid_argument &) {
        rejected = true;
      }
      require(rejected, "Native caller bypassed full-dump capacity guard");
      grid.recordsPerInstance = 1;
      plan.payloadBytes = static_cast<size_t>(1) << 32;
      rejected = false;
      try {
        auto invalid = engine->prepare(meta, plan, grid);
        engine->release(invalid);
      } catch (const std::invalid_argument &) {
        rejected = true;
      }
      require(rejected, "Native caller bypassed full-dump offset guard");
      if (nonDefault)
        check(topsStreamDestroy(stream));
      std::cout << "PASS device=" << device
                << " non_default_stream=" << nonDefault << '\n';
    }
  }
}
