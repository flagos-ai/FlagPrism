#include "Integration/Registration.h"

#include "Conversion/ProtonGPUToLLVM/Passes.h"
#include "Conversion/ProtonGPUToLLVM/ProtonAMDGPUToLLVM/Passes.h"
#include "Conversion/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/Passes.h"
#include "Conversion/ProtonToProtonGPU/Passes.h"
#include "Dialect/Proton/IR/Dialect.h"
#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "Dialect/ProtonGPU/Transforms/Passes.h"

#include "mlir/IR/DialectRegistry.h"

namespace mlir::triton::proton {

void registerFlagTreeProtonPassesAndDialects(
    mlir::DialectRegistry &registry) {
  registerConvertProtonToProtonGPU();
  gpu::registerConvertProtonNvidiaGPUToLLVM();
  gpu::registerConvertProtonAMDGPUToLLVM();
  gpu::registerAllocateProtonSharedMemoryPass();
  gpu::registerAllocateProtonGlobalScratchBufferPass();
  gpu::registerScheduleBufferStorePass();
  gpu::registerAddSchedBarriersPass();
  registry.insert<ProtonDialect, gpu::ProtonGPUDialect>();
}

} // namespace mlir::triton::proton
