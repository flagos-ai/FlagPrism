#include "Integration/Registration.h"

#include "Conversion/ProtonGPUToLLVM/Passes.h"
#if !defined(FLAGPRISM_BACKEND_TIANSHU) && \
    !defined(FLAGPRISM_BACKEND_ASCEND)
#include "Conversion/ProtonGPUToLLVM/ProtonAMDGPUToLLVM/Passes.h"
#include "Conversion/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/Passes.h"
#endif
#include "Conversion/ProtonToProtonGPU/Passes.h"
#include "Dialect/Proton/IR/Dialect.h"
#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "Dialect/ProtonGPU/Transforms/Passes.h"

#include "mlir/IR/DialectRegistry.h"

namespace mlir::triton::proton {

void registerFlagTreeProtonPassesAndDialects(
  mlir::DialectRegistry &registry) {
  registerConvertProtonToProtonGPU();
#if !defined(FLAGPRISM_BACKEND_TIANSHU) && \
    !defined(FLAGPRISM_BACKEND_ASCEND)
  gpu::registerConvertProtonNvidiaGPUToLLVM();
  gpu::registerConvertProtonAMDGPUToLLVM();
#endif
  gpu::registerAllocateProtonSharedMemoryPass();
  gpu::registerAllocateProtonGlobalScratchBufferPass();
  gpu::registerScheduleBufferStorePass();
#if !defined(FLAGPRISM_BACKEND_TIANSHU) && \
    !defined(FLAGPRISM_BACKEND_ASCEND)
  gpu::registerAddSchedBarriersPass();
#endif
  registry.insert<ProtonDialect, gpu::ProtonGPUDialect>();
}

} // namespace mlir::triton::proton
