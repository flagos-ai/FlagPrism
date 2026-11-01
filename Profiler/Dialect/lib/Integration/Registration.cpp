#include "Integration/Registration.h"

#include "Conversion/ProtonGPUToLLVM/Passes.h"
// FlagPrism：MACA does not compile CUDA or AMD device-counter lowering.
#if !defined(FLAGPRISM_BACKEND_METAX) &&                                       \
    !defined(FLAGPRISM_BACKEND_ENFLAME) &&                                     \
    !defined(FLAGPRISM_BACKEND_TIANSHU) &&                                     \
    !defined(FLAGPRISM_BACKEND_ASCEND) &&                                      \
    !defined(FLAGPRISM_BACKEND_MTHREADS) && !defined(FLAGPRISM_BACKEND_NVIDIA)
#include "Conversion/ProtonGPUToLLVM/ProtonAMDGPUToLLVM/Passes.h"
#include "Conversion/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/Passes.h"
#elif defined(FLAGPRISM_BACKEND_NVIDIA)
#include "Conversion/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/Passes.h"
#endif
// FlagPrism end
#include "Conversion/ProtonToProtonGPU/Passes.h"
#include "Dialect/Proton/IR/Dialect.h"
#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "Dialect/ProtonGPU/Transforms/Passes.h"

#include "mlir/IR/DialectRegistry.h"

namespace mlir::triton::proton {

void registerFlagTreeProtonPassesAndDialects(mlir::DialectRegistry &registry) {
  registerConvertProtonToProtonGPU();
// FlagPrism：MACA does not compile CUDA or AMD device-counter lowering.
#if !defined(FLAGPRISM_BACKEND_METAX) &&                                       \
    !defined(FLAGPRISM_BACKEND_ENFLAME) &&                                     \
    !defined(FLAGPRISM_BACKEND_TIANSHU) &&                                     \
    !defined(FLAGPRISM_BACKEND_ASCEND) &&                                      \
    !defined(FLAGPRISM_BACKEND_MTHREADS) && !defined(FLAGPRISM_BACKEND_NVIDIA)
  gpu::registerConvertProtonNvidiaGPUToLLVM();
  gpu::registerConvertProtonAMDGPUToLLVM();
#elif defined(FLAGPRISM_BACKEND_NVIDIA)
  gpu::registerConvertProtonNvidiaGPUToLLVM();
#endif
  // FlagPrism end
  gpu::registerAllocateProtonSharedMemoryPass();
  gpu::registerAllocateProtonGlobalScratchBufferPass();
  gpu::registerScheduleBufferStorePass();
// FlagPrism：MACA does not compile CUDA or AMD device-counter lowering.
#if !defined(FLAGPRISM_BACKEND_METAX) &&                                       \
    !defined(FLAGPRISM_BACKEND_ENFLAME) &&                                     \
    !defined(FLAGPRISM_BACKEND_TIANSHU) &&                                     \
    !defined(FLAGPRISM_BACKEND_ASCEND) &&                                      \
    !defined(FLAGPRISM_BACKEND_MTHREADS) && !defined(FLAGPRISM_BACKEND_NVIDIA)
  gpu::registerAddSchedBarriersPass();
#endif
  // FlagPrism end
  registry.insert<ProtonDialect, gpu::ProtonGPUDialect>();
}

} // namespace mlir::triton::proton
