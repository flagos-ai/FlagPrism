set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls --name ProtonGPUToLLVM)
add_public_tablegen_target(ProtonGPUConversionPassIncGen)

if(FLAGPRISM_BUILD_VENDOR_LOWERING)
  add_subdirectory(ProtonNvidiaGPUToLLVM)
  add_subdirectory(ProtonAMDGPUToLLVM)
endif()
