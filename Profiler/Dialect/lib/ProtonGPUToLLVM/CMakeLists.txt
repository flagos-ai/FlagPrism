add_triton_library(ProtonGPUToLLVM
    AllocateProtonGlobalScratchBuffer.cpp
    AllocateProtonSharedMemory.cpp
    PatternProtonGPUOpToLLVM.cpp
    Utility.cpp

    DEPENDS
    ProtonGPUConversionPassIncGen

    LINK_LIBS PUBLIC
    ProtonIR
    ProtonGPUIR
    ProtonAnalysis
)

if(FLAGPRISM_BUILD_VENDOR_LOWERING)
  add_subdirectory(ProtonNvidiaGPUToLLVM)
  add_subdirectory(ProtonAMDGPUToLLVM)
endif()
