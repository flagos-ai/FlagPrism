if(FLAGPRISM_FLAGTREE_SOURCE_DIR)
  # FlagPrism provides an unambiguous parent-tree root for bundled builds.
  include_directories(${FLAGPRISM_FLAGTREE_SOURCE_DIR}/third_party/nvidia/include)
elseif(FLAGTREE_SOURCE_DIR)
  include_directories(${FLAGTREE_SOURCE_DIR}/third_party/nvidia/include)
else()
  include_directories(${PROJECT_SOURCE_DIR}/third_party/nvidia/include)
endif()

add_triton_library(ProtonNVIDIAGPUToLLVM
    TargetInfo.cpp
    NvidiaPatternProtonGPUOpToLLVM.cpp
    ConvertProtonGPUToLLVM.cpp

    DEPENDS
    ProtonNvidiaGPUConversionPassIncGen

    LINK_LIBS PUBLIC
    ProtonGPUToLLVM
    TritonNVIDIAGPUToLLVM
)
