include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../..)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../..)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
add_subdirectory(include)
add_subdirectory(lib)
# Keep production registration separate from the legacy lit-only test pass.
add_triton_library(ProtonRegistration
  lib/Integration/Registration.cpp

  LINK_LIBS PUBLIC
  ProtonIR
  ProtonGPUIR
  ProtonGPUTransforms
  ProtonToProtonGPU
  ProtonGPUToLLVM
  ProtonAMDGPUToLLVM
  ProtonNVIDIAGPUToLLVM
)
# Parent triton-opt keeps running test/Proton; only the pass implementation and
# its target move into FlagPrism.
add_subdirectory(test)
if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonProton ${CMAKE_CURRENT_SOURCE_DIR}/triton_proton.cc LINK_LIBS ProtonToProtonGPU ProtonGPUToLLVM ProtonAMDGPUToLLVM ProtonNVIDIAGPUToLLVM ProtonAnalysis)
  target_link_libraries(TritonProton PRIVATE Python3::Module pybind11::headers)
endif()
