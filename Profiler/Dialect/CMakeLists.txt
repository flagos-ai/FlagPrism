include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../..)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../..)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

set(_flagprism_vendor_lowering_libs)
if(FLAGPRISM_BUILD_VENDOR_LOWERING)
  if(FLAGPRISM_BACKEND STREQUAL "nvidia")
    # FlagPrism: NVIDIA-only builds need only the NVIDIA lowering library;
    # the all-backend build below keeps both existing vendor lowerings.
    list(APPEND _flagprism_vendor_lowering_libs ProtonNVIDIAGPUToLLVM)
  else()
    list(APPEND _flagprism_vendor_lowering_libs
      ProtonAMDGPUToLLVM
      ProtonNVIDIAGPUToLLVM)
  endif()
endif()

add_subdirectory(include)
add_subdirectory(lib)
# Keep backend API-compatibility switches on the generic instrumentation pass.
flagprism_apply_backend_compile_definitions(ProtonToProtonGPU)
# Keep production registration separate from the legacy lit-only test pass.
add_triton_library(ProtonRegistration
  lib/Integration/Registration.cpp

  LINK_LIBS PUBLIC
  ProtonIR
  ProtonGPUIR
  ProtonGPUTransforms
  ProtonToProtonGPU
  ProtonGPUToLLVM
  ${_flagprism_vendor_lowering_libs}
)
target_include_directories(ProtonRegistration PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
flagprism_apply_backend_compile_definitions(ProtonRegistration)
# Parent triton-opt keeps running test/Proton; only the pass implementation and
# its target move into FlagPrism.
add_subdirectory(test)
if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonProton
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_proton.cc
    LINK_LIBS ProtonToProtonGPU ProtonGPUToLLVM
      ${_flagprism_vendor_lowering_libs} ProtonAnalysis)
  target_link_libraries(TritonProton PRIVATE Python3::Module pybind11::headers)
  flagprism_apply_backend_compile_definitions(TritonProton)
endif()
