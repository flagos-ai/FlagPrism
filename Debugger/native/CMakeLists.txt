if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  message(FATAL_ERROR
    "Debugger native bindings are built from the FlagTree CMake graph")
endif()

if(FLAGPRISM_FLAGTREE_SOURCE_DIR)
  set(FLAGTREE_SOURCE_DIR "${FLAGPRISM_FLAGTREE_SOURCE_DIR}")
else()
  set(FLAGTREE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../..")
endif()
set(FLAGPRISM_PYTHON_DIR "" CACHE PATH
    "Python package root receiving bundled FlagPrism extensions")
if(NOT FLAGPRISM_PYTHON_DIR)
  message(FATAL_ERROR
    "FLAGPRISM_PYTHON_DIR is required for the bundled debugger")
endif()

set(DEBUGGER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(DEBUGGER_INCLUDE_DIR "${DEBUGGER_ROOT}/include")
set(DEBUGGER_GENERATED_REL_DIR "include/Debugger/IR")
set(DEBUGGER_GENERATED_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/${DEBUGGER_GENERATED_REL_DIR}")
file(MAKE_DIRECTORY "${DEBUGGER_GENERATED_DIR}")

set(LLVM_TARGET_DEFINITIONS
    "${DEBUGGER_INCLUDE_DIR}/Debugger/IR/FlagTreeDebugOps.td")
mlir_tablegen("${DEBUGGER_GENERATED_REL_DIR}/Ops.h.inc" -gen-op-decls
              -I "${DEBUGGER_INCLUDE_DIR}" -I "${MLIR_INCLUDE_DIRS}")
mlir_tablegen("${DEBUGGER_GENERATED_REL_DIR}/Ops.cpp.inc" -gen-op-defs
              -I "${DEBUGGER_INCLUDE_DIR}" -I "${MLIR_INCLUDE_DIRS}")
set(LLVM_TARGET_DEFINITIONS
    "${DEBUGGER_INCLUDE_DIR}/Debugger/IR/FlagTreeDebugDialect.td")
mlir_tablegen("${DEBUGGER_GENERATED_REL_DIR}/Dialect.h.inc" -gen-dialect-decls
              -dialect=flagtree_debug -I "${DEBUGGER_INCLUDE_DIR}"
              -I "${MLIR_INCLUDE_DIRS}")
mlir_tablegen("${DEBUGGER_GENERATED_REL_DIR}/Dialect.cpp.inc" -gen-dialect-defs
              -dialect=flagtree_debug -I "${DEBUGGER_INCLUDE_DIR}"
              -I "${MLIR_INCLUDE_DIRS}")
add_public_tablegen_target(FlagPrismDebuggerComponentTableGen)

# Compiler bindings live in libtriton so they use the core's module-local
# MLIR/pybind types without exporting the libtriton C++ ABI.
set(DEBUGGER_COMPILER_IMPLEMENTATION_SOURCES
  "${DEBUGGER_ROOT}/lib/IR/Dialect.cpp"
  "${DEBUGGER_ROOT}/lib/IR/Ops.cpp"
  "${DEBUGGER_ROOT}/lib/Metadata/Passes.cpp"
  "${DEBUGGER_ROOT}/lib/Metadata/TrackedOpTable.cpp"
  "${DEBUGGER_ROOT}/lib/Instrumentation/Collectors.cpp"
  "${DEBUGGER_ROOT}/lib/Instrumentation/Passes.cpp"
  "${DEBUGGER_ROOT}/lib/Instrumentation/RecordBuilder.cpp"
  "${DEBUGGER_ROOT}/lib/Instrumentation/Writer.cpp"
)
add_triton_plugin(TritonDebugger
  "${DEBUGGER_ROOT}/python/CompilerPlugin.cpp"
  "${DEBUGGER_ROOT}/python/CompilerBindings.cpp"
  ${DEBUGGER_COMPILER_IMPLEMENTATION_SOURCES}

  DEPENDS
  FlagPrismDebuggerComponentTableGen

  LINK_LIBS
  MLIRIR
  MLIRPass
  LLVMSupport
)
target_link_libraries(TritonDebugger PRIVATE Python3::Module pybind11::headers)
flagprism_apply_backend_compile_definitions(TritonDebugger)
target_include_directories(TritonDebugger PRIVATE
  "${DEBUGGER_INCLUDE_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}/include"
  "${PYTHON_SRC_PATH}"
  "${FLAGTREE_SOURCE_DIR}/python/src"
  "${LLVM_INCLUDE_DIRS}"
  "${MLIR_INCLUDE_DIRS}"
)

# Runtime transfer and report bindings remain independently loadable, but no
# longer depend on libtriton or its MLIR pybind wrappers.
set(DEBUGGER_RUNTIME_IMPLEMENTATION_SOURCES
  "${DEBUGGER_ROOT}/lib/Metadata/TrackedOpTable.cpp"
  "${DEBUGGER_ROOT}/lib/Runtime/BufferLayout.cpp"
  "${DEBUGGER_ROOT}/lib/Runtime/TransferEngine.cpp"
  "${DEBUGGER_ROOT}/lib/Frontend/Bridge.cpp"
  "${DEBUGGER_ROOT}/lib/Decode/Decoder.cpp"
  "${DEBUGGER_ROOT}/lib/Decode/Reporter.cpp"
)
pybind11_add_module(flagtree_debugger_native MODULE
  "${DEBUGGER_ROOT}/python/NativeModule.cpp"
  "${DEBUGGER_ROOT}/python/Bindings.cpp"
  ${DEBUGGER_RUNTIME_IMPLEMENTATION_SOURCES}
)
target_include_directories(flagtree_debugger_native PRIVATE
  "${DEBUGGER_INCLUDE_DIR}"
  "${LLVM_INCLUDE_DIRS}"
)
target_link_libraries(flagtree_debugger_native PRIVATE
  LLVMSupport
  Python3::Module
)
flagprism_apply_backend_compile_definitions(flagtree_debugger_native)

set_source_files_properties(
  ${DEBUGGER_COMPILER_IMPLEMENTATION_SOURCES}
  ${DEBUGGER_RUNTIME_IMPLEMENTATION_SOURCES}
  PROPERTIES COMPILE_OPTIONS "-fno-rtti"
)

include("${DEBUGGER_ROOT}/lib/Runtime/CoreX.cmake")
if(NOT FLAGPRISM_BACKEND STREQUAL "tianshu")
  include("${DEBUGGER_ROOT}/lib/Runtime/CANN.cmake")
endif()
flagprism_enable_debugger_runtime(flagtree_debugger_native)

set(_flagtree_debugger_output_dir
    "${FLAGPRISM_PYTHON_DIR}/flagtree/debugger")
file(MAKE_DIRECTORY "${_flagtree_debugger_output_dir}")
set_target_properties(flagtree_debugger_native PROPERTIES
  PREFIX ""
  OUTPUT_NAME "_native"
  LIBRARY_OUTPUT_DIRECTORY "${_flagtree_debugger_output_dir}"
  RUNTIME_OUTPUT_DIRECTORY "${_flagtree_debugger_output_dir}"
)
if(PYTHON_EXTENSION_SUFFIX)
  set_target_properties(flagtree_debugger_native PROPERTIES
    SUFFIX "${PYTHON_EXTENSION_SUFFIX}")
endif()
