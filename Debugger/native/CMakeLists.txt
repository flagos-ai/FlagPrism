cmake_minimum_required(VERSION 3.20)
project(FlagTreeDebuggerPython LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

set(FLAGTREE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." CACHE PATH
    "FlagTree source tree used for compiler headers")
set(FLAGTREE_BUILD_DIR "" CACHE PATH
    "Optional FlagTree build tree containing generated headers")
set(FLAGTREE_LIBTRITON "" CACHE FILEPATH
    "Installed FlagTree libtriton Python extension")

if(NOT FLAGTREE_LIBTRITON)
  message(FATAL_ERROR "FLAGTREE_LIBTRITON must point to the installed libtriton extension")
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(LLVM CONFIG REQUIRED)
find_package(MLIR CONFIG REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}" "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

set(DEBUGGER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(DEBUGGER_INCLUDE_DIR "${DEBUGGER_ROOT}/include")
set(DEBUGGER_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/include/Debugger/IR")
file(MAKE_DIRECTORY "${DEBUGGER_GENERATED_DIR}")

set(LLVM_TARGET_DEFINITIONS
    "${DEBUGGER_INCLUDE_DIR}/Debugger/IR/FlagTreeDebugOps.td")
mlir_tablegen("${DEBUGGER_GENERATED_DIR}/Ops.h.inc" -gen-op-decls
              -I "${DEBUGGER_INCLUDE_DIR}" -I "${MLIR_INCLUDE_DIRS}")
mlir_tablegen("${DEBUGGER_GENERATED_DIR}/Ops.cpp.inc" -gen-op-defs
              -I "${DEBUGGER_INCLUDE_DIR}" -I "${MLIR_INCLUDE_DIRS}")
set(LLVM_TARGET_DEFINITIONS
    "${DEBUGGER_INCLUDE_DIR}/Debugger/IR/FlagTreeDebugDialect.td")
mlir_tablegen("${DEBUGGER_GENERATED_DIR}/Dialect.h.inc" -gen-dialect-decls
              -dialect=flagtree_debug -I "${DEBUGGER_INCLUDE_DIR}"
              -I "${MLIR_INCLUDE_DIRS}")
mlir_tablegen("${DEBUGGER_GENERATED_DIR}/Dialect.cpp.inc" -gen-dialect-defs
              -dialect=flagtree_debug -I "${DEBUGGER_INCLUDE_DIR}"
              -I "${MLIR_INCLUDE_DIRS}")
add_public_tablegen_target(FlagTreeDebuggerComponentTableGen)

set(DEBUGGER_BINDING_SOURCES
  "${DEBUGGER_ROOT}/python/NativeModule.cpp"
  "${DEBUGGER_ROOT}/python/CompilerBindings.cpp"
  "${DEBUGGER_ROOT}/python/Bindings.cpp"
)
set(DEBUGGER_IMPLEMENTATION_SOURCES
  "${DEBUGGER_ROOT}/lib/IR/Dialect.cpp"
  "${DEBUGGER_ROOT}/lib/IR/Ops.cpp"
  "${DEBUGGER_ROOT}/lib/Metadata/Passes.cpp"
  "${DEBUGGER_ROOT}/lib/Metadata/TrackedOpTable.cpp"
  "${DEBUGGER_ROOT}/lib/Instrumentation/Collectors.cpp"
  "${DEBUGGER_ROOT}/lib/Instrumentation/Passes.cpp"
  "${DEBUGGER_ROOT}/lib/Instrumentation/RecordBuilder.cpp"
  "${DEBUGGER_ROOT}/lib/Instrumentation/Writer.cpp"
  "${DEBUGGER_ROOT}/lib/Runtime/BufferLayout.cpp"
  "${DEBUGGER_ROOT}/lib/Runtime/TransferEngine.cpp"
  "${DEBUGGER_ROOT}/lib/Frontend/Bridge.cpp"
  "${DEBUGGER_ROOT}/lib/Decode/Decoder.cpp"
  "${DEBUGGER_ROOT}/lib/Decode/Reporter.cpp"
)
pybind11_add_module(flagtree_debugger_native MODULE
  ${DEBUGGER_BINDING_SOURCES}
  ${DEBUGGER_IMPLEMENTATION_SOURCES}
)
add_dependencies(flagtree_debugger_native FlagTreeDebuggerComponentTableGen)
set_source_files_properties(${DEBUGGER_IMPLEMENTATION_SOURCES}
  PROPERTIES COMPILE_OPTIONS "-fno-rtti")

target_include_directories(flagtree_debugger_native PRIVATE
  "${DEBUGGER_INCLUDE_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}/include"
  "${FLAGTREE_SOURCE_DIR}/include"
  "${FLAGTREE_SOURCE_DIR}/python/src"
  "${FLAGTREE_SOURCE_DIR}/third_party"
  "${LLVM_INCLUDE_DIRS}"
  "${MLIR_INCLUDE_DIRS}"
)
if(FLAGTREE_BUILD_DIR)
  target_include_directories(flagtree_debugger_native PRIVATE
    "${FLAGTREE_BUILD_DIR}/include"
    "${FLAGTREE_BUILD_DIR}/third_party"
  )
endif()

target_link_libraries(flagtree_debugger_native PRIVATE
  "${FLAGTREE_LIBTRITON}"
  MLIRIR
  MLIRPass
  LLVMSupport
  Python3::Module
)

include("${DEBUGGER_ROOT}/lib/Runtime/CANN.cmake")
flagtree_debugger_enable_cann(flagtree_debugger_native)

set_target_properties(flagtree_debugger_native PROPERTIES
  PREFIX ""
  OUTPUT_NAME "_native"
  BUILD_RPATH "$ORIGIN/../triton/_C"
  INSTALL_RPATH "$ORIGIN/../triton/_C"
)
if(PYTHON_EXTENSION_SUFFIX)
  set_target_properties(flagtree_debugger_native PROPERTIES
    SUFFIX "${PYTHON_EXTENSION_SUFFIX}")
endif()
