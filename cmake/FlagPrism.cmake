include_guard(GLOBAL)

# FlagPrism: keep all build policy for the bundled Debugger and
# Profiler at this boundary. Call sites in the main CMake file should only add
# components at the compiler lifecycle point where their targets are needed.
if(NOT FLAGPRISM_SOURCE_DIR)
  set(FLAGPRISM_SOURCE_DIR
      "${PROJECT_SOURCE_DIR}/third_party/FlagPrism")
endif()
get_filename_component(FLAGPRISM_SOURCE_DIR "${FLAGPRISM_SOURCE_DIR}" ABSOLUTE)
set(FLAGPRISM_FLAGTREE_SOURCE_DIR "${PROJECT_SOURCE_DIR}")

if(FLAGTREE_BACKEND)
  set(_flagprism_default OFF)
else()
  set(_flagprism_default ON)
endif()

option(TRITON_BUILD_FLAGPRISM
       "Build the bundled FlagPrism debugger and profiler"
       ${_flagprism_default})

set(_flagprism_supported_backends all ascend tianshu)
set(_flagprism_backend_default "all")
if(FLAGTREE_BACKEND STREQUAL "ascend")
  set(_flagprism_backend_default "ascend")
elseif(FLAGTREE_BACKEND STREQUAL "tianshu" OR
       FLAGTREE_BACKEND STREQUAL "iluvatar" OR
       FLAGTREE_BACKEND STREQUAL "corex")
  set(_flagprism_backend_default "tianshu")
endif()
if(NOT FLAGPRISM_BACKEND)
  set(FLAGPRISM_BACKEND "${_flagprism_backend_default}" CACHE STRING
      "FlagPrism vendor backend to compile (all, ascend, or tianshu)" FORCE)
endif()
string(TOLOWER "${FLAGPRISM_BACKEND}" FLAGPRISM_BACKEND)
set_property(CACHE FLAGPRISM_BACKEND PROPERTY STRINGS all ascend tianshu)
if(NOT FLAGPRISM_BACKEND IN_LIST _flagprism_supported_backends)
  message(FATAL_ERROR
    "Unsupported FLAGPRISM_BACKEND='${FLAGPRISM_BACKEND}'. "
    "Choose all, ascend, or tianshu.")
endif()
set(FLAGPRISM_BUILD_VENDOR_LOWERING OFF)
if(FLAGPRISM_BACKEND STREQUAL "all")
  set(FLAGPRISM_BUILD_VENDOR_LOWERING ON)
endif()
message(STATUS "FlagPrism vendor backend: ${FLAGPRISM_BACKEND}")

function(flagprism_apply_backend_compile_definitions target)
  if(FLAGPRISM_BACKEND STREQUAL "tianshu")
    target_compile_definitions(${target}
      PRIVATE FLAGPRISM_BACKEND_TIANSHU=1)
  elseif(FLAGPRISM_BACKEND STREQUAL "ascend")
    target_compile_definitions(${target}
      PRIVATE FLAGPRISM_BACKEND_ASCEND=1)
  endif()
endfunction()

function(flagprism_enable_debugger_runtime target)
  if(FLAGPRISM_BACKEND STREQUAL "tianshu")
    flagtree_debugger_enable_corex(${target})
  elseif(FLAGPRISM_BACKEND STREQUAL "ascend")
    flagtree_debugger_enable_cann(${target})
  else()
    flagtree_debugger_enable_cann(${target})
    flagtree_debugger_enable_corex(${target})
  endif()
endfunction()

function(flagprism_validate_sources)
  set(_required_sources)
  if(TRITON_BUILD_FLAGPRISM)
    list(APPEND _required_sources
      "${FLAGPRISM_SOURCE_DIR}/Debugger/native/CMakeLists.txt"
      "${FLAGPRISM_SOURCE_DIR}/Debugger/python/CompilerBindings.cpp"
      "${FLAGPRISM_SOURCE_DIR}/Debugger/python/CompilerPlugin.cpp"
      "${FLAGPRISM_SOURCE_DIR}/Debugger/python/flagtree_debugger/__init__.py"
      "${FLAGPRISM_SOURCE_DIR}/Debugger/python/flagtree_debugger/language.py"
      "${FLAGPRISM_SOURCE_DIR}/Debugger/python/flagtree_debugger/statement.py"
      "${FLAGPRISM_SOURCE_DIR}/Profiler/CMakeLists.txt"
      "${FLAGPRISM_SOURCE_DIR}/Profiler/Dialect/include/Integration/Registration.h"
      "${FLAGPRISM_SOURCE_DIR}/Profiler/Dialect/lib/Integration/Registration.cpp"
      "${FLAGPRISM_SOURCE_DIR}/Profiler/Dialect/test/CMakeLists.txt"
      "${FLAGPRISM_SOURCE_DIR}/Profiler/Dialect/test/TestScopeIdAllocation.cpp"
      "${FLAGPRISM_SOURCE_DIR}/Profiler/python/flagtree_profiler/__init__.py")
  endif()
  foreach(_source IN LISTS _required_sources)
    if(NOT EXISTS "${_source}")
      message(FATAL_ERROR
        "FlagPrism source is missing: ${_source}. "
        "Run `git submodule update --init --recursive`.")
    endif()
  endforeach()
endfunction()

if(TRITON_BUILD_FLAGPRISM)
  flagprism_validate_sources()
endif()

if(TRITON_BUILD_FLAGPRISM)
  # Proton is the stable name of the Profiler's internal compiler dialect.
  add_compile_definitions(__PROTON__=1)
elseif(NOT TARGET TritonTestProton)
  # Preserve the 3.5.x tool link contract in profiler-free builds.
  add_library(TritonTestProton INTERFACE)
endif()

macro(flagprism_add_profiler_components)
  if(TRITON_BUILD_FLAGPRISM)
    if(TRITON_BUILD_PYTHON_MODULE)
      add_subdirectory("${FLAGPRISM_SOURCE_DIR}/Profiler"
                       "third_party/FlagPrism/Profiler")
      # Keep the compiler plugin name because it exposes the internal Proton
      # dialect through triton._C.libtriton.proton.
      list(APPEND TRITON_PLUGIN_NAMES "proton")
    endif()
    add_subdirectory("${FLAGPRISM_SOURCE_DIR}/Profiler/Dialect"
                     "third_party/FlagPrism/Profiler/Dialect")
  endif()
endmacro()

macro(flagprism_add_python_components)
  flagprism_add_profiler_components()
  if(TRITON_BUILD_FLAGPRISM)
    # The Debugger compiler binding follows the existing libtriton plugin
    # model; its runtime binding remains a separate wheel-local extension.
    list(APPEND TRITON_PLUGIN_NAMES "debugger")
    add_subdirectory("${FLAGPRISM_SOURCE_DIR}/Debugger/native"
                     "third_party/FlagPrism/Debugger/native")
  endif()
endmacro()

# Present one integration entry point to FlagTree while retaining the Profiler
# compiler target set required by non-Python tools such as triton-opt.
macro(flagprism_add_components)
  if(TRITON_BUILD_PYTHON_MODULE)
    flagprism_add_python_components()
  else()
    flagprism_add_profiler_components()
  endif()
endmacro()
