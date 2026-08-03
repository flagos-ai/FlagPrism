include_guard(GLOBAL)

# FlagPrism: keep all build policy for the bundled Debugger and
# Proton at this boundary. Call sites in the main CMake file should only add
# components at the compiler lifecycle point where their targets are needed.
set(FLAGPRISM_SOURCE_DIR
    "${PROJECT_SOURCE_DIR}/third_party/FlagPrism")

set(_flagprism_legacy_value)
set(_flagprism_legacy_names)
foreach(_option IN ITEMS
    TRITON_BUILD_DEVTOOLS TRITON_BUILD_PROTON)
  if(DEFINED ${_option})
    if(${_option})
      set(_value ON)
    else()
      set(_value OFF)
    endif()
    if(DEFINED _flagprism_legacy_value AND
       NOT "${_flagprism_legacy_value}" STREQUAL "${_value}")
      message(FATAL_ERROR
        "FlagPrism components cannot be enabled independently. Set "
        "TRITON_BUILD_FLAGPRISM=ON for the combined tools build or OFF "
        "for a core-only build.")
    endif()
    set(_flagprism_legacy_value ${_value})
    list(APPEND _flagprism_legacy_names ${_option})
  endif()
endforeach()

if(DEFINED TRITON_BUILD_FLAGPRISM)
  if(TRITON_BUILD_FLAGPRISM)
    set(_flagprism_enabled ON)
  else()
    set(_flagprism_enabled OFF)
  endif()
elseif(DEFINED _flagprism_legacy_value)
  set(_flagprism_enabled ${_flagprism_legacy_value})
  message(DEPRECATION
    "${_flagprism_legacy_names} is a compatibility input; use "
    "TRITON_BUILD_FLAGPRISM instead.")
elseif(FLAGTREE_BACKEND)
  set(_flagprism_enabled OFF)
else()
  set(_flagprism_enabled ON)
endif()

set(TRITON_BUILD_FLAGPRISM ${_flagprism_enabled} CACHE BOOL
    "Build the bundled FlagPrism debugger and profiler" FORCE)
# Existing Proton and out-of-tree backend checks consume this internal alias.
set(TRITON_BUILD_PROTON ${TRITON_BUILD_FLAGPRISM} CACHE BOOL
    "Internal FlagPrism compatibility alias" FORCE)

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
      "${FLAGPRISM_SOURCE_DIR}/proton/CMakeLists.txt"
      "${FLAGPRISM_SOURCE_DIR}/proton/Dialect/include/Integration/Registration.h"
      "${FLAGPRISM_SOURCE_DIR}/proton/Dialect/lib/Integration/Registration.cpp"
      "${FLAGPRISM_SOURCE_DIR}/proton/Dialect/test/CMakeLists.txt"
      "${FLAGPRISM_SOURCE_DIR}/proton/Dialect/test/TestScopeIdAllocation.cpp"
      "${FLAGPRISM_SOURCE_DIR}/proton/proton/__init__.py")
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
  add_compile_definitions(__PROTON__=1)
elseif(NOT TARGET TritonTestProton)
  # Preserve the 3.5.x tool link contract in profiler-free builds.
  add_library(TritonTestProton INTERFACE)
endif()

macro(flagprism_add_proton_components)
  if(TRITON_BUILD_FLAGPRISM)
    if(TRITON_BUILD_PYTHON_MODULE)
      add_subdirectory("${FLAGPRISM_SOURCE_DIR}/proton"
                       "third_party/FlagPrism/proton")
      list(APPEND TRITON_PLUGIN_NAMES "proton")
    endif()
    add_subdirectory("${FLAGPRISM_SOURCE_DIR}/proton/Dialect"
                     "third_party/FlagPrism/proton/Dialect")
  endif()
endmacro()

macro(flagprism_add_python_components)
  flagprism_add_proton_components()
  if(TRITON_BUILD_FLAGPRISM)
    # The Debugger compiler binding follows the existing libtriton plugin
    # model; its runtime binding remains a separate wheel-local extension.
    list(APPEND TRITON_PLUGIN_NAMES "debugger")
    add_subdirectory("${FLAGPRISM_SOURCE_DIR}/Debugger/native"
                     "third_party/FlagPrism/Debugger/native")
  endif()
endmacro()

# Present one integration entry point to FlagTree while retaining the smaller
# Proton-only target set required by non-Python tools such as triton-opt.
macro(flagprism_add_components)
  if(TRITON_BUILD_PYTHON_MODULE)
    flagprism_add_python_components()
  else()
    flagprism_add_proton_components()
  endif()
endmacro()
