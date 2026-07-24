cmake_minimum_required(VERSION 3.20)
project(Proton LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(FLAGTREE_PROTON_STANDALONE OFF)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(FLAGTREE_PROTON_STANDALONE ON)
endif()

set(FLAGTREE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../.." CACHE PATH
    "FlagTree source tree used for compiler headers")
set(FLAGTREE_BUILD_DIR "" CACHE PATH
    "FlagTree build tree containing generated compiler headers")
set(FLAGTREE_LIBTRITON "" CACHE FILEPATH
    "Installed FlagTree libtriton Python extension")

set(PROTON_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/csrc")
set(PROTON_COMMON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/common")
set(FLAGTREE_PROFILER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/flagtree_profiler")
set(FLAGTREE_PROFILER_CSRC_DIR "${FLAGTREE_PROFILER_DIR}/csrc")

# ============ Check for includes =============
if(NOT CUPTI_INCLUDE_DIR)
  message(FATAL_ERROR "CUPTI include directory not defined")
endif()
if(NOT ROCTRACER_INCLUDE_DIR)
  message(FATAL_ERROR "ROCTRACER include directory not defined")
endif()
if(NOT JSON_INCLUDE_DIR)
  message(FATAL_ERROR "JSON include directory not defined")
endif()

# ============ Dependencies =============
find_package(Python3 REQUIRED Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED HINTS "${Python3_SITELIB}")

if(FLAGTREE_PROTON_STANDALONE)
  if(NOT FLAGTREE_BUILD_DIR)
    message(FATAL_ERROR "FLAGTREE_BUILD_DIR must point to the FlagTree core CMake build tree")
  endif()
  if(NOT FLAGTREE_LIBTRITON)
    message(FATAL_ERROR "FLAGTREE_LIBTRITON must point to the installed libtriton extension")
  endif()
  find_package(LLVM CONFIG REQUIRED)
  find_package(MLIR CONFIG REQUIRED)
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}" "${MLIR_CMAKE_DIR}")
  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  set(TRITON_BUILD_PYTHON_MODULE ON)
  add_compile_definitions(FLAGTREE_PROTON_COMPONENT_BUILD=1 __PROTON__=1)
  include_directories(
    "${CMAKE_CURRENT_BINARY_DIR}"
    "${FLAGTREE_SOURCE_DIR}"
    "${FLAGTREE_SOURCE_DIR}/include"
    "${FLAGTREE_SOURCE_DIR}/python/src"
    "${FLAGTREE_SOURCE_DIR}/third_party"
    "${FLAGTREE_BUILD_DIR}/include"
    "${FLAGTREE_BUILD_DIR}/third_party"
    "${LLVM_INCLUDE_DIRS}"
    "${MLIR_INCLUDE_DIRS}"
  )
endif()

# ============ Define a GLOBAL property to store object-libraries ============
set_property(GLOBAL PROPERTY PROTON_LIBS "")

# ============ Define a function to create object libraries ============
function(add_proton_library name)
  add_library(${name} OBJECT ${ARGN})

  target_link_libraries(${name} PRIVATE Python3::Module pybind11::headers)

  # Use system to skip warnings caused by legacy clang compilers
  target_include_directories(${name}
    SYSTEM PRIVATE
      "${ROCTRACER_INCLUDE_DIR}"
  )

  target_include_directories(${name}
    PRIVATE
      "${CUPTI_INCLUDE_DIR}"
      "${JSON_INCLUDE_DIR}"
      "${PROTON_COMMON_DIR}/include"
      "${PROTON_SRC_DIR}/include"
      "${FLAGTREE_PROFILER_CSRC_DIR}/include"
  )

  # If HIP is AMD-based
  target_compile_definitions(${name} PRIVATE __HIP_PLATFORM_AMD__)

  # Append this library name to the GLOBAL property "PROTON_LIBS"
  set_property(GLOBAL APPEND PROPERTY PROTON_LIBS ${name})
endfunction()

if(FLAGTREE_PROTON_STANDALONE)
  set_property(GLOBAL PROPERTY PROTON_COMPILER_LIBS "")

  function(add_triton_library name)
    cmake_parse_arguments(ARG "" "" "DEPENDS;LINK_LIBS" ${ARGN})
    add_library(${name} OBJECT ${ARG_UNPARSED_ARGUMENTS})
    if(ARG_DEPENDS)
      add_dependencies(${name} ${ARG_DEPENDS})
    endif()
    llvm_update_compile_flags(${name})
    set_property(GLOBAL APPEND PROPERTY PROTON_COMPILER_LIBS ${name})
  endfunction()

  function(add_triton_plugin name)
    cmake_parse_arguments(ARG "" "SHARED_LIB" "LINK_LIBS" ${ARGN})
    if(ARG_SHARED_LIB)
      message(FATAL_ERROR "standalone Proton does not support SHARED_LIB plugins")
    endif()
    add_library(${name} OBJECT ${ARG_UNPARSED_ARGUMENTS})
    set_property(GLOBAL APPEND PROPERTY PROTON_COMPILER_LIBS ${name})
  endfunction()
endif()

# ============ Add subdirectory with actual code that calls add_proton_library ============
add_subdirectory("${PROTON_COMMON_DIR}")
add_subdirectory("${PROTON_SRC_DIR}")
add_subdirectory("${FLAGTREE_PROFILER_CSRC_DIR}")
if(FLAGTREE_PROTON_STANDALONE)
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/Dialect"
                   "${CMAKE_CURRENT_BINARY_DIR}/proton/Dialect")
endif()

# ============ Add subdirectory with proton tests ============
if(TRITON_BUILD_UT)
  add_subdirectory(test)
endif()

# ============ Possibly handle macOS specifics ============
if(APPLE)
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
  # Other platforms build with -flto, but we found that this adds significant overhead to our macos CI without providing a major benefit.
  set(PROTON_PYTHON_LDFLAGS "-undefined dynamic_lookup")
endif()

# ============ Collect all object libraries from property and build final shared lib ============
get_property(_proton_obj_libs GLOBAL PROPERTY PROTON_LIBS)
if(FLAGTREE_PROTON_STANDALONE)
  get_property(_proton_compiler_obj_libs GLOBAL PROPERTY PROTON_COMPILER_LIBS)
  list(APPEND _proton_obj_libs ${_proton_compiler_obj_libs})
endif()

if(NOT _proton_obj_libs)
  message(WARNING "No object libraries were defined in 'PROTON_LIBS'!")
endif()

set(_proton_obj_sources "")
foreach(_lib IN LISTS _proton_obj_libs)
  list(APPEND _proton_obj_sources $<TARGET_OBJECTS:${_lib}>)
  message(STATUS "Collecting object files from ${_lib}")
endforeach()

add_library(proton SHARED ${_proton_obj_sources})

set_target_properties(proton PROPERTIES
  PREFIX ""
  OUTPUT_NAME "libproton"
)
if(PYTHON_EXTENSION_SUFFIX)
  set_target_properties(proton PROPERTIES SUFFIX "${PYTHON_EXTENSION_SUFFIX}")
endif()

target_link_libraries(proton PRIVATE Python3::Module)
if(FLAGTREE_PROTON_STANDALONE)
  target_link_libraries(proton PRIVATE
    "${FLAGTREE_LIBTRITON}"
    MLIRIR
    MLIRPass
    LLVMSupport
  )
endif()
if(CMAKE_DL_LIBS)
  target_link_libraries(proton PRIVATE ${CMAKE_DL_LIBS})
endif()
# Apply any macOS linker flags or extra link options
if(PROTON_PYTHON_LDFLAGS)
  target_link_options(proton PRIVATE ${PROTON_PYTHON_LDFLAGS})
endif()
if(FLAGTREE_PROTON_STANDALONE)
  set_target_properties(proton PROPERTIES
    BUILD_RPATH "$ORIGIN/../triton/_C"
    INSTALL_RPATH "$ORIGIN/../triton/_C"
  )
endif()
