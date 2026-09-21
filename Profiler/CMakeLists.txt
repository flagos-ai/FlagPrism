cmake_minimum_required(VERSION 3.20)
project(FlagTreeProfiler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(FLAGTREE_PROFILER_STANDALONE OFF)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(FLAGTREE_PROFILER_STANDALONE ON)
endif()

if(FLAGTREE_PROFILER_STANDALONE)
  set(FLAGTREE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../.." CACHE PATH
      "FlagTree source tree used for compiler headers")
else()
  # Do not let a reused build tree retain a source path from the former
  # component directory name.
  if(FLAGPRISM_FLAGTREE_SOURCE_DIR)
    set(FLAGTREE_SOURCE_DIR "${FLAGPRISM_FLAGTREE_SOURCE_DIR}")
  else()
    set(FLAGTREE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
  endif()
endif()
set(FLAGTREE_BUILD_DIR "" CACHE PATH
    "FlagTree build tree containing generated compiler headers")
set(FLAGTREE_LIBTRITON "" CACHE FILEPATH
    "Installed FlagTree libtriton Python extension")

set(PROFILER_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/csrc")
set(PROFILER_COMMON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/common")

# MThreads' vendor profiler loads the MUPTI activity ABI dynamically, so its
# SDK headers are optional and libmupti is not a hard link dependency.
set(FLAGTREE_MTHREADS_MUPTI_INCLUDE_DIR "" CACHE PATH
    "MThreads MUPTI include directory")
if(FLAGPRISM_BACKEND STREQUAL "mthreads" OR
   FLAGPRISM_BACKEND STREQUAL "all")
  if(NOT FLAGTREE_MTHREADS_MUPTI_INCLUDE_DIR)
    # An empty cached value makes find_path() skip its search. Remove only the
    # empty entry so automatic SDK discovery still runs on the first configure.
    unset(FLAGTREE_MTHREADS_MUPTI_INCLUDE_DIR CACHE)
    find_path(FLAGTREE_MTHREADS_MUPTI_INCLUDE_DIR
      NAMES mupti_activity.h
      HINTS
        "$ENV{MUSA_HOME}/include"
        "/usr/local/musa/include"
    )
  endif()
endif()

# FlagTree's Ascend/Tianshu/MThreads paths use vendor APIs instead of
# compiling against CUDA/CUPTI/ROCtracer headers. Keep the switch explicit so
# vendor builds can validate the component without CUDA SDK packages.
option(FLAGTREE_PROFILER_GPU_RUNTIME
       "Build the CUDA/HIP profiler runtime adapters" ON)
if(FLAGTREE_BACKEND MATCHES "^(ascend|mthreads|tianshu|enflame)$" OR
   FLAGPRISM_BACKEND MATCHES "^(ascend|mthreads|tianshu|enflame)$")
  set(FLAGTREE_PROFILER_GPU_RUNTIME OFF)
endif()

# FlagPrism: keep the historical aggregate switch for compatibility, but
# split the SDK requirements so an NVIDIA-only build does not need ROCm.
set(FLAGTREE_PROFILER_CUDA_RUNTIME OFF)
set(FLAGTREE_PROFILER_ROCTRACER_RUNTIME OFF)
if(FLAGTREE_PROFILER_GPU_RUNTIME)
  if(FLAGPRISM_BACKEND STREQUAL "nvidia")
    set(FLAGTREE_PROFILER_CUDA_RUNTIME ON)
  else()
    set(FLAGTREE_PROFILER_CUDA_RUNTIME ON)
    set(FLAGTREE_PROFILER_ROCTRACER_RUNTIME ON)
  endif()
endif()

# ============ Check for includes =============
if(FLAGTREE_PROFILER_CUDA_RUNTIME)
  if(NOT CUPTI_INCLUDE_DIR)
    message(FATAL_ERROR "CUPTI include directory not defined")
  endif()
endif()
if(FLAGTREE_PROFILER_ROCTRACER_RUNTIME)
  if(NOT ROCTRACER_INCLUDE_DIR)
    message(FATAL_ERROR "ROCTRACER include directory not defined")
  endif()
endif()
if(NOT JSON_INCLUDE_DIR)
  message(FATAL_ERROR "JSON include directory not defined")
endif()

if(FLAGPRISM_BACKEND STREQUAL "enflame")
  find_path(TOPSPTI_INCLUDE_DIR topspti_activity.h HINTS /opt/tops/extras/TOPSPTI/include REQUIRED)
  find_path(TOPS_INCLUDE_DIR tops/tops_runtime.h HINTS /opt/tops/include REQUIRED)
  find_library(TOPSPTI_ACTIVITY_LIBRARY topspti HINTS /opt/tops/extras/TOPSPTI/lib64 REQUIRED)
  find_library(TOPS_LIBRARY topsrt HINTS /opt/tops/lib REQUIRED)
  include_directories(${TOPSPTI_INCLUDE_DIR} ${TOPS_INCLUDE_DIR})
endif()

# ============ Dependencies =============
find_package(Python3 REQUIRED Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED HINTS "${Python3_SITELIB}")

if(FLAGTREE_PROFILER_STANDALONE)
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
  add_compile_definitions(FLAGTREE_PROFILER_COMPONENT_BUILD=1 __PROTON__=1)
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
set_property(GLOBAL PROPERTY FLAGTREE_PROFILER_LIBS "")

# ============ Define a function to create object libraries ============
function(add_profiler_library name)
  add_library(${name} OBJECT ${ARGN})

  target_link_libraries(${name} PRIVATE Python3::Module pybind11::headers)

  if(FLAGTREE_PROFILER_ROCTRACER_RUNTIME)
    # Use system to skip warnings caused by legacy clang compilers.
    target_include_directories(${name} SYSTEM PRIVATE
      "${ROCTRACER_INCLUDE_DIR}")
  endif()

  target_include_directories(${name}
    PRIVATE
      "${JSON_INCLUDE_DIR}"
      "${PROFILER_COMMON_DIR}/include"
      "${PROFILER_SRC_DIR}/include"
  )

  if(FLAGTREE_PROFILER_CUDA_RUNTIME)
    target_include_directories(${name} PRIVATE "${CUPTI_INCLUDE_DIR}")
  endif()

  target_compile_definitions(${name} PRIVATE
    FLAGTREE_PROFILER_GPU_RUNTIME=$<BOOL:${FLAGTREE_PROFILER_GPU_RUNTIME}>
    FLAGTREE_PROFILER_CUDA_RUNTIME=$<BOOL:${FLAGTREE_PROFILER_CUDA_RUNTIME}>
    FLAGTREE_PROFILER_ROCTRACER_RUNTIME=$<BOOL:${FLAGTREE_PROFILER_ROCTRACER_RUNTIME}>)
  flagprism_apply_backend_compile_definitions(${name})
  if(FLAGTREE_MTHREADS_MUPTI_INCLUDE_DIR AND
     (FLAGPRISM_BACKEND STREQUAL "mthreads" OR
      FLAGPRISM_BACKEND STREQUAL "all"))
    target_include_directories(${name} SYSTEM PRIVATE
      "${FLAGTREE_MTHREADS_MUPTI_INCLUDE_DIR}")
    target_compile_definitions(${name} PRIVATE
      FLAGTREE_MTHREADS_HAS_MUPTI=1)
  endif()
  if(FLAGTREE_PROFILER_ROCTRACER_RUNTIME)
    # If HIP is AMD-based.
    target_compile_definitions(${name} PRIVATE __HIP_PLATFORM_AMD__)
  endif()

  set_property(GLOBAL APPEND PROPERTY FLAGTREE_PROFILER_LIBS ${name})
endfunction()

if(FLAGTREE_PROFILER_STANDALONE)
  set_property(GLOBAL PROPERTY FLAGTREE_PROFILER_COMPILER_LIBS "")

  function(add_triton_library name)
    cmake_parse_arguments(ARG "" "" "DEPENDS;LINK_LIBS" ${ARGN})
    add_library(${name} OBJECT ${ARG_UNPARSED_ARGUMENTS})
    if(ARG_DEPENDS)
      add_dependencies(${name} ${ARG_DEPENDS})
    endif()
    llvm_update_compile_flags(${name})
    set_property(GLOBAL APPEND PROPERTY FLAGTREE_PROFILER_COMPILER_LIBS ${name})
  endfunction()

  function(add_triton_plugin name)
    cmake_parse_arguments(ARG "" "SHARED_LIB" "LINK_LIBS" ${ARGN})
    if(ARG_SHARED_LIB)
      message(FATAL_ERROR "standalone Profiler does not support SHARED_LIB plugins")
    endif()
    add_library(${name} OBJECT ${ARG_UNPARSED_ARGUMENTS})
    set_property(GLOBAL APPEND PROPERTY FLAGTREE_PROFILER_COMPILER_LIBS ${name})
  endfunction()
endif()

# ============ Add Profiler runtime and internal compiler dialect ============
add_subdirectory("${PROFILER_COMMON_DIR}")
add_subdirectory("${PROFILER_SRC_DIR}")
if(FLAGTREE_PROFILER_STANDALONE)
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/Dialect"
                   "${CMAKE_CURRENT_BINARY_DIR}/Profiler/Dialect")
endif()

# ============ Add Profiler tests ============
if(TRITON_BUILD_UT)
  add_subdirectory(test)
endif()

# ============ Possibly handle macOS specifics ============
if(APPLE)
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
  # Other platforms build with -flto, but we found that this adds significant overhead to our macos CI without providing a major benefit.
  set(PROFILER_PYTHON_LDFLAGS "-undefined dynamic_lookup")
endif()

# ============ Collect all object libraries from property and build final shared lib ============
get_property(_profiler_obj_libs GLOBAL PROPERTY FLAGTREE_PROFILER_LIBS)
if(FLAGTREE_PROFILER_STANDALONE)
  get_property(_profiler_compiler_obj_libs GLOBAL PROPERTY FLAGTREE_PROFILER_COMPILER_LIBS)
  list(APPEND _profiler_obj_libs ${_profiler_compiler_obj_libs})
endif()

if(NOT _profiler_obj_libs)
  message(WARNING "No object libraries were defined for FlagTree Profiler")
endif()

set(_profiler_obj_sources "")
foreach(_lib IN LISTS _profiler_obj_libs)
  list(APPEND _profiler_obj_sources $<TARGET_OBJECTS:${_lib}>)
  message(STATUS "Collecting object files from ${_lib}")
endforeach()

add_library(flagtree_profiler_native SHARED ${_profiler_obj_sources})

set_target_properties(flagtree_profiler_native PROPERTIES
  PREFIX ""
  OUTPUT_NAME "_native"
)
if(PYTHON_EXTENSION_SUFFIX)
  set_target_properties(flagtree_profiler_native PROPERTIES SUFFIX "${PYTHON_EXTENSION_SUFFIX}")
endif()

if(FLAGPRISM_PYTHON_DIR)
  set(_profiler_output_dir "${FLAGPRISM_PYTHON_DIR}/flagtree/profiler")
  file(MAKE_DIRECTORY "${_profiler_output_dir}")
  set_target_properties(flagtree_profiler_native PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${_profiler_output_dir}"
    RUNTIME_OUTPUT_DIRECTORY "${_profiler_output_dir}")
  foreach(_config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set_target_properties(flagtree_profiler_native PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY_${_config} "${_profiler_output_dir}")
  endforeach()
endif()

target_link_libraries(flagtree_profiler_native PRIVATE Python3::Module)
if(FLAGTREE_PROFILER_STANDALONE)
  target_link_libraries(flagtree_profiler_native PRIVATE
    "${FLAGTREE_LIBTRITON}"
    MLIRIR
    MLIRPass
    LLVMSupport
  )
endif()
if(CMAKE_DL_LIBS)
  target_link_libraries(flagtree_profiler_native PRIVATE ${CMAKE_DL_LIBS})
endif()
# Apply any macOS linker flags or extra link options
if(PROFILER_PYTHON_LDFLAGS)
  target_link_options(flagtree_profiler_native PRIVATE ${PROFILER_PYTHON_LDFLAGS})
endif()
if(FLAGTREE_PROFILER_STANDALONE)
  set_target_properties(flagtree_profiler_native PROPERTIES
    BUILD_RPATH "$ORIGIN/../../triton/_C"
    INSTALL_RPATH "$ORIGIN/../../triton/_C"
  )
endif()

if(FLAGPRISM_BACKEND STREQUAL "enflame")
  target_link_libraries(flagtree_profiler_native PRIVATE ${TOPSPTI_ACTIVITY_LIBRARY} ${TOPS_LIBRARY})
endif()
