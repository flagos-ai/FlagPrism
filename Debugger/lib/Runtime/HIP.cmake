include_guard(GLOBAL)
function(flagtree_debugger_enable_hip target)
  find_path(FLAGPRISM_HIP_INCLUDE_DIR hip/hip_runtime_api.h
    HINTS "$ENV{DTKROOT}/include" "$ENV{HIP_PATH}/include" /opt/dtk/include /opt/rocm/include REQUIRED)
  find_library(FLAGPRISM_HIP_RUNTIME amdhip64
    HINTS "$ENV{DTKROOT}/lib" "$ENV{HIP_PATH}/lib" /opt/dtk/lib /opt/rocm/lib REQUIRED)
  target_compile_definitions(${target} PRIVATE FLAGTREE_DEBUGGER_HAS_HIP_RUNTIME=1 __HIP_PLATFORM_AMD__)
  target_include_directories(${target} PRIVATE "${FLAGPRISM_HIP_INCLUDE_DIR}")
  target_link_libraries(${target} PRIVATE "${FLAGPRISM_HIP_RUNTIME}")
endfunction()
