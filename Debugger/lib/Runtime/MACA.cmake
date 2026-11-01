# FlagPrism：discover the MACA runtime without requiring CUDA compatibility libraries.
include_guard(GLOBAL)
function(flagtree_debugger_enable_maca target)
  find_path(FLAGPRISM_MACA_INCLUDE_DIR mcr/mc_runtime_api.h
    HINTS "$ENV{MACA_PATH}/include" "/opt/maca/include" REQUIRED)
  find_library(FLAGPRISM_MACA_RUNTIME mcruntime
    HINTS "$ENV{MACA_PATH}/lib" "/opt/maca/lib" REQUIRED)
  target_compile_definitions(${target} PRIVATE FLAGTREE_DEBUGGER_HAS_MACA_RUNTIME=1)
  target_include_directories(${target} PRIVATE "${FLAGPRISM_MACA_INCLUDE_DIR}")
  target_link_libraries(${target} PRIVATE "${FLAGPRISM_MACA_RUNTIME}")
endfunction()
# FlagPrism end
