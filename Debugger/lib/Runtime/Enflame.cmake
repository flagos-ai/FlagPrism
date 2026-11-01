include_guard(GLOBAL)

# TOPS memory-transfer dependencies shared by bundled and standalone builds.
# tops_runtime.h is provided via a symlink under Debugger/lib/Runtime/Enflame
# that points at Profiler/csrc/include/Driver/Enflame. libtopsrt is loaded with
# dlopen, so no TopsPlatform install is needed at build time.
function(flagtree_debugger_enable_enflame target)
  set(FLAGTREE_TOPS_INCLUDE_DIR "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/Enflame")
  target_compile_definitions(${target} PRIVATE FLAGTREE_DEBUGGER_HAS_ENFLAME_RUNTIME=1)
  target_include_directories(${target} PRIVATE "${FLAGTREE_TOPS_INCLUDE_DIR}")
  if(CMAKE_DL_LIBS)
    target_link_libraries(${target} PRIVATE ${CMAKE_DL_LIBS})
  endif()
endfunction()
