include_guard(GLOBAL)

# CoreX exposes a CUDA-compatible driver surface. The runtime implementation
# resolves that surface lazily so a FlagPrism wheel remains usable on hosts
# without the Tianshu SDK installed.
function(flagtree_debugger_enable_corex target)
  target_compile_definitions(${target}
    PRIVATE
      FLAGTREE_DEBUGGER_HAS_COREX_RUNTIME=1
  )
  if(CMAKE_DL_LIBS)
    target_link_libraries(${target} PRIVATE ${CMAKE_DL_LIBS})
  endif()
endfunction()
