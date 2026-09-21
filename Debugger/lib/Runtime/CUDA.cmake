include_guard(GLOBAL)

# FlagPrism: CUDA debugger transfers use the dynamically loaded NVIDIA driver
# ABI, so building the adapter does not require a CUDA toolkit link dependency.
function(flagtree_debugger_enable_cuda target)
  target_compile_definitions(${target}
    PRIVATE
      FLAGTREE_DEBUGGER_HAS_CUDA_RUNTIME=1
  )
  if(CMAKE_DL_LIBS)
    target_link_libraries(${target} PRIVATE ${CMAKE_DL_LIBS})
  endif()
endfunction()
