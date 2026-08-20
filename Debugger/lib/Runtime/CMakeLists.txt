cmake_minimum_required(VERSION 3.20)

project(FlagTreeDebuggerRuntimeStandalone LANGUAGES CXX)

include(${CMAKE_CURRENT_SOURCE_DIR}/CoreX.cmake)
if(NOT FLAGPRISM_BACKEND STREQUAL "tianshu")
  include(${CMAKE_CURRENT_SOURCE_DIR}/CANN.cmake)
endif()

if(NOT COMMAND flagprism_enable_debugger_runtime)
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
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

option(FLAGTREE_DEBUGGER_RUNTIME_BUILD_SMOKE_TEST
       "Build a tiny standalone smoke test for the debugger runtime module"
       ON)
option(FLAGTREE_DEBUGGER_RUNTIME_BUILD_TRACE_DEMO
       "Build a standalone trace demo for the debugger runtime module"
       ON)
option(FLAGTREE_DEBUGGER_RUNTIME_BUILD_ASYNC_LOOP_TEST
       "Build a standalone async export loop test for the debugger runtime module"
       ON)

add_library(FlagTreeDebuggerRuntimeStandalone STATIC
  BufferLayout.cpp
  TransferEngine.cpp
)
flagprism_enable_debugger_runtime(FlagTreeDebuggerRuntimeStandalone)

target_include_directories(FlagTreeDebuggerRuntimeStandalone
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

if(FLAGTREE_DEBUGGER_RUNTIME_BUILD_SMOKE_TEST)
  enable_testing()

  add_executable(FlagTreeDebuggerRuntimeStandaloneSmokeTest
    SmokeTest.cpp
  )
  target_link_libraries(FlagTreeDebuggerRuntimeStandaloneSmokeTest
    PRIVATE
      FlagTreeDebuggerRuntimeStandalone
  )
  flagprism_enable_debugger_runtime(FlagTreeDebuggerRuntimeStandaloneSmokeTest)

  add_test(
    NAME FlagTreeDebuggerRuntimeStandaloneSmokeTest
    COMMAND FlagTreeDebuggerRuntimeStandaloneSmokeTest
  )
endif()

if(FLAGTREE_DEBUGGER_RUNTIME_BUILD_TRACE_DEMO)
  add_executable(FlagTreeDebuggerRuntimeTraceDemo
    RuntimeTraceDemo.cpp
  )
  target_link_libraries(FlagTreeDebuggerRuntimeTraceDemo
    PRIVATE
      FlagTreeDebuggerRuntimeStandalone
  )
  flagprism_enable_debugger_runtime(FlagTreeDebuggerRuntimeTraceDemo)
endif()

if(FLAGTREE_DEBUGGER_RUNTIME_BUILD_ASYNC_LOOP_TEST)
  add_executable(FlagTreeDebuggerRuntimeAsyncLoopTest
    AsyncExportLoopTest.cpp
  )
  target_link_libraries(FlagTreeDebuggerRuntimeAsyncLoopTest
    PRIVATE
      FlagTreeDebuggerRuntimeStandalone
  )
  flagprism_enable_debugger_runtime(FlagTreeDebuggerRuntimeAsyncLoopTest)
endif()
