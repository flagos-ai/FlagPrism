include(${CMAKE_CURRENT_SOURCE_DIR}/Runtime/Enflame.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/Runtime/CoreX.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/Runtime/CUDA.cmake)
if(NOT FLAGPRISM_BACKEND STREQUAL "tianshu")
  include(${CMAKE_CURRENT_SOURCE_DIR}/Runtime/CANN.cmake)
endif()
include(${CMAKE_CURRENT_SOURCE_DIR}/Runtime/MUSA.cmake)

add_triton_library(FlagTreeDebuggerRuntime
  Runtime/BufferLayout.cpp
  Runtime/TransferEngine.cpp
)
flagprism_enable_debugger_runtime(FlagTreeDebuggerRuntime)
flagprism_apply_backend_compile_definitions(FlagTreeDebuggerRuntime)

add_triton_library(FlagTreeDebuggerFrontend
  Frontend/Bridge.cpp
  Metadata/TrackedOpTable.cpp

  LINK_LIBS PUBLIC
  FlagTreeDebuggerRuntime
  LLVMSupport
)
flagprism_enable_debugger_runtime(FlagTreeDebuggerFrontend)
flagprism_apply_backend_compile_definitions(FlagTreeDebuggerFrontend)

add_library(FlagTreeDebuggerDecode STATIC
  Metadata/TrackedOpTable.cpp
  Runtime/BufferLayout.cpp
  Decode/Decoder.cpp
  Decode/Reporter.cpp
)
target_link_libraries(FlagTreeDebuggerDecode PUBLIC LLVMSupport)
llvm_update_compile_flags(FlagTreeDebuggerDecode)

add_triton_library(FlagTreeDebugger
  IR/Dialect.cpp
  IR/Ops.cpp
  Metadata/Passes.cpp
  Instrumentation/Collectors.cpp
  Instrumentation/Passes.cpp
  Instrumentation/RecordBuilder.cpp
  Instrumentation/Writer.cpp
  Decode/Decoder.cpp
  Decode/Reporter.cpp

  DEPENDS
  FlagTreeDebugTableGen

  LINK_LIBS PUBLIC
  FlagTreeDebuggerFrontend
  FlagTreeDebuggerRuntime
  MLIRArithDialect
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRPtrDialect
  MLIRTensorDialect
  TritonIR
  TritonGPUIR
)
flagprism_enable_debugger_runtime(FlagTreeDebugger)
flagprism_apply_backend_compile_definitions(FlagTreeDebugger)

foreach(_flagtree_debugger_target
        FlagTreeDebuggerRuntime
        FlagTreeDebuggerFrontend
        FlagTreeDebuggerDecode
        FlagTreeDebugger)
  target_include_directories(${_flagtree_debugger_target} PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include>"
  )
endforeach()
