include(${CMAKE_CURRENT_SOURCE_DIR}/Runtime/CoreX.cmake)
if(NOT FLAGPRISM_BACKEND STREQUAL "tianshu")
  include(${CMAKE_CURRENT_SOURCE_DIR}/Runtime/CANN.cmake)
endif()

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
)
flagprism_enable_debugger_runtime(FlagTreeDebugger)
flagprism_apply_backend_compile_definitions(FlagTreeDebugger)
