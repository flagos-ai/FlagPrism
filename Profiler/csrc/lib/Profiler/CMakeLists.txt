if(FLAGTREE_PROFILER_GPU_RUNTIME)
  set(PROFILER_RUNTIME_SOURCES
    Instrumentation/InstrumentationProfiler.cpp
    Instrumentation/Metadata.cpp
  )
  if(FLAGTREE_PROFILER_CUDA_RUNTIME)
    list(APPEND PROFILER_RUNTIME_SOURCES
      Cupti/CuptiPCSampling.cpp
      Cupti/CuptiProfiler.cpp
      Cupti/NvidiaHardwareCounters.cpp
      Instrumentation/CudaRuntime.cpp
    )
  endif()
  if(FLAGTREE_PROFILER_ROCTRACER_RUNTIME)
    list(APPEND PROFILER_RUNTIME_SOURCES
      RocTracer/RoctracerProfiler.cpp
      Instrumentation/HipRuntime.cpp
    )
  endif()
  add_profiler_library(FlagTreeProfilerRuntime
    ${PROFILER_RUNTIME_SOURCES}
  )
endif()
