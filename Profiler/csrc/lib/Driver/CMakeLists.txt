set(PROFILER_DRIVER_SOURCES Device.cpp)
if(FLAGTREE_PROFILER_GPU_RUNTIME)
  list(APPEND PROFILER_DRIVER_SOURCES
    GPU/CudaApi.cpp
    GPU/CuptiApi.cpp
    GPU/HipApi.cpp
    GPU/HsaApi.cpp
    GPU/RoctracerApi.cpp
  )
endif()

add_profiler_library(FlagTreeProfilerDriver ${PROFILER_DRIVER_SOURCES})
