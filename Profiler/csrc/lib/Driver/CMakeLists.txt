set(PROFILER_DRIVER_SOURCES
  Device.cpp
)
# FlagPrism：MetaX builds do not require CoreX device queries.
if(NOT FLAGPRISM_BACKEND STREQUAL "metax" AND
   NOT FLAGPRISM_BACKEND STREQUAL "enflame" AND
# FlagPrism end
   NOT FLAGPRISM_BACKEND STREQUAL "ascend" AND
   NOT FLAGPRISM_BACKEND STREQUAL "mthreads" AND
   NOT FLAGPRISM_BACKEND STREQUAL "nvidia")
  list(APPEND PROFILER_DRIVER_SOURCES Tianshu/TianshuApi.cpp)
endif()
if(FLAGTREE_PROFILER_CUDA_RUNTIME)
  list(APPEND PROFILER_DRIVER_SOURCES
    GPU/CudaApi.cpp
    GPU/CuptiApi.cpp
  )
endif()
if(FLAGTREE_PROFILER_ROCTRACER_RUNTIME)
  list(APPEND PROFILER_DRIVER_SOURCES
    GPU/HipApi.cpp
    GPU/HsaApi.cpp
    GPU/RoctracerApi.cpp
  )
endif()

add_profiler_library(FlagTreeProfilerDriver ${PROFILER_DRIVER_SOURCES})
