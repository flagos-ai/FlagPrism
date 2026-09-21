add_profiler_library(FlagTreeProfilerBindings
  Profiler.cpp
)

if(FLAGPRISM_BACKEND STREQUAL "enflame")
  add_profiler_library(FlagTreeProfilerEnflame
    lib/Profiler/Vendor/EnflameProfiler.cpp
    lib/Profiler/Vendor/Adapter.cpp
    lib/Profiler/Vendor/Mode.cpp
  )
elseif(FLAGPRISM_BACKEND STREQUAL "mthreads")
  add_profiler_library(FlagTreeProfilerMthreads
    lib/Driver/Mthreads/MthreadsApi.cpp
    lib/Profiler/Vendor/Adapter.cpp
    lib/Profiler/Vendor/MthreadsAdapter.cpp
    lib/Profiler/Vendor/MthreadsProfiler.cpp
    lib/Profiler/Vendor/Mode.cpp
  )
elseif(FLAGPRISM_BACKEND STREQUAL "tianshu")
  add_profiler_library(FlagTreeProfilerTianshu
    lib/Profiler/Vendor/Adapter.cpp
    lib/Profiler/Vendor/TianshuAdapter.cpp
    lib/Profiler/Vendor/TianshuProfiler.cpp
    lib/Profiler/Vendor/Mode.cpp
  )
elseif(FLAGPRISM_BACKEND STREQUAL "ascend")
  add_profiler_library(FlagTreeProfilerCann
    lib/Driver/Ascend/AscendApi.cpp
    lib/Profiler/Vendor/Adapter.cpp
    lib/Profiler/Vendor/CannAdapter.cpp
    lib/Profiler/Vendor/CannProfiler.cpp
    lib/Profiler/Vendor/Mode.cpp
  )
elseif(FLAGPRISM_BACKEND STREQUAL "nvidia")
  # FlagPrism: NVIDIA vendor profiling reuses CUPTI's existing runtime
  # collector and adds only the adapter/importer layer here.
  add_profiler_library(FlagTreeProfilerNvidia
    lib/Profiler/Vendor/Adapter.cpp
    lib/Profiler/Vendor/NvidiaAdapter.cpp
    lib/Profiler/Vendor/Mode.cpp
  )
else()
  add_profiler_library(FlagTreeProfilerCann
    lib/Driver/Ascend/AscendApi.cpp
    lib/Profiler/Vendor/Adapter.cpp
    lib/Profiler/Vendor/CannAdapter.cpp
    lib/Profiler/Vendor/CannProfiler.cpp
    lib/Profiler/Vendor/TianshuAdapter.cpp
    lib/Profiler/Vendor/TianshuProfiler.cpp
    lib/Driver/Mthreads/MthreadsApi.cpp
    lib/Profiler/Vendor/MthreadsAdapter.cpp
    lib/Profiler/Vendor/MthreadsProfiler.cpp
    lib/Profiler/Vendor/NvidiaAdapter.cpp
    lib/Profiler/Vendor/Mode.cpp
  )
endif()

add_subdirectory(lib)
